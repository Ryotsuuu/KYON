"""
agents/Resource_Management/simulation_runner.py
==============================================
High-Throughput Parallel Simulation Engine with Auto-Streaming & Memory Guard.

Performance Targets:
- 50 to 250+ simulations in 10 seconds
- Single Agent: 100 to 5,000 games in streaming batches
- Multi-Agent: up to 2,000 games
- Hardware Concurrency: Multi-worker thread pool (utilizing 12 CPU cores) + GPU tensor evaluation
- Memory Safety: 95% RAM Cap enforcement via MemoryGuard with auto-disk streaming
"""
import os
import sys
import json
import time
import math
import ctypes
import random
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from cg.game import battle_start, battle_select, battle_finish
from simulation.Decision_Engine import MasterAgent
from agents.Resource_Management.hardware_manager import get_hardware_manager
from agents.Resource_Management.memory_guard import get_memory_guard
from agents.GPU_config import configure_gpu

logger = logging.getLogger(__name__)


def _calculate_wilson_ci(wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate the Wilson Score 95% Confidence Interval for win rate."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence z-score
    p = wins / total
    denominator = 1 + (z ** 2) / total
    centre_adjusted = p + (z ** 2) / (2 * total)
    adjusted_spread = z * math.sqrt((p * (1 - p) + (z ** 2) / (4 * total)) / total)
    lower = max(0.0, (centre_adjusted - adjusted_spread) / denominator)
    upper = min(1.0, (centre_adjusted + adjusted_spread) / denominator)
    return round(lower, 4), round(upper, 4)


import statistics


def _simulate_single_game(
    deck1: List[int],
    deck2: List[int],
    agent1_kwargs: Optional[dict] = None,
    agent2_kwargs: Optional[dict] = None,
    max_turns: int = 200,
    collect_replay: bool = False
) -> Dict[str, Any]:
    """Simulate a single battle thread-safely with seat and turn analytics."""
    agent1 = MasterAgent(deck=deck1, config=agent1_kwargs or {})
    agent2 = MasterAgent(deck=deck2, config=agent2_kwargs or {})

    game_record = {
        'winner': -1,  # 0=deck1, 1=deck2, 2=draw
        'turns': 0,
        'first_player': 0,
        'duration_ms': 0.0,
        'replay_states': [] if collect_replay else None,
    }

    t0 = time.perf_counter()
    obs, sd = battle_start(deck1, deck2)

    if obs is None:
        battle_finish()
        game_record['winner'] = 2
        game_record['duration_ms'] = (time.perf_counter() - t0) * 1000
        return game_record

    try:
        first_p = -1
        turns = 0
        while obs and obs.get('select') is not None and turns < max_turns:
            cur = obs.get('current', {}) if obs else {}
            if first_p not in (0, 1):
                fp = cur.get('firstPlayer', -1)
                if fp in (0, 1):
                    first_p = fp
                elif cur.get('turn', 0) >= 1:
                    first_p = cur.get('yourIndex', 0)

            p_idx = cur.get('yourIndex', 0) if cur else 0
            active_agent = agent1 if p_idx == 0 else agent2
            selection = active_agent(obs)
            if not selection:
                selection = []

            if collect_replay and len(game_record['replay_states']) < 2:
                game_record['replay_states'].append(cur)

            try:
                obs = battle_select(selection)
            except Exception:
                break
            turns += 1

        if first_p not in (0, 1):
            first_p = random.choice([0, 1])

        game_record['first_player'] = first_p
        game_record['turns'] = turns

        # Determine winner from final state
        if obs and 'current' in obs:
            current = obs['current']
            result = current.get('result', -1)
            if result in (0, 1):
                game_record['winner'] = result
            else:
                players = current.get('players', [])
                if len(players) == 2:
                    p0_prizes = players[0].get('prize', [])
                    p1_prizes = players[1].get('prize', [])
                    p0_remaining = sum(1 for p in p0_prizes if p is not None)
                    p1_remaining = sum(1 for p in p1_prizes if p is not None)
                    if p0_remaining == 0:
                        game_record['winner'] = 0
                    elif p1_remaining == 0:
                        game_record['winner'] = 1
                    else:
                        game_record['winner'] = 2

    except Exception as e:
        logger.warning(f"Battle loop error: {e}")
        game_record['winner'] = 2

    finally:
        battle_finish()

    game_record['duration_ms'] = (time.perf_counter() - t0) * 1000
    return game_record


class SimulationRunner:
    """Orchestrates high-throughput simulations with automatic memory safety streaming."""

    SINGLE_AGENT_MAX_LIMIT = 5000
    MULTI_AGENT_MAX_LIMIT = 2000

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.hw_manager = get_hardware_manager()
        self.memory_guard = get_memory_guard()
        self.output_file = self.data_dir / "simulation_results.json"
        self.replay_file = self.data_dir / "replay_buffer.json"

        # Register auto-flush callback
        self._in_memory_replays: List[dict] = []
        self._replay_lock = threading.Lock()
        self.memory_guard.register_flush_callback(self.flush_replays_to_disk)

    def flush_replays_to_disk(self) -> int:
        """Flush in-memory replays to disk with fast write."""
        with self._replay_lock:
            if not self._in_memory_replays:
                return 0
            count = len(self._in_memory_replays)
            try:
                # Keep max 500 in-memory items
                to_save = list(self._in_memory_replays[-500:])
                with open(self.replay_file, 'w', encoding='utf-8') as f:
                    json.dump(to_save, f, ensure_ascii=False)
                self._in_memory_replays.clear()
                return count
            except Exception as e:
                logger.warning(f"Error flushing replays to disk: {e}")
                return 0

    def run_simulations(
        self,
        deck1: List[int],
        deck2: List[int],
        total_games: int = 100,
        agent1_config: Optional[dict] = None,
        agent2_config: Optional[dict] = None,
        chunk_size: int = 50,
        max_workers: Optional[int] = None,
        collect_replays: bool = False,
        allow_exceed_limits: bool = False,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None,
    ) -> Dict[str, Any]:
        """Execute parallel batch simulations with streaming auto-save and 95% RAM protection.

        Args:
            deck1: 60-card list for Player 1
            deck2: 60-card list for Player 2
            total_games: Number of simulation games (default 100, up to 5000)
            agent1_config: Archetype/hyperparameter config for agent 1
            agent2_config: Archetype/hyperparameter config for agent 2
            chunk_size: Streaming batch size for disk persistence (default 50)
            max_workers: Worker thread count (auto-tuned if None)
            collect_replays: Store game states for GPU NN training
            allow_exceed_limits: Override 5k / 2k ceiling if explicitly authorized
            progress_callback: Callback fn(completed, total, current_stats)

        Returns:
            Dict containing aggregated metrics, win rates, Wilson CIs, throughput.
        """
        # Hardcore limit validation
        if not allow_exceed_limits and total_games > self.SINGLE_AGENT_MAX_LIMIT:
            logger.warning(f"Requested {total_games} games exceeds single-agent limit ({self.SINGLE_AGENT_MAX_LIMIT}). Clamping to {self.SINGLE_AGENT_MAX_LIMIT}.")
            total_games = self.SINGLE_AGENT_MAX_LIMIT

        workers = max_workers or self.hw_manager.get_optimal_worker_count()
        gpu_config = configure_gpu()

        metrics = {
            'total_games': total_games,
            'completed_games': 0,
            'wins_p1': 0,
            'wins_p2': 0,
            'draws': 0,
            'win_rate_p1': 0.0,
            'win_rate_p2': 0.0,
            'draw_rate': 0.0,
            'wilson_ci_95_p1': (0.0, 0.0),
            'avg_turns': 0.0,
            'mean_turns': 0.0,
            'median_turns': 0.0,
            'mode_turns': [],
            'std_turns': 0.0,
            'min_turns': 0,
            'max_turns': 0,
            'p25_turns': 0.0,
            'p75_turns': 0.0,
            'iqr_turns': 0.0,
            'p1_first_count': 0,
            'p1_first_wins': 0,
            'p2_first_count': 0,
            'p2_first_wins': 0,
            'p1_win_rate_when_first': 0.0,
            'p1_win_rate_when_second': 0.0,
            'first_player_win_rate': 0.0,
            'total_duration_sec': 0.0,
            'games_per_second': 0.0,
            'simulations_in_10s': 0.0,
            'workers_used': workers,
            'gpu_accelerator': gpu_config['device_name'],
            'ram_used_percent': self.memory_guard.get_memory_stats()['ram_used_percent'],
        }

        turns_list = []
        turns_p1 = []
        turns_p2 = []
        start_time = time.perf_counter()

        # Run simulations in streamed chunks to prevent RAM accumulation
        games_remaining = total_games
        while games_remaining > 0:
            # Memory safety check (95% RAM Cap)
            self.memory_guard.check_and_enforce()

            current_chunk = min(chunk_size, games_remaining)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {}
                for g_i in range(current_chunk):
                    swap = (g_i % 2 == 1)
                    if swap:
                        fut = executor.submit(
                            _simulate_single_game,
                            deck2, deck1,
                            agent2_config, agent1_config,
                            200, collect_replays
                        )
                    else:
                        fut = executor.submit(
                            _simulate_single_game,
                            deck1, deck2,
                            agent1_config, agent2_config,
                            200, collect_replays
                        )
                    future_map[fut] = swap

                for future in as_completed(future_map):
                    swap = future_map[future]
                    try:
                        record = future.result()
                        metrics['completed_games'] += 1
                        turns_val = record.get('turns', 0)
                        turns_list.append(turns_val)

                        raw_winner = record.get('winner', 2)
                        if swap:
                            winner = 1 if raw_winner == 0 else (0 if raw_winner == 1 else 2)
                            agent1_was_first = False
                        else:
                            winner = raw_winner
                            agent1_was_first = True

                        if agent1_was_first:
                            metrics['p1_first_count'] += 1
                            if winner == 0:
                                metrics['p1_first_wins'] += 1
                        else:
                            metrics['p2_first_count'] += 1
                            if winner == 1:
                                metrics['p2_first_wins'] += 1

                        if winner == 0:
                            metrics['wins_p1'] += 1
                            turns_p1.append(turns_val)
                        elif winner == 1:
                            metrics['wins_p2'] += 1
                            turns_p2.append(turns_val)
                        else:
                            metrics['draws'] += 1

                        if collect_replays and record.get('replay_states'):
                            with self._replay_lock:
                                self._in_memory_replays.append({
                                    'winner': winner,
                                    'turns': record['turns'],
                                    'states': record['replay_states'],
                                })

                        if progress_callback:
                            progress_callback(metrics['completed_games'], total_games, metrics)

                    except Exception as e:
                        logger.warning(f"Simulation worker exception: {e}")

            games_remaining -= current_chunk

            # Auto-save chunk replays to disk to release memory
            if collect_replays and len(self._in_memory_replays) >= chunk_size:
                self.flush_replays_to_disk()

        elapsed_sec = max(0.001, time.perf_counter() - start_time)
        completed = max(1, metrics['completed_games'])

        a1_name = agent1_config.get('name', 'Player 1') if agent1_config else 'Player 1'
        a2_name = agent2_config.get('name', 'Player 2') if agent2_config else 'Player 2'
        metrics['agent1_name'] = a1_name
        metrics['agent2_name'] = a2_name

        metrics['total_duration_sec'] = round(elapsed_sec, 3)
        metrics['win_rate_p1'] = round(metrics['wins_p1'] / completed, 4)
        metrics['win_rate_p2'] = round(metrics['wins_p2'] / completed, 4)
        metrics['draw_rate'] = round(metrics['draws'] / completed, 4)
        metrics['wilson_ci_95_p1'] = _calculate_wilson_ci(metrics['wins_p1'], completed)

        # Full Statistical Distribution
        if turns_list:
            metrics['mean_turns'] = round(statistics.mean(turns_list), 2)
            metrics['avg_turns'] = metrics['mean_turns']
            metrics['median_turns'] = round(statistics.median(turns_list), 2)
            try:
                metrics['mode_turns'] = statistics.multimode(turns_list)[:3]
            except Exception:
                metrics['mode_turns'] = [int(metrics['median_turns'])]
            metrics['std_turns'] = round(statistics.stdev(turns_list), 2) if len(turns_list) >= 2 else 0.0
            metrics['min_turns'] = min(turns_list)
            metrics['max_turns'] = max(turns_list)

            # Quantiles / IQR
            sorted_t = sorted(turns_list)
            p25_idx = int(len(sorted_t) * 0.25)
            p75_idx = int(len(sorted_t) * 0.75)
            metrics['p25_turns'] = sorted_t[p25_idx]
            metrics['p75_turns'] = sorted_t[p75_idx]
            metrics['iqr_turns'] = round(metrics['p75_turns'] - metrics['p25_turns'], 2)

        # Per-Agent Victory Pacing
        metrics['p1_win_turns_mean'] = round(statistics.mean(turns_p1), 2) if turns_p1 else 0.0
        metrics['p1_win_turns_min'] = min(turns_p1) if turns_p1 else 0
        metrics['p1_win_turns_max'] = max(turns_p1) if turns_p1 else 0
        metrics['p2_win_turns_mean'] = round(statistics.mean(turns_p2), 2) if turns_p2 else 0.0
        metrics['p2_win_turns_min'] = min(turns_p2) if turns_p2 else 0
        metrics['p2_win_turns_max'] = max(turns_p2) if turns_p2 else 0

        # Seat / First-Player Advantage Analytics
        p1_first_cnt = max(1, metrics['p1_first_count'])
        p2_first_cnt = max(1, metrics['p2_first_count'])
        metrics['p1_win_rate_when_first'] = round(metrics['p1_first_wins'] / p1_first_cnt, 4)
        p1_sec_cnt = max(1, completed - metrics['p1_first_count'])
        p1_sec_wins = metrics['wins_p1'] - metrics['p1_first_wins']
        metrics['p1_win_rate_when_second'] = round(p1_sec_wins / p1_sec_cnt, 4)

        metrics['p2_win_rate_when_first'] = round(metrics['p2_first_wins'] / p2_first_cnt, 4)
        p2_sec_cnt = max(1, completed - metrics['p2_first_count'])
        p2_sec_wins = metrics['wins_p2'] - metrics['p2_first_wins']
        metrics['p2_win_rate_when_second'] = round(p2_sec_wins / p2_sec_cnt, 4)

        total_first_wins = metrics['p1_first_wins'] + metrics['p2_first_wins']
        metrics['first_player_win_rate'] = round(total_first_wins / completed, 4)

        metrics['games_per_second'] = round(completed / elapsed_sec, 2)
        metrics['simulations_in_10s'] = round(metrics['games_per_second'] * 10, 1)
        metrics['ram_used_percent'] = self.memory_guard.get_memory_stats()['ram_used_percent']

        # Final disk flush
        if collect_replays:
            self.flush_replays_to_disk()

        # Save summary report
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass

        # Save immutable structured simulation run record with proper timestamp-based naming
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            run_dir = self.data_dir / "simulation_runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            safe_a1 = str(a1_name).replace(' ', '_').replace('/', '_')
            safe_a2 = str(a2_name).replace(' ', '_').replace('/', '_')
            run_file = run_dir / f"sim_{safe_a1}_vs_{safe_a2}_{timestamp_str}_{completed}g.json"
            with open(run_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            metrics['saved_run_file'] = str(run_file)
        except Exception:
            pass

        return metrics


_simulation_runner: Optional[SimulationRunner] = None


def get_simulation_runner(data_dir: Optional[Path] = None) -> SimulationRunner:
    global _simulation_runner
    if _simulation_runner is None:
        _simulation_runner = SimulationRunner(data_dir)
    return _simulation_runner
