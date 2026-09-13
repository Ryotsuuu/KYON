"""
agents/Learning_System/replay_buffer.py
======================================
Enriched Multi-Dimensional Replay Buffer with Dynamic Condition Queries.

Stores rich game trajectories, telemetry metrics, and state vectors.
Provides high-speed queries for NN training, ML card valuation, and RL learning.
"""
import os
import time
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


class EnrichedReplayBuffer:
    """Enriched Replay Buffer storing game trajectories, dynamic conditions, and state vectors."""

    def __init__(self, buffer_file: Optional[Path] = None, max_disk_capacity: int = 10000):
        self.buffer_file = buffer_file or Path("data") / "replay_buffer.json"
        self.max_disk_capacity = max_disk_capacity
        self.buffer_file.parent.mkdir(parents=True, exist_ok=True)
        self.games: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load persisted replay games from disk."""
        if self.buffer_file.exists():
            try:
                with open(self.buffer_file, 'r', encoding='utf-8') as f:
                    self.games = json.load(f)
            except Exception as e:
                logger.warning(f"Replay buffer load error: {e}")
                self.games = []

    def save(self) -> None:
        """Persist in-memory games to disk with rolling window cap."""
        try:
            if len(self.games) > self.max_disk_capacity:
                self.games = self.games[-self.max_disk_capacity:]

            with open(self.buffer_file, 'w', encoding='utf-8') as f:
                json.dump(self.games, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Replay buffer save error: {e}")

    def add_game(
        self,
        winner: Any = 0,
        deck1: Optional[List[int]] = None,
        deck2: Optional[List[int]] = None,
        total_turns: int = 1,
        agent_name: str = "P1",
        opponent_name: str = "P2",
        macro_summary: Optional[Dict[str, Any]] = None,
        states: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Record a completed match with full telemetry, accepting either a dict or positional arguments."""
        if isinstance(winner, dict):
            g = winner
            record = {
                'winner': g.get('winner', 0),
                'total_turns': g.get('total_turns', g.get('turns', 1)),
                'deck1': g.get('deck1', []),
                'deck2': g.get('deck2', []),
                'agent_name': g.get('agent_name', "P1"),
                'opponent_name': g.get('opponent_name', "P2"),
                'macro_summary': g.get('macro_summary', {}),
                'states': g.get('states', []),
                'timestamp': os.path.getmtime(self.buffer_file) if self.buffer_file.exists() else 0.0,
            }
        else:
            record = {
                'winner': winner,
                'total_turns': total_turns,
                'deck1': deck1 or [],
                'deck2': deck2 or [],
                'agent_name': agent_name,
                'opponent_name': opponent_name,
                'macro_summary': macro_summary or {},
                'states': states or [],
                'timestamp': os.path.getmtime(self.buffer_file) if self.buffer_file.exists() else 0.0,
            }
        self.games.append(record)

    def add(self, item: Any) -> None:
        """Alias for add_game to support flexible transition and replay ingestion."""
        if isinstance(item, dict):
            if 'states' in item:
                self.add_game(item)
            else:
                # Single transition frame: encapsulate into match record
                self.games.append({
                    'winner': item.get('winner', item.get('reward', 0)),
                    'total_turns': 1,
                    'deck1': item.get('deck1', []),
                    'deck2': item.get('deck2', []),
                    'agent_name': item.get('agent1', "P1"),
                    'opponent_name': item.get('agent2', "P2"),
                    'macro_summary': {},
                    'states': [item],
                    'timestamp': 0.0
                })
        else:
            self.add_game(winner=item)

    def get_card_win_rates(self) -> Dict[int, Dict[str, float]]:
        """Calculate empirical win rates, appearances, and impact for every card ID."""
        stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'total': 0})

        for g in self.games:
            w = g.get('winner', -1)
            d1 = set(g.get('deck1', []))
            d2 = set(g.get('deck2', []))

            for cid in d1:
                stats[cid]['total'] += 1
                if w == 0:
                    stats[cid]['wins'] += 1
                elif w == 1:
                    stats[cid]['losses'] += 1
                else:
                    stats[cid]['draws'] += 1

            for cid in d2:
                stats[cid]['total'] += 1
                if w == 1:
                    stats[cid]['wins'] += 1
                elif w == 0:
                    stats[cid]['losses'] += 1
                else:
                    stats[cid]['draws'] += 1

        results = {}
        for cid, data in stats.items():
            tot = data['total']
            wr = (data['wins'] + 0.5 * data['draws']) / max(1, tot)
            results[cid] = {
                'card_id': cid,
                'appearances': tot,
                'wins': data['wins'],
                'losses': data['losses'],
                'draws': data['draws'],
                'win_rate': round(wr, 4),
            }

        return results

    def get_cooccurrence_matrix(self) -> Dict[Tuple[int, int], int]:
        """Compute pairwise card co-occurrences across winning decks for Cards Matrix."""
        cooccurrences = defaultdict(int)
        for g in self.games:
            w = g.get('winner', -1)
            winning_deck = g.get('deck1', []) if w == 0 else (g.get('deck2', []) if w == 1 else [])
            unique_cards = sorted(list(set(winning_deck)))

            for i in range(len(unique_cards)):
                for j in range(i + 1, len(unique_cards)):
                    pair = (unique_cards[i], unique_cards[j])
                    cooccurrences[pair] += 1

        return dict(cooccurrences)

    def sample_batch(self, batch_size: int = 64) -> List[Dict[str, Any]]:
        """Sample random game records for batch learning."""
        if len(self.games) <= batch_size:
            return list(self.games)
        return random.sample(self.games, batch_size)

    def compute_td_lambda_returns(self, states: List[Dict[str, Any]], final_reward: float, gamma: float = 0.99, lambda_param: float = 0.95) -> List[float]:
        """
        Computes Temporal Difference TD(λ) bootstrapping returns for state transitions:
        G_t^λ = (1 - λ) * Σ_{n=1}^{T-t-1} λ^{n-1} G_{t:t+n} + λ^{T-t-1} G_t
        """
        T = len(states)
        if T == 0:
            return []
        
        returns = [0.0] * T
        returns[-1] = float(final_reward)
        for t in reversed(range(T - 1)):
            v_next = states[t + 1].get('value', 0.0) if isinstance(states[t + 1], dict) else 0.0
            returns[t] = gamma * ((1.0 - lambda_param) * v_next + lambda_param * returns[t + 1])
        return returns

    def sample_prioritized_batch(self, batch_size: int = 64, alpha: float = 0.6) -> List[Dict[str, Any]]:
        """
        Prioritized Experience Replay sampling emphasizing match turning points
        (inflections with large state value or HP differential shifts).
        """
        if not self.games:
            return []
        if len(self.games) <= batch_size:
            return list(self.games)

        # Assign priorities based on turning point indicators and match length
        priorities = []
        for g in self.games:
            states = g.get('states', [])
            turning_point_boost = 2.0 if len(states) >= 10 else 1.0
            p = max(0.1, (len(states) * turning_point_boost) ** alpha)
            priorities.append(p)

        total_p = sum(priorities)
        probs = [p / total_p for p in priorities]
        
        sampled_indices = random.choices(range(len(self.games)), weights=probs, k=batch_size)
        return [self.games[i] for i in sampled_indices]

    def export_to_perpetual_vault(self, vault_dir: Optional[Path] = None) -> Path:
        """Save an immutable versioned snapshot of the current replay buffer to the perpetual data lake."""
        v_dir = vault_dir or (ROOT_DIR / "data" / "replay_vault")
        v_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_file = v_dir / f"replays_{timestamp}_{len(self.games)}_games.json"
        try:
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(self.games, f, ensure_ascii=False)
            logger.info(f"Exported {len(self.games)} replay games to perpetual vault: {out_file}")
        except Exception as e:
            logger.warning(f"Could not export replay vault: {e}")
        return out_file

    def load_from_perpetual_vault(self, vault_dir: Optional[Path] = None) -> int:
        """Load and merge all historical replay datasets from perpetual data vault."""
        v_dir = vault_dir or (ROOT_DIR / "data" / "replay_vault")
        if not v_dir.exists():
            return 0
        loaded_count = 0
        for fpath in sorted(v_dir.glob("replays_*.json")):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.games.extend(data)
                        loaded_count += len(data)
            except Exception:
                pass
        return loaded_count

    def __len__(self) -> int:
        return len(self.games)


_replay_buffer_instance: Optional[EnrichedReplayBuffer] = None


def get_replay_buffer(data_dir: Optional[Path] = None) -> EnrichedReplayBuffer:
    global _replay_buffer_instance
    if _replay_buffer_instance is None:
        path = (data_dir or Path("data")) / "replay_buffer.json"
        _replay_buffer_instance = EnrichedReplayBuffer(path)
    return _replay_buffer_instance
