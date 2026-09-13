"""
agents/Self_Evolving/alpha_evolve.py
===================================
AlphaEvolve: Self-Evolving Training System with Parallel Simulation Coupling.

Orchestration:
1. High-Throughput Self-Play via SimulationRunner
2. Replay Buffer condition and telemetry logging
3. HiveMindPolicyValueNet GPU training
4. GeneticOptimizer deck evolution with strict chromosome legality repair
5. Champion promotion tournaments
"""
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.Resource_Management.simulation_runner import get_simulation_runner
from agents.Genetic_Algorithm.deck_optimizer import GeneticOptimizer
from agents.NN import get_hive_mind_net
from agents.Learning_System import get_replay_buffer
from agents.csv_data import get_csv_index

logger = logging.getLogger(__name__)


class AlphaEvolveTrainer:
    """End-to-End Self-Evolving Agent & Deck Evolution Training Loop."""

    def __init__(
        self,
        base_deck: List[int],
        opponent_decks: Optional[List[List[int]]] = None,
        population_size: int = 8,
        seed: int = 42
    ):
        self.csv_idx = get_csv_index(Path("data"))
        self.runner = get_simulation_runner()
        self.hive_mind = get_hive_mind_net()
        self.replay_buffer = get_replay_buffer()
        self.optimizer = GeneticOptimizer(
            base_deck=base_deck,
            population_size=population_size,
            seed=seed,
            csv_index=self.csv_idx
        )
        self.champion_deck = list(base_deck)
        self.generation = 0
        self.opponent_decks = opponent_decks or [list(base_deck)]

    def train_cycle(self, games_per_eval: int = 20, nn_epochs: int = 2) -> Dict[str, Any]:
        """Execute one complete evolutionary generation:
        1. Evaluate population against benchmark opponents via high-throughput SimulationRunner
        2. Assign empirical fitness scores
        3. Evolve GA population with legality repair
        4. Train Hive-Mind Neural Network on collected replay games
        5. Champion tournament promotion
        """
        t0 = time.perf_counter()
        pop = self.optimizer.population
        pop_results = []

        # 1. High-Throughput Simulation Evaluation for each individual
        for ind in pop:
            opp_deck = self.opponent_decks[0]
            metrics = self.runner.run_simulations(
                deck1=ind.deck,
                deck2=opp_deck,
                total_games=games_per_eval,
                collect_replays=True
            )
            ind.wins = metrics['wins_p1']
            ind.losses = metrics['wins_p2']
            ind.draws = metrics['draws']
            ind.games_played = metrics['completed_games']
            ind.win_rate = metrics['win_rate_p1']
            avg_turns = metrics.get('avg_turns', 25.0) or 25.0
            tempo_bonus = max(0.0, 30.0 - avg_turns) / 60.0
            ind.fitness = metrics['win_rate_p1'] + tempo_bonus

            pop_results.append({
                'agent_id': ind.agent_id,
                'win_rate': round(ind.win_rate, 4),
                'wins': ind.wins,
            })

        # 2. Advance Genetic Generation
        self.optimizer.evolve()
        best_current_deck = self.optimizer.get_best_deck()

        # 3. Train Hive-Mind on GPU
        nn_res = self.hive_mind.train_on_replays(epochs=nn_epochs)

        # 4. Champion Check
        if best_current_deck:
            champ_eval = self.runner.run_simulations(
                deck1=best_current_deck,
                deck2=self.champion_deck,
                total_games=games_per_eval,
                collect_replays=False
            )
            if champ_eval['win_rate_p1'] > 0.52:
                self.champion_deck = list(best_current_deck)
                promoted = True
            else:
                promoted = False
        else:
            promoted = False

        duration = round(time.perf_counter() - t0, 2)
        self.generation += 1

        return {
            'generation': self.generation,
            'duration_sec': duration,
            'population_stats': self.optimizer.get_stats(),
            'champion_promoted': promoted,
            'nn_training_summary': nn_res,
            'pop_sample_results': pop_results[:3],
        }
