"""
agents/Evaluation_System/elo_benchmark.py
=========================================
Continuous Background ELO Rating Benchmark & Matchmaking Engine.

Implements competitive TrueSkill/ELO ratings across all candidate agents:
R'_A = R_A + K * (S_A - E_A)
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
"""
import os
import json
import math
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = ROOT_DIR / "ptcg-system" / "agents_registry.json"


class EloBenchmarkEngine:
    """Computes and maintains empirical ELO ratings across the agent catalog."""

    DEFAULT_ELO = 1500.0
    K_FACTOR = 32.0

    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or REGISTRY_PATH

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Returns expected score E_A in [0, 1]."""
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def update_elo(self, rating_a: float, rating_b: float, score_a: float, k: Optional[float] = None) -> Tuple[float, float]:
        """
        Updates ELO ratings given actual match score_a (1.0 for win, 0.5 for draw, 0.0 for loss).
        Returns (new_rating_a, new_rating_b).
        """
        k_val = k or self.K_FACTOR
        expected_a = self.calculate_expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        score_b = 1.0 - score_a

        new_a = rating_a + k_val * (score_a - expected_a)
        new_b = rating_b + k_val * (score_b - expected_b)
        return round(new_a, 2), round(new_b, 2)

    def run_auto_benchmark(
        self,
        top_k: int = 16,
        rounds: int = 3,
        games_per_match: int = 5,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Runs an automated round-robin tournament across the top-K active or selected agents,
        updating their persistent ELO ratings and W-L-D records in agents_registry.json.
        """
        from ptcg import _load_registry, _get_agent_deck, _record_batch_results
        from agents.Resource_Management.simulation_runner import get_simulation_runner

        reg = _load_registry()
        if not reg:
            return {"error": "Registry empty"}

        # Select candidate pool
        all_agents = list(reg.keys())
        # Sort by games played and existing ELO
        all_agents.sort(
            key=lambda a: (reg[a].get('elo', self.DEFAULT_ELO), reg[a].get('games', 0)),
            reverse=True
        )
        pool = all_agents[:max(4, min(top_k, len(all_agents)))]

        runner = get_simulation_runner()
        total_matches = rounds * (len(pool) * (len(pool) - 1) // 2)
        match_idx = 0

        logger.info(f"Starting ELO Auto-Benchmark across {len(pool)} agents ({total_matches} total match series)...")

        for r in range(rounds):
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    a1 = pool[i]
                    a2 = pool[j]

                    d1 = _get_agent_deck(a1)
                    d2 = _get_agent_deck(a2)
                    if not d1 or not d2:
                        continue

                    res = runner.run_simulations(
                        d1, d2, total_games=games_per_match, collect_replays=False,
                        agent1_config={'name': a1}, agent2_config={'name': a2}
                    )

                    w1 = res.get('wins_p1', 0)
                    w2 = res.get('wins_p2', 0)
                    dr = res.get('draws', 0)
                    tot = max(1, w1 + w2 + dr)

                    # Fractional actual score for Player 1
                    actual_score_p1 = (w1 + 0.5 * dr) / tot

                    r1 = reg.get(a1, {}).get('elo', self.DEFAULT_ELO)
                    r2 = reg.get(a2, {}).get('elo', self.DEFAULT_ELO)

                    new_r1, new_r2 = self.update_elo(r1, r2, actual_score_p1)

                    if a1 in reg:
                        reg[a1]['elo'] = new_r1
                    if a2 in reg:
                        reg[a2]['elo'] = new_r2

                    _record_batch_results(a1, a2, w1, w2, dr, sim_type="benchmark")
                    match_idx += 1

                    if progress_callback:
                        progress_callback(match_idx, total_matches, a1, a2, new_r1, new_r2)

        # Save updated ELOs
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(reg, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist ELO updates: {e}")

        # Ingest benchmark replays into Neural Network & ML Models
        training_res = {}
        try:
            from agents.Learning_System.replay_buffer import get_replay_buffer
            from agents.NN.policy_value_net import get_hive_mind_net
            from agents.ML.card_value_model import get_card_value_model
            from agents.ML.cards_matrix import get_cards_matrix

            rb = get_replay_buffer()
            net = get_hive_mind_net()
            ml_model = get_card_value_model()
            pmi_matrix = get_cards_matrix()

            if len(rb) >= 1:
                training_res = net.train_on_replays(epochs=2)
                ml_model.train(rb)
                pmi_matrix.compute_from_replays(rb)
        except Exception as e:
            logger.warning(f"Benchmark GPU auto-training warning: {e}")

        # Return sorted leaderboard
        leaderboard = []
        for aid in pool:
            meta = reg.get(aid, {})
            leaderboard.append({
                "agent_id": aid,
                "elo": meta.get('elo', self.DEFAULT_ELO),
                "games": meta.get('games', 0),
                "wins": meta.get('wins', 0),
                "losses": meta.get('losses', 0),
                "draws": meta.get('draws', 0),
                "win_rate": round(meta.get('wins', 0) / max(1, meta.get('games', 1)), 4)
            })

        leaderboard.sort(key=lambda x: x['elo'], reverse=True)
        return {
            "leaderboard": leaderboard,
            "total_matches_played": match_idx,
            "training_result": training_res,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


_elo_engine: Optional[EloBenchmarkEngine] = None


def get_elo_engine() -> EloBenchmarkEngine:
    global _elo_engine
    if _elo_engine is None:
        _elo_engine = EloBenchmarkEngine()
    return _elo_engine
