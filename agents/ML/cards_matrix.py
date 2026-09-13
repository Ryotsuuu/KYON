"""
agents/ML/cards_matrix.py
=========================
Cards Matrix: Pointwise Mutual Information (PMI) & Card Relationship Analytics.

Calculates statistical synergy between cards:
  PMI(c1, c2) = ln( P(c1, c2) / (P(c1) * P(c2)) )
Positive PMI indicates cards that win together far more than independent chance.
"""
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from itertools import combinations

from cg.api import all_card_data
from agents.Learning_System.replay_buffer import get_replay_buffer

logger = logging.getLogger(__name__)


class CardsMatrix:
    """Computes and stores pairwise card synergy, evolution links, and counter-matchups."""

    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file or Path("data") / "cards_matrix.json"
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.pmi_scores: Dict[Tuple[int, int], float] = {}
        self.card_frequencies: Dict[int, int] = {}
        self._load()

    def _load(self):
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    self.pmi_scores = {
                        tuple(map(int, k.split(','))): v
                        for k, v in raw.get('pmi_scores', {}).items()
                    }
                    self.card_frequencies = {
                        int(k): v for k, v in raw.get('card_frequencies', {}).items()
                    }
            except Exception as e:
                logger.debug(f"CardsMatrix load exception: {e}")

    def save(self):
        try:
            serialized = {
                'pmi_scores': {f"{k[0]},{k[1]}": v for k, v in self.pmi_scores.items()},
                'card_frequencies': self.card_frequencies,
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(serialized, f, indent=2)
        except Exception as e:
            logger.warning(f"CardsMatrix save exception: {e}")

    def compute_from_replays(self, replay_buffer=None) -> Dict[str, Any]:
        """Compute Pointwise Mutual Information (PMI) across winning deck profiles and champion registries."""
        rb = replay_buffer or get_replay_buffer()
        try:
            rb.load_from_perpetual_vault()
        except Exception:
            pass

        card_freq = Counter()
        pair_freq = Counter()
        total_profiles = 0

        # 1. Ingest winning match decks from replay buffer
        winning_games = [g for g in rb.games if g.get('winner') in (0, 1)]
        for g in winning_games:
            w = g.get('winner')
            deck = g.get('deck1', []) if w == 0 else g.get('deck2', [])
            if not deck and 'states' in g:
                # Extract cards that appeared in winning game states
                seen_in_game = set()
                for st in g['states']:
                    if isinstance(st, dict):
                        p = st.get('players', [{}, {}])[w] if len(st.get('players', [])) > w else {}
                        for act in p.get('active', []):
                            if isinstance(act, dict) and 'cardId' in act:
                                seen_in_game.add(act['cardId'])
                        for b in p.get('bench', []):
                            if isinstance(b, dict) and 'cardId' in b:
                                seen_in_game.add(b['cardId'])
                deck = list(seen_in_game)

            unique_cards = sorted(list(set(deck)))
            if len(unique_cards) >= 2:
                total_profiles += 1
                for cid in unique_cards:
                    card_freq[cid] += 1
                for c1, c2 in combinations(unique_cards, 2):
                    pair = (min(c1, c2), max(c1, c2))
                    pair_freq[pair] += 1

        # 2. Ingest 521 tournament-legal champion decks from agents_registry.json
        reg_file = Path(__file__).resolve().parent.parent.parent / "ptcg-system" / "agents_registry.json"
        if reg_file.exists():
            try:
                with open(reg_file, 'r', encoding='utf-8') as f:
                    reg_data = json.load(f)
                for aid, ainfo in reg_data.items():
                    d = ainfo.get('deck', [])
                    u_cards = sorted(list(set(d)))
                    if len(u_cards) >= 2:
                        total_profiles += 1
                        for cid in u_cards:
                            card_freq[cid] += 1
                        for c1, c2 in combinations(u_cards, 2):
                            pair = (min(c1, c2), max(c1, c2))
                            pair_freq[pair] += 1
            except Exception as e:
                logger.debug(f"Could not load agents_registry in CardsMatrix: {e}")

        if total_profiles < 5:
            return {'status': 'insufficient_data', 'winning_games': total_profiles}

        self.card_frequencies = dict(card_freq)
        self.pmi_scores = {}

        for (c1, c2), count in pair_freq.most_common(5000):
            p_joint = count / total_profiles
            p_c1 = card_freq[c1] / total_profiles
            p_c2 = card_freq[c2] / total_profiles
            expected = p_c1 * p_c2
            if expected > 0:
                pmi = math.log((p_joint / expected) + 1e-9)
                self.pmi_scores[(c1, c2)] = round(pmi, 4)

        self.save()
        # Mirror save to csv-data if it exists
        csv_p = Path(__file__).resolve().parent.parent.parent / "csv-data" / "cards_matrix.json"
        if csv_p.parent.exists():
            try:
                serialized = {
                    'pmi_scores': {f"{k[0]},{k[1]}": v for k, v in self.pmi_scores.items()},
                    'card_frequencies': self.card_frequencies,
                }
                with open(csv_p, 'w', encoding='utf-8') as f:
                    json.dump(serialized, f, indent=2)
            except Exception:
                pass

        return {
            'status': 'success',
            'winning_games_analyzed': len(winning_games),
            'profiles_analyzed': total_profiles,
            'synergy_pairs_computed': len(self.pmi_scores),
        }

    def get_synergy(self, card1: int, card2: int) -> float:
        """Get the PMI synergy score between two cards."""
        pair = (min(card1, card2), max(card1, card2))
        return self.pmi_scores.get(pair, 0.0)

    def get_best_partners(self, card_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get the top-K synergistic card partners for a given card ID."""
        partners = []
        for (c1, c2), pmi in self.pmi_scores.items():
            if c1 == card_id:
                partners.append((c2, pmi))
            elif c2 == card_id:
                partners.append((c1, pmi))
        partners.sort(key=lambda x: -x[1])
        return partners[:top_k]


_cards_matrix_instance: Optional[CardsMatrix] = None


def get_cards_matrix() -> CardsMatrix:
    global _cards_matrix_instance
    if _cards_matrix_instance is None:
        _cards_matrix_instance = CardsMatrix()
    return _cards_matrix_instance

