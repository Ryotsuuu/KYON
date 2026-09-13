"""
agents/ML/card_value_model.py
=============================
Machine Learning Operator: RandomForest Card Value Prediction & Ranking Subsystem.

Learns non-linear card values from empirical simulation win rates stored in ReplayBuffer.
Provides value estimation for deck mutation, GA fitness evaluation, and draft prioritization.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from cg.api import all_card_data, all_attack, CardType, EnergyType

logger = logging.getLogger(__name__)

ALL_ENERGY_TYPES = [
    "Grass", "Fire", "Water", "Lightning", "Psychic",
    "Fighting", "Darkness", "Metal", "Colorless", "Dragon"
]

FEATURE_NAMES = (
    ["hp", "retreat_cost", "prize_value", "ability_count", "attack_count",
     "mean_attack_damage", "mean_attack_cost"]
    + [f"type_{t}" for t in ALL_ENERGY_TYPES]
    + ["is_basic", "is_stage1", "is_stage2", "is_ex", "is_mega_ex",
       "is_ace_spec", "is_tera", "is_fossil", "has_rule_box"]
)
FEATURE_DIM = len(FEATURE_NAMES)


def card_to_features(card: Any, attack_table: Optional[dict] = None) -> np.ndarray:
    """Extract a normalized 28-dimensional feature vector from a card."""
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    feats[0] = (card.hp or 0) / 400.0
    feats[1] = (card.retreatCost or 0) / 4.0
    pv = 3 if getattr(card, 'megaEx', False) else (2 if getattr(card, 'ex', False) else 1)
    feats[2] = pv / 3.0
    feats[3] = len(getattr(card, 'skills', []))
    feats[4] = len(getattr(card, 'attacks', []))

    if attack_table is None:
        try:
            attack_table = {a.attackId: a for a in all_attack()}
        except Exception:
            attack_table = {}

    if getattr(card, 'attacks', None) and attack_table:
        damages = [attack_table[aid].damage for aid in card.attacks if aid in attack_table]
        costs = [len(attack_table[aid].energies) for aid in card.attacks if aid in attack_table]
        if damages:
            feats[5] = (sum(damages) / len(damages)) / 350.0
        if costs:
            feats[6] = (sum(costs) / len(costs)) / 5.0

    etype_map = {t: i for i, t in enumerate(ALL_ENERGY_TYPES)}
    t_name = str(card.energyType).split(".")[-1] if hasattr(card.energyType, 'name') else str(card.energyType)
    if t_name in etype_map:
        feats[7 + etype_map[t_name]] = 1.0

    offset = 7 + len(ALL_ENERGY_TYPES)
    feats[offset + 0] = 1.0 if getattr(card, 'basic', False) else 0.0
    feats[offset + 1] = 1.0 if getattr(card, 'stage1', False) else 0.0
    feats[offset + 2] = 1.0 if getattr(card, 'stage2', False) else 0.0
    feats[offset + 3] = 1.0 if getattr(card, 'ex', False) else 0.0
    feats[offset + 4] = 1.0 if getattr(card, 'megaEx', False) else 0.0
    feats[offset + 5] = 1.0 if getattr(card, 'aceSpec', False) else 0.0
    feats[offset + 6] = 1.0 if getattr(card, 'tera', False) else 0.0
    feats[offset + 7] = 1.0 if "fossil" in getattr(card, 'name', '').lower() else 0.0
    feats[offset + 8] = 1.0 if (getattr(card, 'ex', False) or getattr(card, 'megaEx', False)) else 0.0

    return feats


class CardValueModel:
    """RandomForest model predicting card utility and empirical win impact."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("models") / "card_value_model.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.cards_cache = None
        self.attacks_cache = None
        self._load_model()

    def _get_cards(self):
        if self.cards_cache is None:
            try:
                self.cards_cache = {c.cardId: c for c in all_card_data()}
            except Exception:
                self.cards_cache = {}
        return self.cards_cache

    def _get_attacks(self):
        if self.attacks_cache is None:
            try:
                self.attacks_cache = {a.attackId: a for a in all_attack()}
            except Exception:
                self.attacks_cache = {}
        return self.attacks_cache

    def _load_model(self):
        if _JOBLIB_AVAILABLE and self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                logger.debug(f"CardValueModel load exception: {e}")

    def save_model(self):
        if _JOBLIB_AVAILABLE and self.model is not None:
            try:
                joblib.dump(self.model, self.model_path)
            except Exception as e:
                logger.warning(f"CardValueModel save exception: {e}")

    def build_training_data(self, replay_buffer=None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate (X, y) feature and target matrix from replay win-rates or heuristic baseline."""
        cards = self._get_cards()
        attacks = self._get_attacks()
        from agents.Learning_System.replay_buffer import get_replay_buffer
        rb = replay_buffer or get_replay_buffer()

        card_stats = rb.get_card_win_rates()
        has_replays = len(card_stats) >= 10

        X_list = []
        y_list = []

        for cid, card in cards.items():
            feats = card_to_features(card, attacks)
            X_list.append(feats)

            if has_replays and cid in card_stats and card_stats[cid]['appearances'] >= 3:
                target_val = card_stats[cid]['win_rate']
            else:
                # Heuristic baseline
                base = (card.hp or 0) / 300.0
                if card.attacks:
                    damages = [attacks[aid].damage for aid in card.attacks if aid in attacks]
                    if damages:
                        base += max(damages) / 220.0
                if getattr(card, 'ex', False):
                    base += 0.4
                if getattr(card, 'megaEx', False):
                    base += 0.7
                if getattr(card, 'skills', None):
                    base += 0.25 * len(card.skills)
                if getattr(card, 'aceSpec', False):
                    base += 0.5
                target_val = max(0.1, min(1.0, base / 2.5))

            y_list.append(target_val)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def train(self, replay_buffer=None, n_estimators: int = 100) -> Dict[str, Any]:
        """Train RandomForest regressor on card features against empirical win rates."""
        if not _SKLEARN_AVAILABLE:
            return {'status': 'error', 'message': 'scikit-learn not installed'}

        X, y = self.build_training_data(replay_buffer)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        self.save_model()

        importances = sorted(
            zip(FEATURE_NAMES, self.model.feature_importances_),
            key=lambda x: -x[1]
        )[:8]

        return {
            'status': 'success',
            'samples': len(X),
            'mae': round(float(mae), 4),
            'top_features': [(name, round(float(val), 4)) for name, val in importances],
        }

    def predict_value(self, card_id: int) -> float:
        """Predict expected base utility score for a card ID."""
        cards = self._get_cards()
        if card_id not in cards:
            return 0.5

        if self.model is not None:
            try:
                feats = card_to_features(cards[card_id], self._get_attacks()).reshape(1, -1)
                return float(self.model.predict(feats)[0])
            except Exception:
                pass

        # Heuristic fallback
        card = cards[card_id]
        score = (card.hp or 0) / 400.0 + (0.3 if getattr(card, 'ex', False) else 0.0)
        return round(float(score), 4)

    def predict_contextual_value(
        self,
        card_id: int,
        turn: int = 1,
        my_prizes_remaining: int = 6,
        opp_prizes_remaining: int = 6,
        has_combo_target: bool = True,
        cards_played_this_turn: Optional[List[int]] = None
    ) -> float:
        """Predict phase-conditioned, state-aware dynamic card utility V(c | s, turn, sequence)."""
        base_val = self.predict_value(card_id)
        cards = self._get_cards()
        if card_id not in cards:
            return base_val

        card = cards[card_id]
        if isinstance(my_prizes_remaining, (list, tuple)):
            if cards_played_this_turn is None:
                cards_played_this_turn = list(my_prizes_remaining)
            my_prizes_remaining = 6
        if isinstance(opp_prizes_remaining, (list, tuple)):
            opp_prizes_remaining = 6
        try:
            my_prz = int(my_prizes_remaining)
        except Exception:
            my_prz = 6
        try:
            opp_prz = int(opp_prizes_remaining)
        except Exception:
            opp_prz = 6
        prize_diff = opp_prz - my_prz
        
        # 1. Early Game (Turns 1-2): Premium on Setup, Basics, Search
        if turn <= 2:
            if getattr(card, 'is_basic', False) or getattr(card, 'supertype', '') == 'Basic':
                base_val += 0.25
            if card_id in (1074, 1075, 1076, 1077, 1078):  # Nest Ball, Ultra Ball, etc.
                base_val += 0.35
            if getattr(card, 'is_stage2', False):
                base_val -= 0.20  # Cannot play on Turn 1 without setup

        # 2. Mid Game (Turns 3-5): Premium on Evolution & Rare Candy
        elif 3 <= turn <= 5:
            if card_id == 1079:  # Rare Candy
                base_val += 0.45 if has_combo_target else -0.50
            if getattr(card, 'is_stage1', False) or getattr(card, 'is_stage2', False) or getattr(card, 'is_ex', False):
                base_val += 0.30

        # 3. Late Game (Turns 6+): Premium on Disruption, Lethal Gusts & Recovery
        else:
            if card_id in (1084, 1088):  # Boss's Orders, Prime Catcher
                base_val += 0.50 if opp_prizes_remaining <= 2 or prize_diff <= 0 else 0.20
            if card_id in (1073, 1080):  # Super Rod, Unfair Stamp
                base_val += 0.35

        # 4. Sequence-Aware Valuation (Round 5 Sovereign Frontier):
        # Discount redundant search and draw cards played consecutively in the same turn
        if cards_played_this_turn:
            played_set = set(cards_played_this_turn)
            # Repeated identical item/search card
            if card_id in played_set:
                base_val -= 0.25
            # Already played a search ball this turn -> secondary search has diminishing return
            search_ids = {1074, 1075, 1076, 1077, 1078}
            if card_id in search_ids and bool(played_set.intersection(search_ids)):
                base_val -= 0.15

        return round(float(np.clip(base_val, 0.0, 1.0)), 4)

    def predict_opponent_hand_likelihood(
        self,
        card_id: int,
        opp_revealed_energy_types: List[str],
        turns_elapsed: int = 1
    ) -> float:
        """Estimate likelihood P(c_opp in Hand) for tactical counter-play."""
        cards = self._get_cards()
        if card_id not in cards:
            return 0.10

        card = cards[card_id]
        card_types = getattr(card, 'types', []) or []

        # Archetype match check
        type_match = any(t in opp_revealed_energy_types for t in card_types) if opp_revealed_energy_types else True
        if not type_match and card_types:
            return 0.02

        # Universal staples (Boss, Research, Iono) scale with turns elapsed
        if card_id in (1084, 1075, 1080, 1088):
            prob = 0.25 + min(0.60, turns_elapsed * 0.08)
            return round(prob, 3)

        return 0.20

    def rank_cards(self, top_k: int = 25) -> List[Dict[str, Any]]:
        """Return the top-K highest-value cards in the game."""
        cards = self._get_cards()
        if not cards:
            return []
        card_items = list(cards.items())
        if self.model is not None:
            try:
                attacks = self._get_attacks()
                feats = np.array([card_to_features(c, attacks) for _, c in card_items])
                preds = self.model.predict(feats)
                ranked = [
                    {'card_id': cid, 'name': c.name, 'value': round(float(preds[i]), 4)}
                    for i, (cid, c) in enumerate(card_items)
                ]
                ranked.sort(key=lambda x: -x['value'])
                return ranked[:top_k]
            except Exception:
                pass

        ranked = [
            {'card_id': cid, 'name': c.name, 'value': round(self.predict_value(cid), 4)}
            for cid, c in card_items
        ]
        ranked.sort(key=lambda x: -x['value'])
        return ranked[:top_k]


_card_value_model_instance: Optional[CardValueModel] = None


def get_card_value_model() -> CardValueModel:
    global _card_value_model_instance
    if _card_value_model_instance is None:
        _card_value_model_instance = CardValueModel()
    return _card_value_model_instance

