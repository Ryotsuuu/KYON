u"""agents/rule_engine.py
==============================
Rule Engine: Enforces deck building and gameplay rules.
"""
from typing import List, Dict, Optional


class RuleEngine:
    """Validates and enforces PTCG rules."""

    def __init__(self):
        self.rules = []
        self._setup_rules()

    def _setup_rules(self):
        self.rules = [
            {'name': 'deck_size_60', 'check': self._check_deck_size},
            {'name': 'max_4_copies', 'check': self._check_copy_limit},
            {'name': 'one_ace_spec', 'check': self._check_ace_spec},
            {'name': 'has_basic', 'check': self._check_basic_pokemon},
        ]

    def _check_deck_size(self, deck: List[int]) -> bool:
        return len(deck) == 60

    def _check_copy_limit(self, deck: List[int]) -> bool:
        from collections import Counter
        counts = Counter(deck)
        for cid, cnt in counts.items():
            # Basic energy has no limit
            try:
                from agents.csv_data import get_csv_index
                idx = get_csv_index()
                card = idx.get_card(cid)
                if card and card.category == 'Basic Energy':
                    continue
            except Exception:
                pass
            if cnt > 4:
                return False
        return True

    def _check_ace_spec(self, deck: List[int]) -> bool:
        from collections import Counter
        counts = Counter(deck)
        ace_count = 0
        try:
            from agents.csv_data import get_csv_index
            idx = get_csv_index()
            for cid in counts:
                card = idx.get_card(cid)
                if card and card.is_ace_spec:
                    ace_count += counts[cid]
        except Exception:
            pass
        return ace_count <= 1

    def _check_basic_pokemon(self, deck: List[int]) -> bool:
        try:
            from agents.csv_data import get_csv_index
            idx = get_csv_index()
            for cid in deck:
                card = idx.get_card(cid)
                if card and card.is_basic and card.is_pokemon:
                    return True
        except Exception:
            pass
        return False

    def validate(self, deck: List[int]) -> Dict[str, bool]:
        results = {}
        for rule in self.rules:
            try:
                results[rule['name']] = rule['check'](deck)
            except Exception:
                results[rule['name']] = False
        return results

    def is_legal(self, deck: List[int]) -> bool:
        return all(self.validate(deck).values())
