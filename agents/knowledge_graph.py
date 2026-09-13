u"""agents/knowledge_graph.py
============================
Card Knowledge Graph v2: Wraps both cg.api card data and csv-data enrichment.
Provides tag-based search for deck building and validation.
"""
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CardKnowledgeGraph:
    """Indexed knowledge graph mapping every card ID to a set of functional tags.
    Combines cg.api card data with csv-data enrichment.
    """

    def __init__(self):
        self.cards = []
        self.card_dict: Dict[int, Any] = {}
        self.card_tags: Dict[int, Set[str]] = {}
        self.tag_to_card_ids: Dict[str, Set[int]] = {}
        self._csv_idx = None

        try:
            from cg.api import all_card_data, all_attack, CardType, EnergyType
            self.cards = all_card_data()
            self.card_dict = {c.cardId: c for c in self.cards}
            attacks = all_attack()
            self.attack_dict = {a.attackId: a for a in attacks}
            self._build_knowledge_graph()
        except Exception as e:
            # cg not available (e.g. in Kaggle), use csv-data only
            pass

        # Load csv-data enrichment
        try:
            from agents.csv_data import get_csv_index
            self._csv_idx = get_csv_index()
            self._enrich_from_csv_data()
        except Exception:
            pass

    def _build_knowledge_graph(self):
        """Build tags from cg.api card data."""
        import re
        try:
            from cg.api import CardType
        except ImportError:
            return

        for card in self.cards:
            cid = card.cardId
            tags: Set[str] = set()

            if card.cardType == CardType.POKEMON:
                tags.add("category_pokemon")
                if card.basic:
                    tags.add("stage_basic")
                if card.stage1:
                    tags.add("stage_1")
                if card.stage2:
                    tags.add("stage_2")
                if card.ex:
                    tags.add("rule_box_ex")
                    tags.add("ex")
                if card.megaEx:
                    tags.add("rule_box_mega")
                    tags.add("mega_ex")
                if card.tera:
                    tags.add("tera")
                if not card.ex and not card.megaEx:
                    tags.add("single_prize")

                hp = card.hp or 0
                if hp >= 280:
                    tags.add("tank_ultra_high_hp")
                elif hp >= 230:
                    tags.add("tank_high_hp")
                elif hp >= 140:
                    tags.add("hp_medium")
                else:
                    tags.add("hp_low")

                et_name = card.energyType.name if hasattr(card.energyType, "name") else str(card.energyType)
                tags.add(f"type_{et_name.lower()}")

                retreat = card.retreatCost or 0
                if retreat == 0:
                    tags.add("free_retreat")
                elif retreat == 1:
                    tags.add("low_retreat")
                elif retreat >= 3:
                    tags.add("heavy_retreat")

                if card.evolvesFrom:
                    tags.add("evolved_form")

            elif card.cardType == CardType.ITEM:
                tags.add("category_trainer")
                tags.add("category_item")
                if getattr(card, "aceSpec", False):
                    tags.add("ace_spec")
            elif card.cardType == CardType.TOOL:
                tags.add("category_trainer")
                tags.add("category_tool")
                if getattr(card, "aceSpec", False):
                    tags.add("ace_spec")
            elif card.cardType == CardType.SUPPORTER:
                tags.add("category_trainer")
                tags.add("category_supporter")
                if getattr(card, "aceSpec", False):
                    tags.add("ace_spec")
            elif card.cardType == CardType.STADIUM:
                tags.add("category_trainer")
                tags.add("category_stadium")
                if getattr(card, "aceSpec", False):
                    tags.add("ace_spec")
            elif card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                tags.add("category_energy")
                if card.cardType == CardType.BASIC_ENERGY:
                    tags.add("basic_energy")
                else:
                    tags.add("special_energy")

            # Text-based tagging from skills and attacks
            full_text = []
            if hasattr(card, 'skills') and card.skills:
                for sk in card.skills:
                    tags.add("has_ability")
                    full_text.append((getattr(sk, 'name', '') or '') + ' ' + (getattr(sk, 'text', '') or ''))
            if hasattr(card, 'attacks') and card.attacks:
                for atk_id in card.attacks:
                    atk = self.attack_dict.get(atk_id)
                    if atk:
                        full_text.append((atk.name or '') + ' ' + (atk.text or ''))
                        if (atk.damage or 0) >= 200:
                            tags.add("high_damage_attacker")

            combined = ' '.join(full_text).lower()
            rules = {
                "draw": r"\b(draw|hand)\b", "search": r"\b(search|look at|put into your hand)\b",
                "heal": r"\b(heal|remove.*damage counter)\b",
                "switch_pivot": r"\b(switch|active spot|retreat)\b",
                "gust": r"\b(switch 1 of your opponent|benched pokémon to the active)\b",
                "poison": r"\bpoison\b", "burn": r"\bburn\b", "sleep": r"\bsleep\b",
                "paralyze": r"\bparalyz\b", "coin_flip": r"\b(coin|flip)\b",
                "damage_counter": r"\bplace.*damage counter\b",
                "rare_candy_synergy": r"\brare candy\b",
                "energy_acceleration": r"\battach.*energy\b",
                "energy_search": r"\bsearch.*energy\b",
                "stadium_search": r"\bsearch.*stadium\b",
                "pokemon_search": r"\bsearch.*pokémon\b",
                "bench_damage": r"\b(benched pokémon|bench)\b.*damage",
                "spread_damage": r"\b(each of your opponent|all of your opponent)\b",
                "tank_support": r"\bprevent all damage\b",
                "ability_lock": r"\b(ability|abilities).*can't be used\b",
                "mill": r"\bdiscard.*top card.*deck\b",
                "recycle": r"\bshuffle.*discard pile\b",
                "bench_sniper": r"\bdamage to 1 of your opponent's benched\b",
            }
            for tag_name, pattern in rules.items():
                if re.search(pattern, combined):
                    tags.add(tag_name)

            self.card_tags[cid] = tags
            for t in tags:
                self.tag_to_card_ids.setdefault(t, set()).add(cid)

    def _enrich_from_csv_data(self):
        """Add csv-data effect_tags to card_tags."""
        if not self._csv_idx:
            return
        for cid, csv_card in self._csv_idx.cards.items():
            existing = self.card_tags.get(cid, set())
            for tag in csv_card.effect_tags:
                existing.add(tag)
                self.tag_to_card_ids.setdefault(tag, set()).add(cid)
            # Add csv-specific tags
            if csv_card.is_team_rocket_pokemon:
                existing.add("team_rocket_pokemon")
                self.tag_to_card_ids.setdefault("team_rocket_pokemon", set()).add(cid)
            if csv_card.is_dragon_type:
                existing.add("dragon_type")
                self.tag_to_card_ids.setdefault("dragon_type", set()).add(cid)
            self.card_tags[cid] = existing

    def get_tags(self, card_id: int) -> Set[str]:
        return self.card_tags.get(card_id, set())

    def search_by_tags(self, required_tags: List[str], excluded_tags: Optional[List[str]] = None) -> list:
        if not required_tags:
            return []
        matches = set(self.tag_to_card_ids.get(required_tags[0], set()))
        for tag in required_tags[1:]:
            matches &= self.tag_to_card_ids.get(tag, set())
        if excluded_tags:
            for ex_tag in excluded_tags:
                matches -= self.tag_to_card_ids.get(ex_tag, set())
        return [self.card_dict[cid] for cid in matches if cid in self.card_dict]

    def has_tag(self, card_id: int, tag: str) -> bool:
        return tag in self.card_tags.get(card_id, set())
