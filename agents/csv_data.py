"""agents/csv_data.py
=====================
CSV Data Loader: Loads and indexes all csv-data files for deck building.
Provides fast card lookups by type, HP, damage, tags, cross-references, and rankings.

Data files loaded from data/ directory:
- cards.json (1267 cards)
- rankings.json
- cross_reference_analysis.json (9227 synergy links)
- category_analysis.json
- team_rocket_pokemon.json (52 TR Pokemon)
- triple_energy_candidates.json
- dragon_type_cards.json (35 Dragon-type)
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class CsvCardData:
    """A card from csv-data with all enriched fields."""
    __slots__ = (
        'card_id', 'name', 'category', 'pokemon_stage', 'evolves_from',
        'hp', 'type', 'weakness', 'resistance', 'retreat_cost',
        'is_ex', 'is_mega_ex', 'is_ace_spec', 'is_team_rocket_pokemon',
        'is_dragon_type', 'abilities', 'attacks', 'effect_tags',
        'energy_fetch_or_flex', 'archetype_fits',
        'strategic_analysis', 'categorization', 'cross_reference_synergies',
        'ga_mutation_risk', 'ga_mutation_risk_note', 'conditional_effects',
        'is_colorless_type', 'colorless_energy_only', 'confidence',
    )

    def __init__(self, d: dict):
        self.card_id = d.get('card_id', 0)
        self.name = d.get('name', '')
        self.category = d.get('category', '')
        self.pokemon_stage = d.get('pokemon_stage')
        self.evolves_from = d.get('evolves_from')
        self.hp = d.get('hp')
        self.type = d.get('type', '')
        self.weakness = d.get('weakness', '')
        self.resistance = d.get('resistance', '')
        self.retreat_cost = d.get('retreat_cost', 0) or 0
        self.is_ex = d.get('is_ex', False)
        self.is_mega_ex = d.get('is_mega_ex', False)
        self.is_ace_spec = d.get('is_ace_spec', False)
        self.is_team_rocket_pokemon = d.get('is_team_rocket_pokemon', False)
        self.is_dragon_type = d.get('is_dragon_type', False)
        self.abilities = d.get('abilities', [])
        self.attacks = d.get('attacks', [])
        self.effect_tags = d.get('effect_tags', [])
        self.energy_fetch_or_flex = d.get('energy_fetch_or_flex', False)
        self.archetype_fits = d.get('archetype_fits', [])
        self.strategic_analysis = d.get('strategic_analysis', {}) or {}
        self.categorization = d.get('categorization', {}) or {}
        self.cross_reference_synergies = d.get('cross_reference_synergies', []) or []
        self.ga_mutation_risk = d.get('ga_mutation_risk', False)
        self.ga_mutation_risk_note = d.get('ga_mutation_risk_note', '')
        self.conditional_effects = d.get('conditional_effects', []) or []
        self.is_colorless_type = d.get('is_colorless_type', False)
        self.colorless_energy_only = d.get('colorless_energy_only', False)
        self.confidence = d.get('confidence', 'high')

    @property
    def is_pokemon(self) -> bool:
        return self.category == 'Pokemon'

    @property
    def is_basic(self) -> bool:
        return self.pokemon_stage == 'Basic'

    @property
    def is_stage1(self) -> bool:
        return self.pokemon_stage == 'Stage 1'

    @property
    def is_stage2(self) -> bool:
        return self.pokemon_stage == 'Stage 2'

    @property
    def is_trainer(self) -> bool:
        return bool(self.category and self.category.startswith('Trainer-'))

    @property
    def is_item(self) -> bool:
        return self.category == 'Trainer-Item'

    @property
    def is_supporter(self) -> bool:
        return self.category == 'Trainer-Supporter'

    @property
    def is_stadium(self) -> bool:
        return self.category == 'Trainer-Stadium'

    @property
    def is_tool(self) -> bool:
        return self.category == 'Trainer-Tool'

    @property
    def is_special_energy(self) -> bool:
        return self.category == 'Special Energy'

    @property
    def is_basic_energy(self) -> bool:
        return self.category == 'Basic Energy'

    @property
    def gameplay_role(self) -> str:
        return self.strategic_analysis.get('gameplay_role', 'utility')

    @property
    def effectiveness_rating(self) -> str:
        return str(self.strategic_analysis.get('effectiveness_rating', 'medium'))

    @property
    def effectiveness_score(self) -> float:
        val = self.strategic_analysis.get('effectiveness_rating', 'medium')
        if isinstance(val, (int, float)):
            return float(val)
        mapping = {'elite': 9.0, 'very_high': 8.5, 'high': 7.5, 'medium': 5.0, 'low': 3.0, 'situational': 4.0}
        return mapping.get(str(val).lower(), 5.0)

    @property
    def deckbuilding_priority(self) -> str:
        return self.strategic_analysis.get('deckbuilding_priority', 'medium')

    @property
    def is_high_damage(self) -> bool:
        dmg_cat = self.categorization.get('damage', {})
        if isinstance(dmg_cat, dict) and dmg_cat.get('is_high_damage'):
            return True
        return self.max_damage >= 200

    @property
    def has_bench_protection(self) -> bool:
        spec = self.categorization.get('special_ability', {})
        if isinstance(spec, dict) and spec.get('bench_protection'):
            return True
        return 'bench_protection' in self.effect_tags

    @property
    def has_energy_acceleration(self) -> bool:
        spec = self.categorization.get('special_ability', {})
        if isinstance(spec, dict) and spec.get('energy_acceleration'):
            return True
        return 'energy_acceleration' in self.effect_tags

    @property
    def has_heal(self) -> bool:
        hf = self.categorization.get('heal_factor', {})
        if isinstance(hf, dict) and hf.get('has_heal'):
            return True
        return any(t.startswith('heal_') for t in self.effect_tags)

    @property
    def has_free_retreat(self) -> bool:
        nr = self.categorization.get('no_retreat_cost', {})
        if isinstance(nr, dict) and nr.get('has_zero_retreat_cost'):
            return True
        return self.retreat_cost == 0

    def get_attack_energy_types(self) -> set:
        """Extract elemental energy types required by this Pokemon's attacks (excluding Colorless)."""
        symbol_to_type = {
            '{G}': 'Grass',
            '{R}': 'Fire',
            '{W}': 'Water',
            '{L}': 'Lightning',
            '{P}': 'Psychic',
            '{F}': 'Fighting',
            '{D}': 'Darkness',
            '{M}': 'Metal',
        }
        needed = set()
        for att in self.attacks:
            cost_str = att.get('cost', '')
            for sym, etype in symbol_to_type.items():
                if sym in cost_str:
                    needed.add(etype)
        return needed

    @property
    def special_energy_target_type(self) -> Optional[str]:
        """Return the target Pokemon type or attribute required for a special energy card's bonus effect."""
        if not self.is_special_energy:
            return None
        if self.card_id == 18 or 'Grow Grass' in self.name:
            return 'Grass'
        if self.card_id == 19 or 'Telepath' in self.name:
            return 'Psychic'
        if self.card_id == 20 or 'Rock Fighting' in self.name:
            return 'Fighting'
        if self.card_id == 15 or 'Team Rocket' in self.name:
            return 'Team Rocket'
        return None

    @property
    def max_damage(self) -> int:
        if not self.attacks:
            return 0
        best = 0
        for a in self.attacks:
            d = a.get('damage', 0)
            if d is not None:
                try:
                    best = max(best, int(d))
                except (ValueError, TypeError):
                    pass
        return best


class CsvDataIndex:
    """Fast index over all csv-data for deck building queries."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.cards: Dict[int, CsvCardData] = {}
        self.cards_by_name: Dict[str, CsvCardData] = {}

        # Indexed collections
        self.pokemon_by_type: Dict[str, List[CsvCardData]] = {}
        self.basic_pokemon_by_type: Dict[str, List[CsvCardData]] = {}
        self.trainers_by_category: Dict[str, List[CsvCardData]] = {}
        self.special_energies: List[CsvCardData] = []
        self.basic_energies: List[CsvCardData] = []
        self.ace_specs: List[CsvCardData] = []
        self.team_rocket_pokemon: List[CsvCardData] = []
        self.dragon_pokemon: List[CsvCardData] = []
        self.triple_energy_candidates: List[CsvCardData] = []

        # Cross-references
        self.cross_refs: Dict[int, List[dict]] = {}  # card_id -> list of synergy links
        self.rankings: Dict[str, Any] = {}

        # Category analysis
        self.heal_cards: List[dict] = []
        self.status_inflictors: List[dict] = []
        self.stadium_cards: List[dict] = []
        self.high_damage_cards: List[dict] = []
        self.damage_counter_cards: List[dict] = []

        # Advanced strategic role & GA risk indexes
        self.cards_by_role: Dict[str, List[CsvCardData]] = {}
        self.ga_risk_cards: Set[int] = set()
        self.cards_by_archetype: Dict[str, List[CsvCardData]] = {}

        self._load_all()

    def _load_json(self, filename: str) -> Any:
        path = self.data_dir / filename
        if not path.exists():
            # Fallback to csv-data directory if not found in data_dir
            fallback = ROOT / "csv-data" / filename
            if fallback.exists():
                path = fallback
            else:
                return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_all(self):
        # 1. Load cards.json
        raw_cards = self._load_json('cards.json')
        if raw_cards is None:
            raise FileNotFoundError(f"cards.json not found in {self.data_dir}")

        for c in raw_cards:
            card = CsvCardData(c)
            self.cards[card.card_id] = card
            self.cards_by_name[card.name] = card

            # Index by strategic gameplay role
            role = card.gameplay_role
            if role:
                self.cards_by_role.setdefault(role, []).append(card)

            # Index GA mutation risk
            if card.ga_mutation_risk:
                self.ga_risk_cards.add(card.card_id)

            # Index by archetype fits
            for af in card.archetype_fits:
                arch_name = af.get('archetype') if isinstance(af, dict) else str(af)
                if arch_name:
                    self.cards_by_archetype.setdefault(arch_name, []).append(card)

        # 2. Index by type
        for card in self.cards.values():
            if card.is_pokemon:
                self.pokemon_by_type.setdefault(card.type, []).append(card)
                if card.is_basic:
                    self.basic_pokemon_by_type.setdefault(card.type, []).append(card)
            elif card.is_trainer:
                cat = card.category  # e.g. Trainer-Item, Trainer-Stadium
                self.trainers_by_category.setdefault(cat, []).append(card)
            elif card.is_special_energy:
                self.special_energies.append(card)
            elif card.is_basic_energy:
                self.basic_energies.append(card)

            if card.is_ace_spec:
                self.ace_specs.append(card)
            if card.is_team_rocket_pokemon:
                self.team_rocket_pokemon.append(card)
            if card.is_dragon_type:
                self.dragon_pokemon.append(card)

        # 3. Load cross-references
        cr = self._load_json('cross_reference_analysis.json')
        if cr and isinstance(cr, dict):
            for card_id_str, refs in cr.items():
                self.cross_refs[int(card_id_str)] = refs

        # 4. Load rankings
        rankings = self._load_json('rankings.json')
        if rankings:
            self.rankings = rankings

        # 5. Load team rocket pokemon
        tr_data = self._load_json('team_rocket_pokemon.json')
        if tr_data and isinstance(tr_data, dict) and 'cards' in tr_data:
            for c in tr_data['cards']:
                cid = c.get('card_id')
                if cid and cid in self.cards:
                    self.cards[cid].is_team_rocket_pokemon = True
                    if self.cards[cid] not in self.team_rocket_pokemon:
                        self.team_rocket_pokemon.append(self.cards[cid])

        # 6. Load dragon type cards
        dr_data = self._load_json('dragon_type_cards.json')
        if dr_data and isinstance(dr_data, dict) and 'cards' in dr_data:
            for c in dr_data['cards']:
                cid = c.get('card_id')
                if cid and cid in self.cards:
                    self.cards[cid].is_dragon_type = True
                    if self.cards[cid] not in self.dragon_pokemon:
                        self.dragon_pokemon.append(self.cards[cid])

        # 7. Load triple energy candidates
        te_data = self._load_json('triple_energy_candidates.json')
        if te_data:
            if isinstance(te_data, dict) and 'cards' in te_data:
                for c in te_data['cards']:
                    cid = c.get('card_id')
                    if cid and cid in self.cards:
                        self.triple_energy_candidates.append(self.cards[cid])
            elif isinstance(te_data, list):
                for c in te_data:
                    cid = c.get('card_id')
                    if cid and cid in self.cards:
                        self.triple_energy_candidates.append(self.cards[cid])

        # 8. Load category analysis
        ca = self._load_json('category_analysis.json')
        if ca and isinstance(ca, dict):
            self.heal_cards = ca.get('heal_cards', [])
            self.status_inflictors = ca.get('status_inflictors', [])
            self.stadium_cards = ca.get('stadium_cards', [])
            self.high_damage_cards = ca.get('high_damage_cards', [])
            self.damage_counter_cards = ca.get('counter_cards', [])

        # 9. Sort pokemon by HP desc within each type
        for t in self.pokemon_by_type:
            self.pokemon_by_type[t].sort(key=lambda c: int(c.hp or 0), reverse=True)
        for t in self.basic_pokemon_by_type:
            self.basic_pokemon_by_type[t].sort(key=lambda c: int(c.hp or 0), reverse=True)

    def get_card(self, card_id: int) -> Optional[CsvCardData]:
        return self.cards.get(card_id)

    def get_synergy_partners(self, card_id: int) -> List[dict]:
        return self.cross_refs.get(card_id, [])

    def get_pokemon_for_energy_types(self, energy_types: List[str]) -> List[CsvCardData]:
        """Get all Pokemon that use any of the given energy types."""
        result = []
        seen = set()
        for et in energy_types:
            for p in self.pokemon_by_type.get(et, []):
                if p.card_id not in seen:
                    seen.add(p.card_id)
                    result.append(p)
        # Add Colorless Pokemon too (universal)
        for p in self.pokemon_by_type.get('Colorless', []):
            if p.card_id not in seen:
                seen.add(p.card_id)
                result.append(p)
        return result

    def get_basic_pokemon_for_energy_types(self, energy_types: List[str]) -> List[CsvCardData]:
        """Get all Basic Pokemon that use any of the given energy types."""
        result = []
        seen = set()
        for et in energy_types:
            for p in self.basic_pokemon_by_type.get(et, []):
                if p.card_id not in seen:
                    seen.add(p.card_id)
                    result.append(p)
        return result

    def get_special_energy_by_type(self, energy_type: str) -> Optional[CsvCardData]:
        """Get special energy card for a given energy type.
        Grow Grass Energy -> Grass, Telepath Psychic Energy -> Psychic, Rock Fighting Energy -> Fighting
        Team Rocket's Energy -> Team Rocket
        """
        mapping = {
            'Grass': 18,    # Grow Grass Energy
            'Psychic': 19,  # Telepath Psychic Energy
            'Fighting': 20, # Rock Fighting Energy
            'Team Rocket': 15,  # Team Rocket's Energy
        }
        target_id = mapping.get(energy_type)
        if target_id:
            return self.cards.get(target_id)
        return None

    def get_evolution_chain(self, card: CsvCardData) -> List[CsvCardData]:
        """Resolve the full evolution chain for a Pokemon card (top-down: Stage2 -> Stage1 -> Basic)."""
        chain = [card]
        current = card
        while current.evolves_from:
            prev_name = current.evolves_from
            # Find the pre-evolution card
            prev_card = None
            for c in self.cards.values():
                if c.name == prev_name and c.is_pokemon:
                    # Prefer same type, or basic
                    if c.is_basic or c.type == current.type:
                        prev_card = c
                        break
            if prev_card is None:
                # Fallback: find any card with that name
                for c in self.cards.values():
                    if c.name == prev_name and c.is_pokemon:
                        prev_card = c
                        break
            if prev_card is None:
                break
            chain.append(prev_card)
            current = prev_card
        chain.reverse()  # Now Basic -> Stage1 -> Stage2
        return chain

    def get_trainer_by_tag(self, tag: str, category: str = None) -> List[CsvCardData]:
        """Get trainer cards that have a specific effect_tag."""
        results = []
        for c in self.cards.values():
            if not c.is_trainer:
                continue
            if category and c.category != category:
                continue
            if tag in c.effect_tags:
                results.append(c)
        return results

    def get_strategic_role(self, card_id: int) -> str:
        """Get the primary gameplay role for a card from strategic analysis."""
        card = self.get_card(card_id)
        return card.gameplay_role if card else 'utility'

    def has_ga_mutation_risk(self, card_id: int) -> bool:
        """Check if a card has GA mutation risk (requires invariant protection)."""
        card = self.get_card(card_id)
        return bool(card and card.ga_mutation_risk)

    def get_ga_mutation_risk_note(self, card_id: int) -> str:
        """Get the specific invariant note for a card with GA mutation risk."""
        card = self.get_card(card_id)
        return card.ga_mutation_risk_note if card else ''

    def get_complementary_cards(self, card_id: int, archetype: Optional[str] = None) -> List[CsvCardData]:
        """Find cards with complementary strategic roles and archetype synergy."""
        card = self.get_card(card_id)
        if not card:
            return []

        role = card.gameplay_role
        # Role complement map
        role_pairs = {
            'attacker': ['support', 'energy_foundation', 'draw_engine'],
            'support': ['attacker', 'utility'],
            'energy_foundation': ['attacker', 'support'],
            'draw_engine': ['attacker', 'evolution_fodder'],
            'disruption': ['attacker', 'support'],
            'utility': ['attacker', 'support'],
        }
        target_roles = role_pairs.get(role, ['attacker', 'support'])
        results = []
        for tr in target_roles:
            for c in self.cards_by_role.get(tr, []):
                if c.card_id == card_id:
                    continue
                if archetype and archetype not in [af.get('archetype') if isinstance(af, dict) else str(af) for af in c.archetype_fits]:
                    continue
                results.append(c)
        return results


# Singleton
_CSV_INDEX = None

def get_csv_index(data_dir: Optional[Path] = None) -> CsvDataIndex:
    global _CSV_INDEX
    if _CSV_INDEX is None:
        _CSV_INDEX = CsvDataIndex(data_dir)
    return _CSV_INDEX
