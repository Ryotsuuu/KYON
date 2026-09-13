"""
agents/csv_deck_builder.py
=============================
CSV-Data-Powered Deck Builder v3 (Deep Energy-Constraint & Strategy Alignment).

Rules enforced:
1. Max 6 Pokemon species, 3 copies each = max 18 Pokemon cards
2. Dual Check: "Card Type" & "Attack Ability Required Energy" - Pokémon are only
   included if at least one of their attacks is fully usable with the deck's energy types
3. Combo-Type Representation: Dual/Triple decks must feature native elemental Pokémon
   from each combo type, alongside strong universal Colorless basics
4. Only 1 evolution chain (Basic->S1 or Basic->S1->S2), prioritizing archetype/combo types
5. Remaining Pokemon slots: EX, Mega EX, or high HP/damage compatible basics
6. At least 4 special energy when applicable (Prism #16 for Dragon, TR #15 for Team Rocket, Type Specials)
7. Attack-Derived Energy Split for Dragon decks: Basic energies match the actual attack costs
   of the selected Dragon Pokémon + 4 Prism Energies (No Metal #8 fallback)
8. At least 2 copies of ID 1119 (Energy Search) in every deck
9. At least 1 ACE SPEC card (1100 for dragon/triple, 1080 standard)
10. At least 1 Rare Candy for stage_2 decks
11. 20-25 energy cards total
12. Special archetype mechanic cards (1-3) for: ability_heal, poison/burn/sleep/paralyze, stadium_control, damage_counter
"""
import sys
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.csv_data import CsvDataIndex, CsvCardData, get_csv_index
from agents.archetype_engine import (
    ArchetypeTemplate, get_archetype_template, get_all_archetypes_for_combo,
)

# Fixed card IDs
RARE_CANDY_ID = 1079
ENERGY_SEARCH_ID = 1119
ENERGY_SEARCH_PRO_ID = 1100
TEAM_ROCKET_ENERGY_ID = 15
PRISM_ENERGY_ID = 16
GROW_GRASS_ENERGY_ID = 18
TELEPATH_PSYCHIC_ENERGY_ID = 19
ROCK_FIGHTING_ENERGY_ID = 20

# Special energy type mapping
SPECIAL_ENERGY_MAP = {
    'Grass': GROW_GRASS_ENERGY_ID,
    'Psychic': TELEPATH_PSYCHIC_ENERGY_ID,
    'Fighting': ROCK_FIGHTING_ENERGY_ID,
}

# Energy type name to API EnergyType value mapping
ENERGY_TYPE_VALUES = {
    'Colorless': 0, 'Grass': 1, 'Fire': 2, 'Water': 3, 'Lightning': 4,
    'Psychic': 5, 'Fighting': 6, 'Darkness': 7, 'Metal': 8, 'Dragon': 9,
}

LETTER_TO_TYPE = {
    'G': 'Grass',
    'R': 'Fire',
    'W': 'Water',
    'L': 'Lightning',
    'P': 'Psychic',
    'F': 'Fighting',
    'D': 'Darkness',
    'M': 'Metal',
}


def extract_attack_required_types(cost_str: str) -> Set[str]:
    """Extract required elemental energy types from attack cost string."""
    if not cost_str:
        return set()
    reqs = set(re.findall(r'\{([A-Z])\}', cost_str))
    return {LETTER_TO_TYPE[l] for l in reqs if l in LETTER_TO_TYPE}


def is_pokemon_attack_compatible(card: CsvCardData, energy_types: List[str], has_prism: bool = False) -> bool:
    """Check if at least ONE of the Pokemon's attacks is fully payable in the deck."""
    if not card or not card.is_pokemon:
        return True
    if not card.attacks:
        return True

    supported = set(energy_types)
    for atk in card.attacks:
        cost = atk.get('cost', '')
        req_types = extract_attack_required_types(cost)
        # 1. Attack requires only Colorless (●) energy -> universally usable!
        if not req_types:
            return True
        # 2. Attack specific energy symbols are all supplied by the deck
        if req_types.issubset(supported):
            return True
        # 3. Prism Energy provides all colors to Basic Pokemon
        if has_prism and card.is_basic:
            return True

    return False


def _hp_val(card: CsvCardData) -> int:
    try:
        return int(card.hp or 0)
    except (ValueError, TypeError):
        return 0


def _damage_val(card: CsvCardData) -> int:
    return card.max_damage


def _score_pokemon(card: CsvCardData, is_active: bool = True) -> float:
    """Score a Pokemon card for deck inclusion."""
    score = 0.0
    score += _hp_val(card) * 0.5
    score += _damage_val(card) * 1.0
    if card.is_ex:
        score += 40
    if card.is_mega_ex:
        score += 60
    if card.retreat_cost == 0:
        score += 15
    elif card.retreat_cost <= 1:
        score += 8
    elif card.retreat_cost >= 3:
        score -= 10
    if card.abilities:
        score += 10
    return score


def _pick_best_pokemon(pool: List[CsvCardData], count: int, rng: random.Random,
                        require_basic: bool = False, require_ex: bool = False) -> List[CsvCardData]:
    """Pick best Pokemon from pool, sorted by score with slight random jitter."""
    filtered = pool
    if require_basic:
        filtered = [p for p in filtered if p.is_basic]
    if require_ex:
        filtered = [p for p in filtered if p.is_ex or p.is_mega_ex]

    scored = [(p, _score_pokemon(p) + rng.gauss(0, 2)) for p in filtered]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:count]]


class CsvDeckBuilder:
    """Build 60-card decks using CSV data with full dual-check energy constraints."""

    def __init__(self, data_index: Optional[CsvDataIndex] = None, data_dir: Optional[Path] = None):
        self.idx = data_index or get_csv_index(data_dir)

    def build_deck(
        self,
        energy_types: List[str],
        archetype_name: str,
        combo_type: str = 'single',
        seed: int = 42,
    ) -> Tuple[List[int], dict]:
        """Build a complete 60-card deck adhering to strict attack energy constraints."""
        rng = random.Random(seed)
        archetype = get_archetype_template(archetype_name, combo_type)

        # ===== STEP 1: Select Pokemon (max 6 species x 3 = max 18 cards) =====
        pokemon_cards, pokemon_info = self._select_pokemon(energy_types, archetype, combo_type, rng)

        # ===== STEP 2: Select Energy (20-25 cards) =====
        energy_cards, energy_info = self._select_energy(energy_types, archetype, combo_type, pokemon_cards, rng)

        # ===== STEP 3: Select Trainers (fill to 60) =====
        trainer_count = 60 - len(pokemon_cards) - len(energy_cards)
        trainer_cards, trainer_info = self._select_trainers(
            energy_types, archetype, combo_type, trainer_count, pokemon_cards, rng
        )

        deck = pokemon_cards + trainer_cards + energy_cards

        build_info = {
            'pokemon': pokemon_info,
            'energy': energy_info,
            'trainers': trainer_info,
            'total': len(deck),
            'archetype': archetype_name,
            'combo_type': combo_type,
        }

        return deck, build_info

    def _select_pokemon(
        self, energy_types: List[str], archetype: ArchetypeTemplate,
        combo_type: str, rng: random.Random
    ) -> Tuple[List[int], dict]:
        """Select max 6 Pokemon species x 3 copies each."""
        if combo_type == 'team_rocket':
            return self._select_pokemon_team_rocket(archetype, rng)
        elif combo_type == 'dragon':
            return self._select_pokemon_dragon(archetype, rng)
        else:
            return self._select_pokemon_standard(energy_types, archetype, rng)

    def _select_pokemon_standard(
        self, energy_types: List[str], archetype: ArchetypeTemplate, rng: random.Random
    ) -> Tuple[List[int], dict]:
        """Standard single/dual/triple Pokemon selection with attack compatibility check."""
        raw_pool = self.idx.get_pokemon_for_energy_types(energy_types)
        # Exclude Dragon type and Team Rocket Pokemon (handled separately) and enforce attack energy constraint
        pool = [
            p for p in raw_pool
            if p.type != 'Dragon' and not p.is_team_rocket_pokemon and is_pokemon_attack_compatible(p, energy_types)
        ]

        selected_species: List[CsvCardData] = []
        selected_names: Set[str] = set()
        info = {'evolution_chain': None, 'species': []}

        evo_mode = archetype.evolution_mode

        # ── 1. Evolution Chain Selection ──────────────────────────────────
        if evo_mode in ('stage2', 'stage2_ex', 'stage2_mega'):
            chain = self._find_best_stage2_chain(pool, energy_types, rng, prefer_mega=(evo_mode == 'stage2_mega'))
            if chain:
                for c in chain:
                    selected_species.append(c)
                    selected_names.add(c.name)
                info['evolution_chain'] = [c.name for c in chain]

        elif evo_mode in ('stage1', 'stage1_ex', 'stage1_mega'):
            chain = self._find_best_stage1_chain(pool, energy_types, rng, prefer_mega=(evo_mode == 'stage1_mega'))
            if chain:
                for c in chain:
                    selected_species.append(c)
                    selected_names.add(c.name)
                info['evolution_chain'] = [c.name for c in chain]

        # ── 2. Combo-Type Elemental Representation ────────────────────────
        # For Dual / Triple decks, ensure at least 1-2 native typed Pokemon exist for each combo type
        if len(energy_types) >= 2:
            for et in energy_types:
                already_has_type = any(s.type == et for s in selected_species)
                if not already_has_type and len(selected_species) < 6:
                    typed_pool = [p for p in pool if p.type == et and p.is_basic and p.name not in selected_names]
                    if typed_pool:
                        best_typed = _pick_best_pokemon(typed_pool, 1, rng)
                        for bt in best_typed:
                            if len(selected_species) < 6:
                                selected_species.append(bt)
                                selected_names.add(bt.name)

        # ── 3. Fill Remaining Slots with Powerhouse Basics / EX / Mega ─────
        remaining = 6 - len(selected_species)
        if remaining > 0:
            if evo_mode in ('stage2_mega', 'stage1_mega', 'mega_stage_2_ex', 'mega_stage_1_ex'):
                # Mega EX basics + powerhouse basics
                mega_pool = [p for p in pool if p.is_basic and p.is_mega_ex and p.name not in selected_names]
                filler = _pick_best_pokemon(mega_pool, remaining, rng)
                if len(filler) < remaining:
                    extra_pool = [p for p in pool if p.is_basic and p.name not in selected_names and p not in filler]
                    more = _pick_best_pokemon(extra_pool, remaining - len(filler), rng)
                    filler.extend(more)
            elif evo_mode in ('stage2_ex', 'stage1_ex'):
                filler = _pick_best_pokemon(pool, remaining, rng, require_basic=True, require_ex=True)
            else:
                basic_pool = [p for p in pool if p.is_basic and p.name not in selected_names]
                filler = _pick_best_pokemon(basic_pool, remaining, rng)

            for f in filler:
                if f.name not in selected_names and len(selected_species) < 6:
                    selected_species.append(f)
                    selected_names.add(f.name)

        # Fallback filler
        if len(selected_species) < 6:
            others = [p for p in pool if p.name not in selected_names]
            for p in others:
                if len(selected_species) >= 6:
                    break
                selected_species.append(p)
                selected_names.add(p.name)

        # Build card list: 3 copies each
        cards = []
        for sp in selected_species[:6]:
            cards.extend([sp.card_id] * 3)
            info['species'].append({'id': sp.card_id, 'name': sp.name, 'hp': sp.hp, 'type': sp.type})

        return cards, info

    def _find_best_stage2_chain(
        self, pool: List[CsvCardData], energy_types: List[str],
        rng: random.Random, prefer_mega: bool = False
    ) -> Optional[List[CsvCardData]]:
        """Find the best Basic -> Stage1 -> Stage2 chain, prioritizing combo elemental types."""
        stage2_cards = [p for p in pool if p.is_stage2]

        def _chain_score(c):
            score = _hp_val(c) + _damage_val(c)
            if c.type in energy_types:
                score += 300.0  # Heavy bonus to native elemental types over Colorless!
            if prefer_mega and (c.is_mega_ex or 'Mega' in c.name):
                score += 400.0  # Mega Stage 2 priority (e.g. Mega Gardevoir ex, Mega Gengar ex)
            return score

        stage2_cards.sort(key=_chain_score, reverse=True)

        for s2 in stage2_cards:
            chain = self.idx.get_evolution_chain(s2)
            if len(chain) == 3 and chain[0].is_basic and chain[1].is_stage1 and chain[2].is_stage2:
                valid = all(is_pokemon_attack_compatible(c, energy_types) for c in chain)
                if valid:
                    return chain
        return None

    def _find_best_stage1_chain(
        self, pool: List[CsvCardData], energy_types: List[str],
        rng: random.Random, prefer_mega: bool = False
    ) -> Optional[List[CsvCardData]]:
        """Find the best Basic -> Stage1 chain, prioritizing combo elemental types."""
        stage1_cards = [p for p in pool if p.is_stage1]

        def _chain_score(c):
            score = _hp_val(c) + _damage_val(c)
            if c.type in energy_types:
                score += 300.0
            if prefer_mega and (c.is_mega_ex or 'Mega' in c.name):
                score += 400.0
            return score

        stage1_cards.sort(key=_chain_score, reverse=True)

        for s1 in stage1_cards:
            chain = self.idx.get_evolution_chain(s1)
            if len(chain) == 2 and chain[0].is_basic and chain[1].is_stage1:
                valid = all(is_pokemon_attack_compatible(c, energy_types) for c in chain)
                if valid:
                    return chain
        return None

    def _select_pokemon_team_rocket(
        self, archetype: ArchetypeTemplate, rng: random.Random
    ) -> Tuple[List[int], dict]:
        """Team Rocket Pokemon selection: 1 complete evolution line (Stage 1 or Stage 2)
        + High-HP, High-Damage powerhouse Basic Pokemon for remaining slots (max 6 species).
        """
        tr_pool = [
            p for p in self.idx.team_rocket_pokemon
            if p.is_pokemon and p.hp and p.attacks
            and is_pokemon_attack_compatible(p, ['Psychic', 'Darkness'])
        ]

        selected: List[CsvCardData] = []
        selected_names: Set[str] = set()
        info = {'species': [], 'team_rocket': True}

        evo_mode = archetype.evolution_mode

        # 1. Always include at least 1 complete Stage 1 or Stage 2 evolution chain
        chain = None
        if evo_mode in ('stage2', 'stage2_ex', 'stage2_mega'):
            chain = self._find_best_stage2_chain(tr_pool, ['Psychic', 'Darkness'], rng)
        elif evo_mode in ('stage1', 'stage1_ex', 'stage1_mega'):
            chain = self._find_best_stage1_chain(tr_pool, ['Psychic', 'Darkness'], rng)
        else:
            # For balanced/aggro/stall: try Stage 2 chain first, fallback to Stage 1
            chain = self._find_best_stage2_chain(tr_pool, ['Psychic', 'Darkness'], rng) or self._find_best_stage1_chain(tr_pool, ['Psychic', 'Darkness'], rng)

        if chain:
            for c in chain:
                selected.append(c)
                selected_names.add(c.name)
            info['evolution_chain'] = [c.name for c in chain]

        # 2. Fill all remaining slots with High-HP, High-Damage Powerhouse Basics
        remaining = 6 - len(selected)
        if remaining > 0:
            tr_basics = [p for p in tr_pool if p.is_basic and p.name not in selected_names and _hp_val(p) >= 100]
            tr_basics.sort(key=lambda c: (_hp_val(c) * 0.5 + _damage_val(c)), reverse=True)
            for p in tr_basics:
                if len(selected) >= 6:
                    break
                if p.name not in selected_names:
                    selected.append(p)
                    selected_names.add(p.name)

        # 3. High-tier powerhouse basic partner/fillers (e.g. Bloodmoon Ursaluna ex, Mega Absol ex, TR Mewtwo ex)
        if len(selected) < 6:
            partner_pool = [p for p in self.idx.basic_pokemon_by_type.get('Psychic', [])]
            partner_pool += [p for p in self.idx.basic_pokemon_by_type.get('Darkness', [])]
            partner_pool += [p for p in self.idx.basic_pokemon_by_type.get('Colorless', [])]
            partner_pool = [
                p for p in partner_pool
                if p.name not in selected_names and _hp_val(p) >= 120
                and is_pokemon_attack_compatible(p, ['Psychic', 'Darkness'])
            ]
            partner_pool.sort(key=lambda c: (_hp_val(c) * 0.5 + _damage_val(c) + (50 if c.is_ex or c.is_mega_ex else 0)), reverse=True)

            if partner_pool:
                for p in partner_pool:
                    if len(selected) >= 6:
                        break
                    if p.name not in selected_names:
                        selected.append(p)
                        selected_names.add(p.name)

        cards = []
        for sp in selected[:6]:
            cards.extend([sp.card_id] * 3)
            info['species'].append({'id': sp.card_id, 'name': sp.name, 'hp': sp.hp, 'type': sp.type})

        return cards, info

    def _select_pokemon_dragon(
        self, archetype: ArchetypeTemplate, rng: random.Random
    ) -> Tuple[List[int], dict]:
        """Dragon Pokemon selection: Top HP & Damage Dragon Pokemon + 1 powerhouse partner."""
        dragon_pool = [p for p in self.idx.dragon_pokemon if p.is_pokemon and p.hp]
        dragon_pool.sort(key=lambda c: (_hp_val(c) * 0.6 + _damage_val(c) * 1.2 + (50 if c.is_ex or c.is_mega_ex else 0)), reverse=True)

        selected: List[CsvCardData] = []
        selected_names: Set[str] = set()
        info = {'species': [], 'dragon': True}

        evo_mode = archetype.evolution_mode
        if evo_mode in ('stage2', 'stage2_ex', 'stage2_mega', 'balanced', 'damage_counter'):
            chain = self._find_best_stage2_chain(dragon_pool, ['Dragon'], rng)
            if chain:
                for c in chain:
                    selected.append(c)
                    selected_names.add(c.name)
                info['evolution_chain'] = [c.name for c in chain]
        elif evo_mode in ('stage1', 'stage1_ex', 'stage1_mega'):
            chain = self._find_best_stage1_chain(dragon_pool, ['Dragon'], rng)
            if chain:
                for c in chain:
                    selected.append(c)
                    selected_names.add(c.name)
                info['evolution_chain'] = [c.name for c in chain]

        # Fill remaining slots with Top-Tier High HP & Damage BASIC Dragon Pokemon
        remaining_dragon = 5 - len(selected)
        dragon_basics = [p for p in dragon_pool if p.is_basic and p.name not in selected_names]
        dragon_basics.sort(key=lambda c: (_hp_val(c) * 0.6 + _damage_val(c) * 1.2 + (50 if c.is_ex or c.is_mega_ex else 0)), reverse=True)

        for p in dragon_basics:
            if remaining_dragon <= 0 or len(selected) >= 5:
                break
            selected.append(p)
            selected_names.add(p.name)
            remaining_dragon -= 1

        # Powerhouse Partner slot
        partner = self._find_dragon_partner(selected, rng)
        if partner and partner.name not in selected_names and len(selected) < 6:
            selected.append(partner)
            selected_names.add(partner.name)

        if len(selected) < 6:
            colorless_basics = [
                p for p in self.idx.basic_pokemon_by_type.get('Colorless', [])
                if p.is_basic and _hp_val(p) >= 180 and p.name not in selected_names
            ]
            colorless_basics.sort(key=lambda c: (_hp_val(c) + _damage_val(c)), reverse=True)
            for p in colorless_basics:
                if len(selected) >= 6:
                    break
                if p.name not in selected_names:
                    selected.append(p)
                    selected_names.add(p.name)

        cards = []
        for sp in selected[:6]:
            cards.extend([sp.card_id] * 3)
            info['species'].append({'id': sp.card_id, 'name': sp.name, 'hp': sp.hp, 'type': sp.type})

        return cards, info

    def _find_dragon_partner(self, selected_dragons: List[CsvCardData], rng: random.Random) -> Optional[CsvCardData]:
        """Find best high-HP powerhouse partner for dragon deck."""
        candidates = []
        for d in selected_dragons:
            for ref in self.idx.get_synergy_partners(d.card_id):
                if ref.get('type') == 'dragon_energy_solution':
                    pid = ref.get('partner_id')
                    p = self.idx.get_card(pid)
                    if p and p.is_basic and p.is_pokemon and _hp_val(p) >= 120 and p.name not in {d.name for d in selected_dragons}:
                        candidates.append(p)

        colorless_basics = [
            p for p in self.idx.basic_pokemon_by_type.get('Colorless', [])
            if p.is_basic and _hp_val(p) >= 180 and p.name not in {d.name for d in selected_dragons}
            and is_pokemon_attack_compatible(p, ['Grass', 'Fire', 'Water', 'Lightning', 'Psychic', 'Fighting', 'Darkness', 'Metal'])
        ]
        candidates.extend(colorless_basics)
        if candidates:
            candidates.sort(key=lambda c: (_hp_val(c) * 0.6 + _damage_val(c) * 1.2 + (60 if c.is_ex or c.is_mega_ex else 0)), reverse=True)
            return candidates[0]
        return None
    def _select_energy(
        self, energy_types: List[str], archetype: ArchetypeTemplate,
        combo_type: str, pokemon_cards: List[int], rng: random.Random
    ) -> Tuple[List[int], dict]:
        """Select 20-25 energy cards with Attack-Derived Energy Assignment."""
        target_count = archetype.energy_count
        selected: List[int] = []
        counts = Counter()
        info = {'special_energies': [], 'basic_energies': []}

        special_to_add: List[Tuple[int, int, str]] = []

        if combo_type == 'team_rocket':
            # Team Rocket's Energy provides {P} and {D}
            special_to_add.append((TEAM_ROCKET_ENERGY_ID, 4, "Team Rocket's Energy"))

        elif combo_type == 'dragon':
            # 4 copies of Prism Energy (provides all colors for Basic Pokemon)
            prism = self.idx.get_card(PRISM_ENERGY_ID)
            if prism:
                special_to_add.append((PRISM_ENERGY_ID, 4, 'Prism Energy (Universal Basic)'))

        else:
            # Standard single/dual/triple
            pokemon_types_in_deck = Counter()
            for cid in pokemon_cards:
                p = self.idx.get_card(cid)
                if p and p.is_pokemon:
                    pokemon_types_in_deck[p.type] += 1

            for et, se_id in SPECIAL_ENERGY_MAP.items():
                if et in energy_types and pokemon_types_in_deck.get(et, 0) >= 3:
                    special_to_add.append((se_id, 4, f'{et} Special Energy'))

        # Add Special Energies
        for se_id, se_count, reason in special_to_add:
            actual = min(se_count, 4 - counts.get(se_id, 0))
            if actual > 0:
                selected.extend([se_id] * actual)
                counts[se_id] = counts.get(se_id, 0) + actual
                info['special_energies'].append({'id': se_id, 'count': actual, 'reason': reason})

        # ── Basic Energy Selection ─────────────────────────────────────────
        remaining = target_count - len(selected)
        if remaining > 0:
            if combo_type == 'dragon':
                # Attack-Derived Energy Split: collect exact elemental energy types from chosen attacks
                needed_types = set()
                for cid in set(pokemon_cards):
                    p = self.idx.get_card(cid)
                    if p and p.attacks:
                        for atk in p.attacks:
                            needed_types.update(extract_attack_required_types(atk.get('cost', '')))

                all_types = sorted(list(needed_types)) if needed_types else ['Fire', 'Psychic', 'Water']
                # User constraint: No more than 3 basic energy types in Dragon decks!
                active_types = all_types[:3]
                be_per_type = remaining // len(active_types)
                be_remainder = remaining % len(active_types)

                for idx_et, et in enumerate(active_types):
                    be_card = self._get_basic_energy_card(et)
                    if be_card:
                        cnt = be_per_type + (1 if idx_et < be_remainder else 0)
                        selected.extend([be_card] * cnt)
                        info['basic_energies'].append({'type': et, 'count': cnt})

            elif combo_type == 'team_rocket':
                # Team Rocket uses Psychic and Darkness basic energy
                tr_types = ['Psychic', 'Darkness']
                be_per_type = remaining // len(tr_types)
                be_remainder = remaining % len(tr_types)
                for idx_et, et in enumerate(tr_types):
                    be_card = self._get_basic_energy_card(et)
                    if be_card:
                        cnt = be_per_type + (1 if idx_et < be_remainder else 0)
                        selected.extend([be_card] * cnt)
                        info['basic_energies'].append({'type': et, 'count': cnt})

            else:
                # Standard single/dual/triple
                be_per_type = remaining // len(energy_types)
                be_remainder = remaining % len(energy_types)
                for idx_et, et in enumerate(energy_types):
                    be_card = self._get_basic_energy_card(et)
                    if be_card:
                        cnt = be_per_type + (1 if idx_et < be_remainder else 0)
                        selected.extend([be_card] * cnt)
                        info['basic_energies'].append({'type': et, 'count': cnt})

        return selected[:target_count], info

    def _get_basic_energy_card(self, energy_type: str) -> Optional[int]:
        """Get the card ID for a basic energy of the given type."""
        type_to_id = {
            'Grass': 1, 'Fire': 2, 'Water': 3, 'Lightning': 4,
            'Psychic': 5, 'Fighting': 6, 'Darkness': 7, 'Metal': 8,
        }
        for c in self.idx.basic_energies:
            if c.type == energy_type:
                return c.card_id
        return type_to_id.get(energy_type, 1)

    def _select_trainers(
        self, energy_types: List[str], archetype: ArchetypeTemplate,
        combo_type: str, target_count: int, pokemon_cards: List[int],
        rng: random.Random
    ) -> Tuple[List[int], dict]:
        """Select trainer cards filling remaining deck slots."""
        selected: List[int] = []
        counts = Counter()
        has_ace_spec = False
        info = {'cards': [], 'ace_spec': None, 'rare_candy': 0, 'energy_search': 0}

        # 1. Energy Search (ID 1119) x2 in EVERY deck
        for _ in range(2):
            selected.append(ENERGY_SEARCH_ID)
            counts[ENERGY_SEARCH_ID] += 1
        info['energy_search'] = 2

        # 2. Rare Candy ONLY if deck actually contains at least one Stage 2 Pokemon
        has_actual_stage2 = any(
            self.idx.get_card(cid) and (self.idx.get_card(cid).is_stage2 or getattr(self.idx.get_card(cid), 'stage2', False))
            for cid in pokemon_cards
        )
        if has_actual_stage2 and archetype.evolution_mode in ('stage2', 'stage2_ex', 'stage2_mega', 'mega_stage_2_ex'):
            selected.append(RARE_CANDY_ID)
            counts[RARE_CANDY_ID] += 1
            info['rare_candy'] = 1

        # 3. ACE SPEC card
        ace_id = self._pick_ace_spec(archetype, combo_type, energy_types)
        if ace_id:
            selected.append(ace_id)
            counts[ace_id] += 1
            has_ace_spec = True
            info['ace_spec'] = {'id': ace_id, 'name': self.idx.get_card(ace_id).name if self.idx.get_card(ace_id) else '?'}

        # 4. Mechanic cards for special archetypes
        if archetype.mechanic_cards:
            mechanic_ids = self._pick_mechanic_cards(archetype, combo_type, rng)
            for mid in mechanic_ids:
                if len(selected) >= target_count:
                    break
                if counts.get(mid, 0) < 4:
                    selected.append(mid)
                    counts[mid] += 1
                    info['cards'].append({'id': mid, 'reason': 'mechanic'})

        # 5. Fill remaining with best trainers
        remaining = target_count - len(selected)
        if remaining > 0:
            filler = self._pick_trainer_filler(energy_types, archetype, remaining, counts, has_ace_spec, rng)
            for fid in filler:
                selected.append(fid)
                counts[fid] += 1
            info['cards'].extend([{'id': fid, 'reason': 'filler'} for fid in filler])

        return selected[:target_count], info

    def _pick_ace_spec(self, archetype: ArchetypeTemplate, combo_type: str,
                       energy_types: List[str]) -> Optional[int]:
        # Mandatory Energy Search Pro (ID 1100) for Dragon and Triple decks
        if combo_type in ('dragon', 'triple'):
            return ENERGY_SEARCH_PRO_ID
        if archetype.preferred_ace_specs:
            for ace_id in archetype.preferred_ace_specs:
                card = self.idx.get_card(ace_id)
                if card and card.is_ace_spec:
                    return ace_id
        return 1080

    def _pick_mechanic_cards(self, archetype: ArchetypeTemplate, combo_type: str, rng: random.Random) -> List[int]:
        results = []
        for tag in archetype.mechanic_cards[:3]:
            if tag in ('stadium_search', 'category_stadium'):
                stadiums = self.idx.get_trainer_by_tag('stadium', 'Trainer-Stadium')
                if not stadiums:
                    stadiums = [c for c in self.idx.cards.values() if c.category == 'Trainer-Stadium']
                if stadiums:
                    best = max(stadiums, key=lambda c: len(c.effect_tags))
                    results.append(best.card_id)
            elif tag == 'heal':
                heal_trainers = self.idx.get_trainer_by_tag('heal', 'Trainer-Item')
                if heal_trainers:
                    results.append(heal_trainers[0].card_id)
            elif tag == 'damage_counter':
                dc_trainers = self.idx.get_trainer_by_tag('damage_counter', 'Trainer-Item')
                if dc_trainers:
                    results.append(dc_trainers[0].card_id)
        return results[:3]

    def _pick_trainer_filler(
        self, energy_types: List[str], archetype: ArchetypeTemplate,
        count: int, existing_counts: Counter, has_ace_spec: bool, rng: random.Random
    ) -> List[int]:
        all_trainers = [c for c in self.idx.cards.values() if c.is_trainer and not c.is_ace_spec]
        scored = []
        for t in all_trainers:
            score = 50.0
            tags = set(t.effect_tags)

            for idx, prio_tag in enumerate(archetype.trainer_tag_priorities):
                if prio_tag in tags:
                    score += (100.0 - idx * 15.0)

            if 'draw' in tags: score += 30.0
            if 'search' in tags: score += 25.0
            if 'switch_pivot' in tags: score += 20.0
            if 'gust' in tags: score += 20.0
            if t.category == 'Trainer-Supporter': score += 15.0
            if t.category == 'Trainer-Stadium': score += 10.0

            scored.append((score + rng.gauss(0, 2), t))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for score, t in scored:
            if len(result) >= count:
                break
            cid = t.card_id
            if existing_counts.get(cid, 0) >= 4:
                continue
            copies = min(4 - existing_counts.get(cid, 0), count - len(result))
            copies = min(4 if score >= 100 else 2, copies)
            result.extend([cid] * copies)
            existing_counts[cid] = existing_counts.get(cid, 0) + copies

        return result[:count]
