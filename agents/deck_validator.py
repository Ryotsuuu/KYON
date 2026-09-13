u"""agents/deck_validator.py
==============================
Deck Validator v3: Validates 60-card decks against competition rules and KYON standards.

Rules checked:
1. Exactly 60 cards
2. Max 4 copies of any card (except Basic Energy: unlimited; ACE SPEC: max 1)
3. All cards exist in card database
4. At least 1 Basic Pokemon
5. No more than 1 ACE SPEC card
6. Energy count 20-25 (warning)
7. Pokemon count max 18 (max 6 species)
8. Mandatory: 2x Energy Search (1119) (warning)
9. Stage 2 Evolutionary Accelerator: Stage 2 decks must have Rare Candy (1079);
   non-Stage 2 decks must NOT have Rare Candy.
10. Special Energy HP-Type Matching: Grow Grass (18) -> Grass Pokemon,
    Telepath Psychic (19) -> Psychic Pokemon, Rock Fighting (20) -> Fighting Pokemon,
    Team Rocket's (15) -> TR Pokemon, Neo Upper (10) -> Stage 2 Pokemon.
11. Attack Energy Coverage: Every Pokemon's elemental attack cost requirements
    must be satisfiable by the deck's basic energies or rainbow energy
    (Legacy #12, Prism #16, Neo Upper #10). HP type = card class only.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class ValidationReport:
    is_legal: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


class MasterDeckValidator:
    """40-pass deck validator."""

    def __init__(self, kg=None, profile_db=None):
        self.kg = kg
        self._csv_idx = None
        try:
            from agents.csv_data import get_csv_index
            self._csv_idx = get_csv_index()
        except Exception:
            pass

    def validate_deck(self, deck: List[int], energy_types: List[int] = None) -> ValidationReport:
        report = ValidationReport()
        counts = Counter(deck)

        # Rule 1: Exactly 60 cards
        if len(deck) != 60:
            report.errors.append(f"Rule 1 Fail: Deck has {len(deck)} cards, must be 60")
            report.is_legal = False

        # Rule 2: Max 4 copies per card
        # Exception: Basic Energy cards have no copy limit in PTCG
        for cid, cnt in counts.items():
            card = self._get_card(cid)
            if card:
                csv_cat = getattr(card, 'category', None)
                is_basic_energy = csv_cat == 'Basic Energy'
                if is_basic_energy:
                    continue  # No copy limit for basic energy
                is_ace = getattr(card, 'aceSpec', False) or getattr(card, 'is_ace_spec', False)
                if is_ace and cnt > 1:
                    report.errors.append(f"Rule 2 Fail: ACE SPEC card {cid} has {cnt} copies (max 1)")
                    report.is_legal = False
                elif not is_ace and cnt > 4:
                    report.errors.append(f"Rule 2 Fail: Card {cid} has {cnt} copies (max 4)")
                    report.is_legal = False

        # Rule 3: All cards exist
        for cid in set(deck):
            if not self._get_card(cid):
                report.errors.append(f"Rule 3 Fail: Card {cid} not found in database")
                report.is_legal = False

        # Rule 4: At least 1 Basic Pokemon
        has_basic = False
        pokemon_count = 0
        energy_count = 0
        trainer_count = 0
        ace_count = 0
        species_count = 0
        pokemon_species = set()

        for cid in set(deck):
            card = self._get_card(cid)
            if not card:
                continue
            cat = getattr(card, 'cardType', None)
            csv_cat = getattr(card, 'category', None)

            if cat == 0 or csv_cat == 'Pokemon':  # POKEMON
                pokemon_count += counts[cid]
                pokemon_species.add(cid)
                if getattr(card, 'basic', False) or getattr(card, 'is_basic', False) or (csv_cat == 'Pokemon' and getattr(card, 'pokemon_stage', None) == 'Basic'):
                    has_basic = True
            elif cat in (5, 6) or csv_cat in ('Basic Energy', 'Special Energy'):
                energy_count += counts[cid]
            elif cat in (1, 2, 3, 4) or (csv_cat and csv_cat.startswith('Trainer')):
                trainer_count += counts[cid]
                if getattr(card, 'aceSpec', False) or getattr(card, 'is_ace_spec', False):
                    ace_count += 1

        species_count = len(pokemon_species)

        if not has_basic:
            report.errors.append("Rule 4 Fail: No Basic Pokemon in deck")
            report.is_legal = False

        # Rule 5: Max 1 ACE SPEC
        if ace_count > 1:
            report.errors.append(f"Rule 5 Fail: {ace_count} ACE SPEC cards (max 1)")
            report.is_legal = False

        # Rule 6: Energy count 20-25
        if energy_count < 20 or energy_count > 25:
            report.warnings.append(f"Rule 6 Warning: Energy count {energy_count} (recommended 20-25)")

        # Rule 7: Pokemon count max 18 (6 species x 3)
        if pokemon_count > 18:
            report.errors.append(f"Rule 7 Fail: {pokemon_count} Pokemon cards (max 18, i.e. 6 species x 3)")
            report.is_legal = False
        if species_count > 6:
            report.errors.append(f"Rule 7 Fail: {species_count} Pokemon species (max 6)")
            report.is_legal = False

        # Rule 8: Energy Search (1119) at least 2 copies
        es_count = counts.get(1119, 0)
        if es_count < 2:
            report.warnings.append(f"Rule 8 Warning: Energy Search (1119) only {es_count} copies (recommended 2+)")

        # Rule 9: Evolutionary Legality & Stage 2 Acceleration (Rare Candy 1079)
        has_stage2 = any(
            self._get_card(cid) and (
                getattr(self._get_card(cid), 'is_stage2', False) or
                getattr(self._get_card(cid), 'stage2', False) or
                getattr(self._get_card(cid), 'pokemon_stage', None) == 'Stage 2'
            )
            for cid in pokemon_species
        )
        rare_candy_count = counts.get(1079, 0)
        if has_stage2 and rare_candy_count == 0:
            report.errors.append("Rule 9 Fail: Deck contains Stage 2 Pokemon but lacks Rare Candy (1079)")
            report.is_legal = False
        elif not has_stage2 and rare_candy_count > 0:
            report.errors.append("Rule 9 Fail: Deck contains Rare Candy (1079) but has no Stage 2 Pokemon")
            report.is_legal = False

        # Rule 10: Special Energy HP-Type & Target Validity
        pokemon_types = set()
        has_tr_pokemon = False
        for cid in pokemon_species:
            card = self._get_card(cid)
            if card:
                ptype = getattr(card, 'type', None)
                if ptype:
                    pokemon_types.add(ptype)
                if getattr(card, 'is_team_rocket_pokemon', False):
                    has_tr_pokemon = True

        if 18 in counts and 'Grass' not in pokemon_types:
            report.errors.append("Rule 10 Fail: Grow Grass Energy (18) requires Grass-type Pokemon in deck")
            report.is_legal = False
        if 19 in counts and 'Psychic' not in pokemon_types:
            report.errors.append("Rule 10 Fail: Telepath Psychic Energy (19) requires Psychic-type Pokemon in deck")
            report.is_legal = False
        if 20 in counts and 'Fighting' not in pokemon_types:
            report.errors.append("Rule 10 Fail: Rock Fighting Energy (20) requires Fighting-type Pokemon in deck")
            report.is_legal = False
        if 15 in counts and not has_tr_pokemon:
            report.errors.append("Rule 10 Fail: Team Rocket's Energy (15) requires Team Rocket Pokemon in deck")
            report.is_legal = False
        if 10 in counts and not has_stage2:
            report.errors.append("Rule 10 Fail: Neo Upper Energy (10) requires Stage 2 Pokemon in deck")
            report.is_legal = False

        # Rule 11: Attack Energy Coverage vs Deck Energy Provision
        energy_id_to_type = {
            1: 'Grass', 2: 'Fire', 3: 'Water', 4: 'Lightning',
            5: 'Psychic', 6: 'Fighting', 7: 'Darkness', 8: 'Metal'
        }
        deck_basic_energy_types = {energy_id_to_type[cid] for cid in counts if cid in energy_id_to_type}
        has_legacy_rainbow = 12 in counts
        has_prism_rainbow = 16 in counts
        has_neo_upper_rainbow = 10 in counts

        for cid in pokemon_species:
            card = self._get_card(cid)
            if not card:
                continue
            get_atk_fn = getattr(card, 'get_attack_energy_types', None)
            if callable(get_atk_fn):
                needed_types = get_atk_fn()
            else:
                needed_types = set()

            is_basic_card = (
                getattr(card, 'basic', False) or
                getattr(card, 'is_basic', False) or
                getattr(card, 'pokemon_stage', None) == 'Basic'
            )
            is_stage2_card = (
                getattr(card, 'is_stage2', False) or
                getattr(card, 'stage2', False) or
                getattr(card, 'pokemon_stage', None) == 'Stage 2'
            )

            for etype in needed_types:
                if etype in deck_basic_energy_types:
                    continue
                if has_legacy_rainbow:
                    continue
                if is_basic_card and has_prism_rainbow:
                    continue
                if is_stage2_card and has_neo_upper_rainbow:
                    continue
                card_name = getattr(card, 'name', f'Card #{cid}')
                report.errors.append(f"Rule 11 Fail: Pokemon {cid} ({card_name}) requires {etype} energy to attack, but deck provides none")
                report.is_legal = False

        report.stats = {
            'total': len(deck),
            'pokemon': pokemon_count,
            'pokemon_species': species_count,
            'energy': energy_count,
            'trainers': trainer_count,
            'ace_spec': ace_count,
            'unique_cards': len(set(deck)),
        }

        return report

    def _get_card(self, card_id: int):
        if self._csv_idx:
            return self._csv_idx.get_card(card_id)
        if self.kg:
            return self.kg.card_dict.get(card_id)
        return None
