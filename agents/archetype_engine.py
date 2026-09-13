"""agents/archetype_engine.py
==============================
Archetype Engine v2: Manages strategy archetypes with explicit card composition templates.

Deck composition rules (v2):
- Total Pokemon cards: Exactly 18 (max 6 species x 3 copies each)
  - Only 1 evolution chain (Basic->Stage1 or Basic->Stage1->Stage2), max 2 Rare Candy
  - Remaining slots: EX, Mega EX, or other high HP/damage Pokemon
- Total Energy cards: 20-25 (Basic + Special Energy)
  - At least 4 Special Energy when applicable (Grow Grass, Telepath Psychic, Rock Fighting, Team Rocket's)
  - ID 1119 Energy Search: at least 2 copies in every deck
- Total Trainer cards: 17-22 (Stadium, Item, ACE SPEC, Supporter, Tool)
  - At least 1 Rare Candy for stage_2 decks (max 2 if 2 different stage2 lines)
  - At least 1 ACE SPEC card
  - For dragon/triple: ACE SPEC 1100 (Energy Search Pro)
- Special archetype rules:
  - ability_heal/poison_heavy/burn_heavy/sleep_heavy/paralyze_heavy/stadium_control/energy_transfer/damage_counter:
    Include 1-3 cards (Pokemon, stadium, tool, or trainer) that enable the named mechanic
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@dataclass
class ArchetypeTemplate:
    name: str
    description: str
    energy_count: int         # 20-25
    trainer_count: int        # 17-22  
    pokemon_count: int = 18   # Always 18 (max 6 species x 3 copies)

    trainer_tag_priorities: List[str] = field(default_factory=list)
    preferred_ace_specs: List[int] = field(default_factory=list)  # card IDs
    mechanic_cards: List[str] = field(default_factory=list)  # tags for mechanic-specific cards

    evolution_mode: Optional[str] = None
    # "stage2" -> exactly one full 3-stage chain + 1 Rare Candy
    # "stage1" -> exactly one 2-stage chain
    # "stage2_ex" -> stage2 chain + filler EX-only
    # "stage1_ex" -> stage1 chain + filler EX-only
    # "stage2_mega" -> stage2 chain + filler Mega-EX-only
    # "stage1_mega" -> stage1 chain + filler Mega-EX-only
    # None -> generic selection


# ============================================================
# ARCHETYPES FOR SINGLE & DUAL ENERGY COMBOS
# ============================================================
ARCHETYPES_SINGLE_DUAL: Dict[str, ArchetypeTemplate] = {
    "balanced": ArchetypeTemplate(
        name="balanced",
        description="Well-rounded ratio of active tanks, bench evolutions, draw trainers, and energy.",
        energy_count=22,
        trainer_count=20,
        trainer_tag_priorities=["draw", "search", "switch_pivot", "gust"],
        preferred_ace_specs=[1080],  # Unfair Stamp
    ),
    "aggro": ArchetypeTemplate(
        name="aggro",
        description="Fast damage output with low energy requirement and heavy gusting.",
        energy_count=20,
        trainer_count=22,
        trainer_tag_priorities=["gust", "search", "switch_pivot", "draw"],
        preferred_ace_specs=[1088],  # Prime Catcher
    ),
    "stall": ArchetypeTemplate(
        name="stall",
        description="High HP tanks with healing, energy disruption, and high energy reserve.",
        energy_count=25,
        trainer_count=17,
        trainer_tag_priorities=["heal", "tank_support", "draw", "switch_pivot"],
        preferred_ace_specs=[1096],  # Poke Vital A
    ),
    "stage_2": ArchetypeTemplate(
        name="stage_2",
        description="One full Basic->Stage1->Stage2 evolution chain (9 cards) anchoring the deck, plus filler species. Mandatory 1 Rare Candy.",
        energy_count=21,
        trainer_count=21,
        trainer_tag_priorities=["rare_candy_synergy", "pokemon_search", "draw", "switch_pivot"],
        preferred_ace_specs=[1085],  # Awakening Drum
        evolution_mode="stage2",
    ),
    "stage_1": ArchetypeTemplate(
        name="stage_1",
        description="One Basic->Stage1 evolution chain (6 cards) anchoring the deck, plus 4 filler species.",
        energy_count=21,
        trainer_count=21,
        trainer_tag_priorities=["pokemon_search", "draw", "switch_pivot", "gust"],
        preferred_ace_specs=[1080],
        evolution_mode="stage1",
    ),
    "stage_2_ex": ArchetypeTemplate(
        name="stage_2_ex",
        description="Stage 2 evolution chain anchor + filler species restricted to Basic EX only. Mandatory Rare Candy.",
        energy_count=21,
        trainer_count=21,
        trainer_tag_priorities=["rare_candy_synergy", "pokemon_search", "gust", "draw"],
        preferred_ace_specs=[1088],
        evolution_mode="stage2_ex",
    ),
    "stage_1_ex": ArchetypeTemplate(
        name="stage_1_ex",
        description="Stage 1 evolution chain anchor + filler species restricted to Basic EX only.",
        energy_count=21,
        trainer_count=21,
        trainer_tag_priorities=["pokemon_search", "gust", "draw", "switch_pivot"],
        preferred_ace_specs=[1080],
        evolution_mode="stage1_ex",
    ),
    "mega_stage_2_ex": ArchetypeTemplate(
        name="mega_stage_2_ex",
        description="Stage 2 evolution chain anchor + filler species restricted to Mega EX only. Mandatory Rare Candy.",
        energy_count=22,
        trainer_count=20,
        trainer_tag_priorities=["rare_candy_synergy", "pokemon_search", "gust", "draw"],
        preferred_ace_specs=[1088],
        evolution_mode="stage2_mega",
    ),
    "mega_stage_1_ex": ArchetypeTemplate(
        name="mega_stage_1_ex",
        description="Stage 1 evolution chain anchor + filler species restricted to Mega EX only.",
        energy_count=22,
        trainer_count=20,
        trainer_tag_priorities=["pokemon_search", "gust", "draw", "switch_pivot"],
        preferred_ace_specs=[1088],
        evolution_mode="stage1_mega",
    ),
    "prize_rush": ArchetypeTemplate(
        name="prize_rush",
        description="Aggressive multi-prize taking with EX / Mega EX active tanks.",
        energy_count=20,
        trainer_count=22,
        trainer_tag_priorities=["search", "gust", "draw", "switch_pivot"],
        preferred_ace_specs=[1088],  # Prime Catcher
    ),
    "stadium_control": ArchetypeTemplate(
        name="stadium_control",
        description="Maintains Stadium control to buffer damage or grant free retreat. Include 1-3 stadium-related cards.",
        energy_count=22,
        trainer_count=20,
        trainer_tag_priorities=["stadium_search", "draw", "search", "switch_pivot"],
        preferred_ace_specs=[1247],  # Neutralization Zone (Stadium ACE SPEC)
        mechanic_cards=["stadium_search", "category_stadium"],
    ),
    "damage_counter": ArchetypeTemplate(
        name="damage_counter",
        description="Places damage counters via abilities or trainer effects. Include 1-3 damage counter cards.",
        energy_count=22,
        trainer_count=20,
        trainer_tag_priorities=["damage_counter", "draw", "search", "gust"],
        preferred_ace_specs=[1095],  # Dangerous Laser
        mechanic_cards=["damage_counter"],
    ),
}

# ============================================================
# ARCHETYPES FOR TRIPLE ENERGY COMBOS (adds ability_heal)
# ============================================================
ARCHETYPES_TRIPLE: Dict[str, ArchetypeTemplate] = {}
for _name, _tmpl in ARCHETYPES_SINGLE_DUAL.items():
    ARCHETYPES_TRIPLE[_name] = ArchetypeTemplate(
        name=_tmpl.name,
        description=_tmpl.description + " [triple energy variant]",
        energy_count=_tmpl.energy_count,
        trainer_count=_tmpl.trainer_count,
        trainer_tag_priorities=list(_tmpl.trainer_tag_priorities),
        preferred_ace_specs=list(_tmpl.preferred_ace_specs),
        mechanic_cards=list(_tmpl.mechanic_cards),
        evolution_mode=_tmpl.evolution_mode,
    )
# Add ability_heal for triple
ARCHETYPES_TRIPLE["ability_heal"] = ArchetypeTemplate(
    name="ability_heal",
    description="Prioritizes healing abilities and healing trainers. Include 1-3 heal cards. [triple variant]",
    energy_count=22,
    trainer_count=20,
    trainer_tag_priorities=["heal", "has_ability", "draw", "switch_pivot"],
    preferred_ace_specs=[1096],
    mechanic_cards=["heal"],
)

# ============================================================
# ARCHETYPES FOR TEAM ROCKET COMBOS (adds ability_heal)
# ============================================================
ARCHETYPES_TEAM_ROCKET: Dict[str, ArchetypeTemplate] = {}
_TR_ARCHETYPES = ["balanced", "aggro", "stall", "stage_2", "stage_1",
                  "stage_2_ex", "stage_1_ex", "mega_stage_2_ex", "mega_stage_1_ex",
                  "ability_heal", "stadium_control"]
for _name in _TR_ARCHETYPES:
    if _name in ARCHETYPES_SINGLE_DUAL:
        _tmpl = ARCHETYPES_SINGLE_DUAL[_name]
    else:
        _tmpl = ARCHETYPES_SINGLE_DUAL["balanced"]
    ARCHETYPES_TEAM_ROCKET[_name] = ArchetypeTemplate(
        name=_tmpl.name,
        description=_tmpl.description + " [team rocket variant]",
        energy_count=_tmpl.energy_count,
        trainer_count=_tmpl.trainer_count,
        trainer_tag_priorities=list(_tmpl.trainer_tag_priorities),
        preferred_ace_specs=list(_tmpl.preferred_ace_specs),
        mechanic_cards=list(_tmpl.mechanic_cards),
        evolution_mode=_tmpl.evolution_mode,
    )

# ============================================================
# ARCHETYPES FOR DRAGON COMBOS (adds ability_heal, damage_counter)
# ============================================================
ARCHETYPES_DRAGON: Dict[str, ArchetypeTemplate] = {}
_DR_ARCHETYPES = ["balanced", "aggro", "stall", "stage_2", "stage_1",
                  "stage_2_ex", "stage_1_ex", "mega_stage_2_ex", "mega_stage_1_ex",
                  "ability_heal", "stadium_control", "damage_counter"]
for _name in _DR_ARCHETYPES:
    if _name in ARCHETYPES_SINGLE_DUAL:
        _tmpl = ARCHETYPES_SINGLE_DUAL[_name]
    else:
        _tmpl = ARCHETYPES_SINGLE_DUAL["balanced"]
    # Dragon decks use ACE SPEC 1100 (Energy Search Pro)
    ace = [1100] if _name in ("balanced", "aggro", "stage_2", "stage_1", "stage_2_ex",
                              "stage_1_ex", "mega_stage_2_ex", "mega_stage_1_ex") else list(_tmpl.preferred_ace_specs)
    ARCHETYPES_DRAGON[_name] = ArchetypeTemplate(
        name=_tmpl.name,
        description=_tmpl.description + " [dragon variant]",
        energy_count=_tmpl.energy_count,
        trainer_count=_tmpl.trainer_count,
        trainer_tag_priorities=list(_tmpl.trainer_tag_priorities),
        preferred_ace_specs=ace,
        mechanic_cards=list(_tmpl.mechanic_cards),
        evolution_mode=_tmpl.evolution_mode,
    )


def get_archetype_template(name: str, combo_type: str = "single") -> ArchetypeTemplate:
    """Return ArchetypeTemplate by name and combo type, defaulting to 'balanced'."""
    if combo_type == "triple":
        return ARCHETYPES_TRIPLE.get(name, ARCHETYPES_TRIPLE["balanced"])
    elif combo_type == "team_rocket":
        return ARCHETYPES_TEAM_ROCKET.get(name, ARCHETYPES_TEAM_ROCKET["balanced"])
    elif combo_type == "dragon":
        return ARCHETYPES_DRAGON.get(name, ARCHETYPES_DRAGON["balanced"])
    else:
        return ARCHETYPES_SINGLE_DUAL.get(name, ARCHETYPES_SINGLE_DUAL["balanced"])


def get_all_archetypes_for_combo(combo_type: str) -> Dict[str, ArchetypeTemplate]:
    """Return all enabled archetypes for a given combo type."""
    if combo_type == "triple":
        return ARCHETYPES_TRIPLE
    elif combo_type == "team_rocket":
        return ARCHETYPES_TEAM_ROCKET
    elif combo_type == "dragon":
        return ARCHETYPES_DRAGON
    else:
        return ARCHETYPES_SINGLE_DUAL


# Backward compat alias
ALL_25_ARCHETYPES = ARCHETYPES_SINGLE_DUAL
