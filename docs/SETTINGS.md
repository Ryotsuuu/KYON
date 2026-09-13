# ⚙️ KYON Sovereign Configuration & Settings Reference Manual

```
███████╗███████╗████████╗████████╗██╗███╗   ██╗ ██████╗ ███████╗
██╔════╝██╔════╝╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝ ██╔════╝
███████╗█████╗     ██║      ██║   ██║██╔██╗ ██║██║  ███╗███████╗
╚════██║██╔══╝     ██║      ██║   ██║██║╚██╗██║██║   ██║╚════██║
███████║███████╗   ██║      ██║   ██║██║ ╚████║╚██████╔╝███████║
╚══════╝╚══════╝   ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝
```

---

## 📑 Table of Contents
1. [Configuration Overview](#1-configuration-overview)
2. [`ptcg-system/settings.json` Schema](#2-ptcg-systemsettingsjson-schema)
3. [Energy Combo Toggles (`energy_combos`)](#3-energy-combo-toggles-energy_combos)
4. [Archetype Strategy Templates (`archetypes`)](#4-archetype-strategy-templates-archetypes)
5. [Combo-Specific Archetype Overrides](#5-combo-specific-archetype-overrides)
6. [Runtime Registry & Catalog Storage Pathways](#6-runtime-registry--catalog-storage-pathways)
7. [Deck Audit & Conservation Validation](#7-deck-audit--conservation-validation)
8. [Common Configuration Recipes](#8-common-configuration-recipes)

---

## 1. Configuration Overview

The [`ptcg-system/settings.json`](file:///e:/PTCG%20Soverign%20Trainer/ptcg%20sovegin%20trainer/ptcg-system/settings.json) configuration file governs the entire compilation and synthesis scope of the KYON deck building pipeline (`python ptcg.py build`). It defines which elemental combinations and tactical archetypes are compiled into the 519-agent registry.

---

## 2. `ptcg-system/settings.json` Schema

```json
{
  "energy_combos": {
    "single": true,
    "dual": true,
    "triple": true,
    "team_rocket": true,
    "dragon": true
  },
  "archetypes": {
    "balanced": true,
    "aggro": true,
    "stall": true,
    "stage_2": true,
    "stage_1": true,
    "stage_2_ex": true,
    "stage_1_ex": true,
    "mega_stage_2_ex": true,
    "mega_stage_1_ex": true,
    "ability_heal": false,
    "prize_rush": true,
    "poison_heavy": false,
    "burn_heavy": false,
    "sleep_heavy": false,
    "paralyze_heavy": false,
    "stadium_control": true,
    "energy_transfer": false,
    "damage_counter": true
  },
  "archetypes_for_triple": {
    "balanced": true,
    "aggro": true,
    "stall": true,
    "stage_2": true,
    "stage_1": true,
    "stage_2_ex": true,
    "stage_1_ex": true,
    "mega_stage_2_ex": true,
    "mega_stage_1_ex": true,
    "ability_heal": true,
    "prize_rush": true,
    "stadium_control": true,
    "damage_counter": true
  },
  "archetypes_for_team_rocket": {
    "balanced": true,
    "aggro": true,
    "stall": true,
    "stage_2": true,
    "stage_1": true,
    "stage_2_ex": true,
    "stage_1_ex": true,
    "mega_stage_2_ex": true,
    "mega_stage_1_ex": true,
    "ability_heal": true,
    "prize_rush": true
  },
  "archetypes_for_dragon": {
    "balanced": true,
    "aggro": true,
    "stall": true,
    "stage_2": true,
    "stage_1": true,
    "stage_2_ex": true,
    "stage_1_ex": true,
    "mega_stage_2_ex": true,
    "mega_stage_1_ex": true,
    "ability_heal": true,
    "prize_rush": true,
    "damage_counter": true
  },
  "conservation_audit": false
}
```

---

## 3. Energy Combo Toggles (`energy_combos`)

| Combo Key | Type Count | Resulting Agent Volume | Description |
| :--- | :---: | :---: | :--- |
| `"single"` | 8 types | **96 Agents** ($8 \times 12$) | Single elemental energy decks (Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal). |
| `"dual"` | 28 pairs | **336 Agents** ($28 \times 12$) | All two-element synergy combinations ($C(8, 2) = 28$). |
| `"triple"` | 5 pools | **60 Agents** ($5 \times 12$) | Curated three-element combinations with flexible energy costs. |
| `"team_rocket"` | 1 special | **11 Agents** | Darkness + Psychic themed decks using Rocket disruption cards. |
| `"dragon"` | 1 special | **12 Agents** | Multi-type Dragon attacker decks with Energy Search acceleration. |

---

## 4. Archetype Strategy Templates (`archetypes`)

Controls archetype template generation for single and dual combos:

| Archetype Key | Active Status | Tactical Strategy | Composition Template |
| :--- | :---: | :--- | :--- |
| `"balanced"` | `true` | Standard midrange tempo. | 22 Energy / 20 Trainer / 18 Pokémon |
| `"aggro"` | `true` | Fast turn 1–2 damage beatdown. | 20 Energy / 22 Trainer / 18 Pokémon |
| `"stall"` | `true` | High-HP tanking & healing. | 25 Energy / 17 Trainer / 18 Pokémon |
| `"stage_2"` | `true` | Stage 2 lines with Rare Candy. | 21 Energy / 21 Trainer / 18 Pokémon |
| `"stage_1"` | `true` | Fast Stage 1 evolution pressure. | 21 Energy / 21 Trainer / 18 Pokémon |
| `"stage_2_ex"` | `true` | Stage 2 EX carries with Basic EX fillers. | 21 Energy / 21 Trainer / 18 Pokémon |
| `"stage_1_ex"` | `true` | Fast 2-prize Stage 1 EX attackers. | 21 Energy / 21 Trainer / 18 Pokémon |
| `"mega_stage_2_ex"` | `true` | 300+ HP Mega EX carries. | 22 Energy / 20 Trainer / 18 Pokémon |
| `"mega_stage_1_ex"` | `true` | Mega EX single evolution beatdown. | 22 Energy / 20 Trainer / 18 Pokémon |
| `"prize_rush"` | `true` | Aggressive 2-prize extraction pace. | 20 Energy / 22 Trainer / 18 Pokémon |
| `"stadium_control"` | `true` | Stadium board lock and damage mitigation. | 22 Energy / 20 Trainer / 18 Pokémon |
| `"damage_counter"` | `true` | Direct damage placement bypassing active defenses. | 22 Energy / 20 Trainer / 18 Pokémon |
| `"ability_heal"` | `false` | Disabled for single/dual due to card pool constraints. | Enabled in triple/dragon |

---

## 5. Combo-Specific Archetype Overrides

- **`archetypes_for_triple`:** Enables `"ability_heal": true` due to the broader card pool available across three energy types.
- **`archetypes_for_team_rocket`:** Enables 11 archetypes tailored to Team Rocket Pokémon abilities and disruption mechanics.
- **`archetypes_for_dragon`:** Enables 12 archetypes supporting multi-energy costs and Dragon Pokémon traits.

---

## 6. Runtime Registry & Catalog Storage Pathways

When `python ptcg.py build` executes, it generates three primary data stores:

| Filepath | Format | Purpose | Key Attributes |
| :--- | :---: | :--- | :--- |
| `ptcg-system/agents_registry.json` | JSON | Master Registry | Complete 60-card integer ID lists, W-L-D match records, metadata. |
| `ptcg-system/agent_catalog.json` | JSON | Fast Lookup Index | Lightweight metadata for fast CLI filtering and search. |
| `ptcg-system/custom_agents/` | Directory | Custom User Decks | Individual CSV files created via `ptcg deck create`. |
| `deck.csv` | CSV | Default Deck | First legal 60-card deck formatted for direct Kaggle submission. |

---

## 7. Deck Audit & Conservation Validation

When `"conservation_audit": true` is enabled in `settings.json`, the compiler performs an additional card pool conservation audit, verifying that card allocations across all generated decks conform to set-level pool distributions.

---

## 8. Common Configuration Recipes

### Recipe 1: Fast Development Build (Single Energy Only)
To compile only single-energy agents for fast local testing:
```json
{
  "energy_combos": {
    "single": true,
    "dual": false,
    "triple": false,
    "team_rocket": false,
    "dragon": false
  }
}
```
*Builds 96 agents in under 1.5 seconds.*

### Recipe 2: Full Tournament Master Suite
Enable all 5 combo types to generate the full 519-agent registry:
```bash
python ptcg.py build
```