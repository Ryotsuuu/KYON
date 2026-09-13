# Card Data Extraction Report

## Overview
- Total cards analyzed: 1267
- Cards with conditional effects: 327
- Cards with effect tags: 546
- Cards with archetype fits: 802
- Cards with GA mutation risk: 510
- Cross-reference synergies found: 9227
- Team Rocket Pokemon: 52
- Dragon-type Pokemon: 35
- Colorless-type Pokemon: 104 (energy-pure: 102)
- Cards referencing Team Rocket Energy by name: 3
- Cards with low confidence: 0

## Rankings Summary
- Top 15 best cards: Leafeon ex (#236) through Okidogi ex (#138)
- Worst 10 cards: Cinccino (#556) through Diggersby (#1074)
- Least useful for GA: Metang (#86) through Cinccino (#556)
- Methodology: Scored by: effectiveness_rating (weighted), deckbuilding_priority, effect_tag_density, archetype_versatility, synergy_count, GA_risk_penalty, category_bonuses, HP_bonus

## Effect Tag Distribution
- coin_flip: 145 cards
- deck_search: 103 cards
- pokemon_search: 72 cards
- energy_search: 38 cards
- heal_fixed: 36 cards
- cant_attack_next_turn: 34 cards
- damage_prevention: 30 cards
- retreat_lock: 29 cards
- stadium_owner_only: 26 cards
- draw_cards: 25 cards
- inflict_paralyzed: 23 cards
- self_energy_discard: 22 cards
- heal_self: 21 cards
- inflict_poisoned: 21 cards
- inflict_confused: 20 cards
- bench_protection: 17 cards
- mill_opponent: 17 cards
- trainer_search: 16 cards
- inflict_burned: 15 cards
- opponent_energy_discard: 15 cards
- self_switch: 14 cards
- inflict_asleep: 13 cards
- hand_reveal: 11 cards
- bench_spread_damage: 10 cards
- forced_opponent_switch: 8 cards
- pokemon_search_to_hand: 7 cards
- energy_acceleration: 6 cards
- heal_full: 5 cards
- pokemon_search_to_bench: 5 cards
- stadium_removal: 4 cards
- retreat_cost_reduction: 4 cards
- prize_reduction: 2 cards
- heal_conditional: 2 cards
- energy_from_discard: 2 cards
- hand_disruption: 2 cards
- heal_bench: 1 cards

## Archetype Distribution
- stage_1: 295 card-archetype assignments
- aggro: 204 card-archetype assignments
- coin_flip: 145 card-archetype assignments
- stage_2: 83 card-archetype assignments
- balanced: 78 card-archetype assignments
- stall: 70 card-archetype assignments
- active_bench_shifter: 62 card-archetype assignments
- team_rocket: 52 card-archetype assignments
- paralyze_heavy: 43 card-archetype assignments
- stage_1_ex: 40 card-archetype assignments
- dragon_combo: 35 card-archetype assignments
- stadium_control: 30 card-archetype assignments
- stage_2_ex: 23 card-archetype assignments
- poison_heavy: 21 card-archetype assignments
- deck_out: 17 card-archetype assignments
- burn_heavy: 15 card-archetype assignments
- hand_control: 13 card-archetype assignments
- sleep_heavy: 13 card-archetype assignments
- mega_stage_2_ex: 10 card-archetype assignments
- mega_stage_1_ex: 10 card-archetype assignments
- heavy_energy: 6 card-archetype assignments
- ability_burn: 2 card-archetype assignments
- ability_poison: 2 card-archetype assignments
- triple_energy: 1 card-archetype assignments

## Confidence Distribution
- medium: 510 cards
- high: 475 cards
- low: 282 cards

## Cross-Reference Synergy Types
- energy_type_compatibility: 8416 synergy links
- search_evo_target: 222 synergy links
- evolution_chain: 203 synergy links
- stadium_synergy: 130 synergy links
- stall_combo: 123 synergy links
- team_rocket_synergy: 63 synergy links
- dragon_energy_solution: 32 synergy links
- energy_accel_target: 22 synergy links
- status_setup_payoff: 16 synergy links

## Unresolved Items (engine cross-check)
- Effect text parsing is regex-based; edge cases in card wording may be missed
- Conditional effect extraction covers known patterns but may miss novel precondition formats
- Strategic analysis is rule-based scoring; actual metagame viability may differ
- Cross-reference synergies are mechanical (shared type, status setup/payoff); creative combos may be missed
- The Japanese character for Dragon type (竜) is normalized to "Dragon" — verify game engine uses same mapping

## Called-Out Cards Verification
- ID 431 (Team Rocket's Mewtwo ex): PRESENT
  - Bench count condition: YES (expected: 4+ Team Rocket Pokemon)
  - GA mutation risk: YES
- ID 904 (Mega Dragonite ex): PRESENT
  - Type: Dragon (expected: Dragon)
  - Stage: Stage 2 (expected: Stage 2)
  - Evolution from: Dragonair (expected: Dragonair)
  - Attack costs: ['{W}{L}{L}']
- ID 236 (Leafeon ex): PRESENT
  - Has heal tag: YES
- ID 12 (Legacy Energy): PRESENT
- ID 15 (Team Rocket's Energy): PRESENT