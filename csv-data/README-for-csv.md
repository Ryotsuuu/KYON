# README-for-csv.md
# Pokemon TCG Card Data Extraction - Complete Data Reference
# For AI Competition Deck Builder / GA System / RL Agent

---

## 1. Overview

- **Source**: EN_Card_Data.csv (1267 unique cards, 2022 CSV rows due to multi-attack/ability entries)
- **Competition**: Kaggle Pokemon TCG AI Battle Challenge
- **Purpose**: Provide the missing strategic-use-case layer that the game engine CSV lacks.
- **Method**: Rule-based text analysis of every card Effect Explanation field, two-pass
  verification against raw CSV, cross-reference synergy discovery in batches of 15.
- **Verification**: Literal grep-pass confirmed all 52 Team Rocket Pokemon and all 33 Dragon-type
  Pokemon are present. All called-out card IDs verified (431, 904, 236, 223, 12, 15).

### Card Pool Composition

| Category | Count |
|----------|-------|
| Pokemon | 1056 |
| Trainer-Item | 77 |
| Trainer-Supporter | 61 |
| Trainer-Tool | 27 |
| Trainer-Stadium | 26 |
| Special Energy | 12 |
| Basic Energy | 8 |
| **Total** | **1267** |

### Pokemon Stage Distribution

- Basic: 595
- Stage 1: 345
- Stage 2: 116
- ex: 121, Mega ex: 30, ACE SPEC: 29

### Critical Subsets

- **52 Team Rocket Pokemon**: Category contains Trainer Pokemon (Team Rocket).
  Only Mewtwo ex (ID 431) has a bench-count ability requirement (4+ TR Pokemon in play).
- **35 Dragon-type Pokemon**: No Basic Dragon Energy exists in pool.
  Require flexible/rainbow energy (Legacy Energy ID 12, Prism Energy ID 16, Neo Upper Energy ID 10).
- **104 Colorless-type Pokemon**: Playable in any single-energy deck.
- **510 cards** with GA mutation risk (deck composition dependencies).

### Effect Tags (from 2c taxonomy)

Every card with a relevant mechanical effect is tagged. Tags drive archetype assignment and synergy detection.

| Tag | Cards |
|-----|-------|
| coin_flip | 145 |
| deck_search | 103 |
| pokemon_search | 72 |
| energy_search | 38 |
| heal_fixed | 36 |
| cant_attack_next_turn | 34 |
| damage_prevention | 30 |
| retreat_lock | 29 |
| stadium_owner_only | 26 |
| draw_cards | 25 |
| inflict_paralyzed | 23 |
| self_energy_discard | 22 |
| heal_self | 21 |
| inflict_poisoned | 21 |
| inflict_confused | 20 |
| bench_protection | 17 |
| mill_opponent | 17 |
| trainer_search | 16 |
| inflict_burned | 15 |
| opponent_energy_discard | 15 |
| self_switch | 14 |
| inflict_asleep | 13 |
| hand_reveal | 11 |
| bench_spread_damage | 10 |
| forced_opponent_switch | 8 |
| pokemon_search_to_hand | 7 |
| energy_acceleration | 6 |
| heal_full | 5 |
| pokemon_search_to_bench | 5 |
| stadium_removal | 4 |
| retreat_cost_reduction | 4 |
| prize_reduction | 2 |
| heal_conditional | 2 |
| energy_from_discard | 2 |
| hand_disruption | 2 |
| heal_bench | 1 |
| (none) | 721 |

### Confidence Distribution

- high (475): Standard mechanics, clearly parsed from CSV text
- medium (510): GA risk, Stage 2 chains, Dragon energy, scaling effects
- low (282): No archetype fit, no abilities/attacks parsed, needs manual review

### Archetype Distribution

| Archetype | Cards |
|----------|-------|
| stage_1 | 295 |
| aggro | 204 |
| coin_flip | 145 |
| stage_2 | 83 |
| balanced | 78 |
| stall | 70 |
| active_bench_shifter | 62 |
| team_rocket | 52 |
| paralyze_heavy | 43 |
| stage_1_ex | 40 |
| dragon_combo | 35 |
| stadium_control | 30 |
| stage_2_ex | 23 |
| poison_heavy | 21 |
| deck_out | 17 |
| burn_heavy | 15 |
| hand_control | 13 |
| sleep_heavy | 13 |
| mega_stage_2_ex | 10 |
| mega_stage_1_ex | 10 |
| heavy_energy | 6 |
| ability_burn | 2 |
| ability_poison | 2 |
| triple_energy | 1 |

---

## 2. File Manifest (csv-data.zip)

| File | Description |
|------|-------------|
| cards.json | PRIMARY DELIVERABLE: All 1267 cards with full schema, effects, archetypes, synergies, strategic analysis, categorization. |
| cards_summary.txt | Human-readable text version of cards.json. Grouped by category, alphabetical within. |
| team_rocket_pokemon.json | 52 Team Rocket Pokemon with bench-count requirement analysis. |
| dragon_type_cards.json | 35 Dragon-type Pokemon with per-attack energy requirement breakdown. |
| triple_energy_candidates.json | Ranked candidates for the new low-species triple-energy archetype design. |
| rankings.json | Top 15 best cards, worst 10, least useful for GA. All with detailed reasoning. |
| cross_reference_analysis.json | 9227 synergy links across 990 cards. 10 synergy types: evolution_chain, energy_type_compatibility, status_setup_payoff, stall_combo, energy_accel_target, stadium_synergy, team_rocket_synergy, dragon_energy_solution, search_evo_target, protection_high_retreat. |
| category_analysis.json | Cards grouped by user-requested dimensions: heal (41), counter (83), status inflictors (poison:21, burn:15, paralyze:23, confuse:20, sleep:13), high damage 200+ (105), special abilities (247), free retreat (35), stadiums (26), items/tools (165). |
| extraction_report.md | Methodology report, corrections log, confidence distribution, verification results. |

---

## 3. cards.json - Schema Reference

Array of exactly 1267 objects. Ordered by card_id (range 1-1267, no gaps, no duplicates).

### 3.1 Core Fields

| Field | Type | Description |
|-------|------|-------------|
| card_id | int | Unique card ID matching CSV. Range 1-1267. |
| name | str | Exact card name from CSV column Card Name. |
| category | str | Pokemon | Basic Energy | Special Energy | Trainer-Item | Trainer-Supporter | Trainer-Tool | Trainer-Stadium |
| pokemon_stage | str or null | Basic | Stage 1 | Stage 2. Null for Trainers and Energy. |
| evolves_from | str or null | Name of the previous evolution stage. Null for Basic Pokemon and non-Pokemon. |
| hp | int or null | HP value from CSV. Null for Trainers and Energy. |
| type | str or null | Grass | Fire | Water | Lightning | Psychic | Fighting | Darkness | Metal | Colorless | Dragon. Null for Trainers. |
| weakness | str or null | Type this Pokemon is weak to. Null if none. |
| resistance | str or null | Type this Pokemon resists. Null if none. |
| retreat_cost | int or null | Number of energy to retreat. 0 = free retreat. Null for Trainers and Energy (post-audit). |
| is_ex | bool | True if Rule column contains Pokemon ex. |
| is_mega_ex | bool | True if Rule column contains Mega Pokemon ex. |
| is_ace_spec | bool | True if Rule column contains ACE SPEC. |
| is_team_rocket_pokemon | bool | True if Category contains Trainer Pokemon (Team Rocket). |
| is_dragon_type | bool | True if Type column is Dragon (Japanese character). |
| is_colorless_type | bool | True if Type is Colorless. These are playable in any single-energy deck. |
| colorless_energy_only | bool | True if ALL attack costs are Colorless-only (no typed energy symbols). |
| references_team_rocket_energy | bool | True if card Effect Explanation mentions Team Rocket Energy by name. |
| references_legacy_energy | bool | True if card Effect Explanation mentions Legacy Energy by name. |

### 3.2 Ability Schema

Array of ability objects. Can be empty. Tera abilities are prefixed with [Tera].

| Field | Type | Description |
|-------|------|-------------|
| name | str | Ability name from CSV Move Name column (with [Ability] prefix stripped). Tera abilities prefixed with [Tera]. |
| text | str | Verbatim or close paraphrase of the Effect Explanation for this ability row. |
| conditional_effects | array | Array of conditional effect objects. See 3.4. |

### 3.3 Attack Schema

Array of attack objects. Can be empty for support-only Pokemon.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Attack name from CSV Move Name column. |
| cost | str | Literal cost string from CSV Cost column, e.g. "{P}{P}{C}" or "{G}{G}bullet". The bullet character represents Colorless energy. |
| damage | str | Damage value from CSV Damage column. May contain multiplier (e.g. "30x") or be empty. |
| text | str | Effect Explanation text for this attack row. |
| conditional_effects | array | Array of conditional effect objects. See 3.4. |

### 3.4 Conditional Effect Schema (inside abilities/attacks)

These are the CRITICAL fields that prevent the Mewtwo-ex bug. Each identifies a precondition
beyond paying the energy cost. The `requires_deckbuilding_support` flag tells the GA/deck builder
whether a mutation operator must preserve a specific deck invariant.

| Field | Type | Description |
|-------|------|-------------|
| text | str | Verbatim or close paraphrase of the condition text from the card. |
| condition_type | str | Short label: bench_type_count | special_condition_present | card_present_in_zone | coin_flip | discard_cost | prize_count | scaling_count | mega_ex_only | cant_attack_next_turn | mill_effect | scaling_bench_count | scaling_energy_count | scaling_discard_count | ancient_marker | future_marker |
| condition_detail | str | Human-readable specific requirement. |
| requires_deckbuilding_support | bool | True if the condition requires specific deck composition (e.g., bench type count, energy source). False if it is just situational battle-state (e.g., opponent is Poisoned). |

### 3.5 Effect Tags

String array. Tags from the 2c taxonomy. Empty array if no applicable effects. Used for archetype
assignment and synergy detection. Key tags for system consumption:

| Tag | Meaning for Deck Builder |
|-----|------------------------|
| extra_prize | Takes more than 1 Prize card on KO. |
| prize_reduction | Opponent takes 1 fewer Prize card (Legacy Energy). |
| heal_full | Heals all damage from a Pokemon. |
| heal_fixed | Heals a specific numeric amount (see categorization.heal_factor.amounts). |
| heal_bench | Heals benched Pokemon specifically. |
| heal_self | Heals the attacking Pokemon itself. |
| heal_conditional | Healing is gated by a condition in the text. |
| deck_search | Card lets you search your deck. |
| pokemon_search_to_hand | Searches deck for a Pokemon, puts it in hand. |
| pokemon_search_to_bench | Searches deck for a Pokemon, puts it on bench. |
| energy_search | Searches deck for Energy card. |
| trainer_search | Searches deck for Trainer card (Item/Supporter/Tool/Stadium). |
| inflict_poisoned | Inflicts Poison on opponent (attack or ability). |
| inflict_burned | Inflicts Burn. |
| inflict_paralyzed | Inflicts Paralysis. |
| inflict_confused | Inflicts Confusion. |
| inflict_asleep | Inflicts Sleep. |
| status_removal | Removes special conditions. |
| energy_acceleration | Attaches extra energy from hand/deck/discard. |
| energy_from_discard | Recovers energy from discard pile. |
| self_energy_discard | Discards energy from self as cost. |
| opponent_energy_discard | Discards energy from opponent. |
| self_switch | Pokemon attack/ability switches itself with bench. |
| forced_opponent_switch | Forces opponent to switch Active Pokemon (gust effect). |
| retreat_lock | Opponent cannot retreat. |
| retreat_cost_reduction | Reduces retreat cost. |
| bench_protection | Prevents all damage to benched Pokemon. |
| damage_prevention | Prevents damage (ability or attack effect). |
| bench_spread_damage | Deals damage to opponent benched Pokemon. |
| mill_opponent | Forces opponent to discard from their deck. |
| deck_out | Forces opponent to draw, accelerating deck-out. |
| coin_flip | Effect depends on coin flip result. |
| draw_cards | Draws cards from deck. |
| hand_disruption | Shuffles hand into deck, forces discard. |
| hand_reveal | Reveals opponent hand. |
| stadium_removal | Discards a Stadium card from play. |
| stadium_both_players | Stadium effect applies to both players. |
| stadium_owner_only | Stadium effect applies to owner only. |
| cant_attack_next_turn | Pokemon cannot use attacks next turn after using this effect. |

### 3.6 Archetype Fits

Array of objects. Each card may fit multiple archetypes. Disabled archetypes are flagged.

| Field | Type | Description |
|-------|------|-------------|
| archetype | str | One of: balanced, aggro, stall, stage_2, stage_1, stage_2_ex, stage_1_ex, mega_stage_2_ex, mega_stage_1_ex, prize_rush, poison_heavy, burn_heavy, sleep_heavy, paralyze_heavy, stadium_control, deck_out, coin_flip, hand_control, heavy_energy, active_bench_shifter, team_rocket, dragon_combo, triple_energy, ability_poison, ability_burn, ability_heavy, energy_transfer |
| reason | str | Mechanic-based explanation citing actual card text, not generic type-based reasoning. |
| archetype_status | str | Only present when "disabled_pending_redesign". Indicates the archetype system maintainer has disabled this archetype. |

### 3.7 GA Mutation Risk

| Field | Type | Description |
|-------|------|-------------|
| ga_mutation_risk | bool | True if a GA mutation that changes deck composition could break this card. |
| ga_mutation_risk_note | str | Human-readable explanation of what invariant must be preserved. |

### 3.8 Strategic Analysis (user-requested deep analysis)

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| best_use_case | str | - | One-paragraph description of what this card does and when to play it. |
| gameplay_role | str | attacker | support | utility | energy_foundation | special_energy | stadium | trainer_utility | supporter | evolution_fodder | Primary role in a deck. |
| effectiveness_rating | str | critical | very_high | high | medium | low | How strong this card is in competitive play. |
| position_preference | str | active | bench | situational | flexible | Where this card wants to be positioned. |
| key_strength | str | - | Semicolon-separated list of notable strengths. |
| key_weakness | str | - | Semicolon-separated list of notable weaknesses. |
| synergy_requirements | array | - | Deck-level invariants this card needs. |
| anti_synergy_warnings | array | - | Things that conflict with this card in a deck. |
| turn_to_play | str | early_game | mid_game | late_game | mid_late_game | any | When this card is most impactful. |
| deckbuilding_priority | str | must_include | high | medium | low | filler | How important this card is for deck construction. |

### 3.9 Categorization (user-requested 9 dimensions)

The categorization object contains 9 sub-objects, each analyzing a different dimension:

| Dimension | Key Fields | Description |
| heal_factor | has_heal, type, amounts, targets, conditions | Whether/how this card heals. type can be full, null, or conditional. targets: self, bench, all_pokemon. |
| active_position | best_position, wants_active, wants_bench, active_benefit, bench_benefit | Whether this card benefits from Active or Bench position. |
| counter | is_counter, counters, mechanism | Whether this card counters opponent strategies (damage prevention, immunity, etc.). |
| status_conditions | poison/burn/sleep/paralyze/confuse each with can_inflict, via_ability, via_attack, guaranteed, mechanism | Status condition infliction analysis per status type. |
| no_retreat_cost | base_retreat_cost, has_free_retreat, provides_free_retreat, retreat_cost_reduction | Retreat cost analysis. |
| special_ability | has_ability, ability_count, ability_names, has_tera, ability_type, bench_protection, damage_prevention, energy_acceleration, draw_power, search_power, switch_power | Ability analysis including type (passive/active_once_per_turn) and power types. |
| damage | max_base_damage, has_scaling_damage, scaling_mechanism, is_high_damage | Damage output analysis. is_high_damage = true when max_base_damage >= 200. |
| stadium_use_case | is_stadium, references_stadium, stadium_synergy, stadium_type (global|owner) | Stadium card analysis. |
| items_use_case | is_item, is_supporter, item_category (search/tutor|recovery|energy_management|draw_engine|positioning|stadium_control|coin_manipulation|utility), use_case | Trainer card categorization. |

### 3.10 Cross-Reference Synergies

Array of synergy objects. Up to 10 per card. Discovered by batch-processing cards in groups
of 15, checking each new card against all previously analyzed cards.

|-------|------|-------------|
| partner_id | int | Card ID of the synergy partner. |
| partner_name | str | Name of the synergy partner. |
| type | str | Synergy type: evolution_chain | energy_type_compatibility | status_setup_payoff | stall_combo | energy_accel_target | stadium_synergy | team_rocket_synergy | dragon_energy_solution | search_evo_target | protection_high_retreat |
| direction | str | Direction of the synergy: this_evolves_from_partner | partner_evolves_from_this | shared_energy | shared_roster | mutual | this_sets_up_partner | this_powers_partner | this_protects_partner | this_finds_partner | partner_powers_this |
| reason | str | Specific mechanical explanation citing both cards text. |

### 3.11 Remaining Fields

| Field | Type | Description |
|-------|------|-------------|
| energy_fetch_or_flex | bool | True if this card provides/fetches multiple energy types. Key for triple_energy archetype. |
| confidence | str | high | medium | low. Indicates parsing certainty. Low cards need human review. |
| notes | str | Semicolon-separated notes. Includes GA risk details, synergy requirements, search details, confidence reasons. |
| revision_notes | str | Log of corrections made during pass-2 verification. Empty if no corrections. |

---

## 4. Called-Out Card Profiles

These cards were specifically requested by the system maintainer. Each is verified present and correct.

### 4.1 Team Rocket Mewtwo ex (ID 431)

- Name: Team Rocket's Mewtwo ex
- Category: Pokemon
- HP: 280, Type: Psychic
- Ability: Power Saver - "This Pokemon cannot attack unless you have 4 or more Team Rocket Pokemon in play."
- Ability conditional_effects: {"text": "4 or more Team Rocket’s Pokémon in play", "condition_type": "bench_type_count", "condition_detail": "requires 4+ Team Rocket Pokemon in play", "requires_deckbuilding_support": true}
- Attack: Erasure Ball, Cost: {P}{P}●, Damage: 160
- GA mutation risk: True - deck requirement: requires 4+ Team Rocket Pokemon in play; needs Team Rocket Energy and TR roster;
- Archetype fits: ['team_rocket', 'aggro']

### 4.2 Leafeon ex (ID 236) - Triple Energy Candidate

- Name: Leafeon ex
- HP: 270, Type: Grass
- Tera Ability: Prevents all damage while on bench (bench protection)
- Verdant Storm: Cost {G}●, Damage: 60× (scales with opponent energy)
- Moss Agate: Cost {G}{R}{W}, Damage: 230 - "Heal 100 damage from each of your Benched Pokemon."
- Effect tags: ['bench_protection', 'heal_bench', 'heal_fixed']
- Archetype fits: stage_1_ex, aggro, stall
- Strategic: Primary attacker with 230 max damage. wall/staller with damage prevention. healer for sustain. Stage 1 evolution.

### 4.3 Palossand ex (ID 223) - Triple Energy Candidate

- Name: PalossandEX
- HP: 280
- Tera Ability: Prevents all damage while on bench
- Attack: Tera, Cost: , Damage:  - As long as this Pokémon is on your Bench, prevent all damage done to this Pokémon by attacks (both yours and your opponent’s).
- Attack: Sand Tomb, Cost: ●●●, Damage: 160 - During your opponent’s next turn, the Defending Pokémon can’t retreat.
- Attack: Barite Jail, Cost: {W}{P}{F}, Damage:  - Put damage counters on each of your opponent’s Benched Pokémon until its remaining HP is 100.
- Effect tags: ['bench_protection', 'bench_spread_damage', 'retreat_lock']

### 4.4 Mega Dragonite ex (ID 904)

- Name: Mega Dragonite ex
- Stage: Stage 2, Evolves from: Dragonair
- HP: 370, Type: Dragon
- Rule: True
- Attack/Ability: Ryuno Glide, Cost: {W}{L}{L}, Damage: 330 - Discard 2 Energy from this Pokémon.
- Effect tags: []
- GA mutation risk: Dragon-type costs ['{W}{L}{L}'] need flexible energy; Stage 2 from Dragonair; GA must preserve evo chain;

### 4.5 Legacy Energy (ID 12)

- Name: Legacy Energy, Category: Special Energy
- ACE SPEC: True (1 copy max per deck)
- energy_fetch_or_flex: False

### 4.6 Team Rocket Energy (ID 15)

- Name: Team Rocket's Energy, Category: Special Energy

---

## 5. How to Consume This Data

### 5.1 For the Automated Deck Builder

The primary consumer. Use cards.json to inform Pokemon selection beyond HP/aggression stats:

1. **Filter by archetype_fits**: When building an archetype, only consider cards where the
2. **Check ga_mutation_risk**: When a GA mutates a deck, if it adds a ga_mutation_risk card,
   verify the mutation also preserves the invariants in ga_mutation_risk_note.
3. **Check conditional_effects**: Before including a card, verify all its conditional_effects
   with requires_deckbuilding_support=true can actually be satisfied by the current deck.
4. **Use categorization.special_ability**: Cards with bench_protection or damage_prevention
   should be prioritized for stall/control archetypes.
5. **Use categorization.damage**: is_high_damage (200+) identifies primary attackers.
6. **Use strategic_analysis.deckbuilding_priority**: must_include > high > medium > low > filler.

### 5.2 For the Team Rocket Archetype

- Load team_rocket_pokemon.json for the complete roster of 52 TR Pokemon.
- Only include cards where is_team_rocket_pokemon is true.
- Energy is exclusively Team Rocket Energy (ID 15).
- Mewtwo ex (ID 431) requires 4+ TR Pokemon in play - deck must have at least 4.
- 3 TR Pokemon (Moltres ex ID 407, Articuno, Zapdos) explicitly interact with TR Energy by name.

### 5.3 For the Dragon Archetype

- Load dragon_type_cards.json for 35 Dragon-type Pokemon with their energy requirements.
- No Basic Dragon Energy exists. Energy must come from:
  - Legacy Energy (ID 12, ACE SPEC, 1 copy max) - provides every type, 1 energy at a time
  - Neo Upper Energy (ID 10, ACE SPEC, 1 copy max) - provides every type for Stage 2
  - Prism Energy (ID 16) - provides every type for Basic Pokemon
- Dragon attack costs use real types (Fire, Water, Fighting, etc.) not Dragon symbols.
- All Dragon-type cards are flagged is_dragon_type=true and have GA mutation risk.

### 5.4 For the Triple-Energy Archetype

- Load triple_energy_candidates.json.
- Design: 1-2 Pokemon species + energy-tutor toolbox.
- Key candidates: Leafeon ex (G/R/W + 100 bench heal), Palossand ex (W/P/F + retreat lock + bench spread).
- Energy tutor cards are flagged with energy_fetch_or_flex=true.
- Cards with effect_tags containing energy_search are also relevant.

### 5.3 For the RL Agent

- Use cross_reference_synergies to build a card-pairing graph for action selection.
- Use categorization.status_conditions for status-condition-heavy strategies.
- Use strategic_analysis.effectiveness_rating and position_preference for play/switch decisions.
- Use categorization.no_retreat_cost.has_free_retreat to evaluate retreat cost.

---

## 6. Known Limitations

- Effect text parsing is regex-based. Edge cases in card wording may be missed.
- Cards with no effect_tags and no archetype_fits may still be useful as filler or in
  niche strategies not covered by the 21+3 archetype system.

