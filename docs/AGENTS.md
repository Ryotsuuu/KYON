# 🤖 KYON Sovereign Agent Catalog & Taxonomy Reference

```
█████╗  ██████╗ ███████╗███╗   ██╗████████╗███████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔════╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ███████╗
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝
```

---

## 📑 Table of Contents
1. [Taxonomy & Naming Conventions](#1-taxonomy--naming-conventions)
2. [Elemental Archetype Specifications](#2-elemental-archetype-specifications)
3. [The 519 Registered Legal Agent Catalog](#3-the-519-registered-legal-agent-catalog)
4. [Autonomous Master Meta-Agents](#4-autonomous-master-meta-agents)
5. [Genetic Algorithm Evolved Agents (`_ga`)](#5-genetic-algorithm-evolved-agents-_ga)
6. [Interactive Custom Deck Agents (`Custom_*`)](#6-interactive-custom-deck-agents-custom_)
7. [Decision Engine Architecture (MasterAgent v10.0)](#7-decision-engine-architecture-masteragent-v100)
8. [AlphaZero PUCT MCTS & HiveMind Network](#8-alphazero-puct-mcts--hivemind-network)

---

## 1. Taxonomy & Naming Conventions

All agents in the KYON Sovereign Platform adhere to a strict architectural identifier format:

```
[PREFIX]_[ENERGY_COMBINATION]_[ARCHETYPE]_[MODIFIER]
```

### Agent Origin Prefixes

| Prefix | Category | Description | Identifier Example |
| :--- | :--- | :--- | :--- |
| `S_` | Single Energy | Decks operating on one of 8 basic elemental energies. | `S_GRA_stage_2_ex`, `S_FIR_aggro` |
| `D_` | Dual Energy | Synergy decks combining 2 elemental energy pools ($C(8,2) = 28$ pairs). | `D_FIR+WAT_balanced`, `D_PSY+DAR_mega` |
| `T_` | Triple Energy | Complex synergy decks utilizing 3 elemental types (5 curated pools). | `T_GRA+FIR+WAT_balanced` |
| `TR_` | Team Rocket | Themed Psychic + Darkness disruption decks utilizing Rocket Pokémon. | `TR_balanced`, `TR_stage_2_ex` |
| `DR_` | Dragon | Multi-energy Dragon attacker decks with Energy Search acceleration. | `DR_balanced`, `DR_damage_counter` |
| `MASTER_` | Master Autonomous | Meta-agents synthesized from empirical winning patterns and weakness counters. | `MASTER_FIG_prize_rush`, `MASTER_WAT_stall` |
| `Custom_` | User-Created | Decks registered via `python ptcg.py deck create` with 60-card legality checks. | `Custom_Hero_Zygarde` |

### Elemental Energy 3-Letter Abbreviations
- `GRA` = Grass
- `FIR` = Fire
- `WAT` = Water
- `LIG` = Lightning
- `PSY` = Psychic
- `FIG` = Fighting
- `DAR` = Darkness
- `MET` = Metal

---

## 2. Elemental Archetype Specifications

The platform defines 12 core archetype strategies, each adhering to strict 60-card ratio templates:

| Archetype | Energy Ratio | Trainer Ratio | Pokémon Ratio | Core Strategic Behavior | Key Invariant Cards |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `balanced` | 22 | 20 | 18 | Midrange tempo, consistent evolution lines, stable energy curve. | *Professor's Research*, *Ultra Ball* |
| `aggro` | 20 | 22 | 18 | Turn 1–2 damage maximization, low energy requirements, heavy gusting. | *Prime Catcher*, *Boss's Orders* |
| `stall` | 25 | 17 | 18 | High HP tanks, status conditions, passive damage, deck-out pressure. | *Hero's Cape*, *Heavy Baton* |
| `stage_2` | 21 | 21 | 18 | Full Basic $\to$ Stage 1 $\to$ Stage 2 line with Rare Candy acceleration. | *Rare Candy (x4)*, *Stage 2 EX* |
| `stage_1` | 21 | 21 | 18 | Fast Stage 1 evolution pressure with heavy Basic search support. | *Evolution Incense*, *Nest Ball* |
| `stage_2_ex` | 21 | 21 | 18 | High-HP EX Stage 2 carry Pokémon backed by basic EX sub-attackers. | *Stage 2 EX (x4)*, *Rare Candy* |
| `stage_1_ex` | 21 | 21 | 18 | Fast 2-prize EX attackers designed to trade favorably early. | *Stage 1 EX*, *Energy Search* |
| `mega_stage_2_ex` | 22 | 20 | 18 | 300+ HP Mega Pokémon carries with massive energy attack costs. | *Mega EX (x4)*, *Energy Search Pro* |
| `mega_stage_1_ex` | 22 | 20 | 18 | Fast Mega EX beatdown utilizing single evolution stages. | *Mega EX (x4)*, *Switch* |
| `prize_rush` | 20 | 22 | 18 | Multi-prize extraction pace, prioritizing 2-prize knockouts. | *Prime Catcher*, *Boss's Orders* |
| `stadium_control` | 22 | 20 | 18 | Stadium board lock, reducing opponent damage and granting free retreat. | *Stadium Cards (x4)*, *Supporters* |
| `damage_counter` | 22 | 20 | 18 | Direct damage counter placement bypassing active defensive effects. | *Abilities*, *Damage Items* |

---

## 3. The 519 Registered Legal Agent Catalog

The platform catalog currently holds **519 fully verified, 100% tournament-legal decks** indexed in `ptcg-system/agents_registry.json` and `ptcg-system/agent_catalog.json`:

```
519 Total Agents = 96 Single + 336 Dual + 60 Triple + 11 Team Rocket + 12 Dragon + Master Agents + Custom Decks
```

- **Single Energy (96 Agents):** 8 elemental types $\times$ 12 archetypes.
- **Dual Energy (336 Agents):** 28 unique two-type combinations $\times$ 12 archetypes.
- **Triple Energy (60 Agents):** 5 curated three-type combinations $\times$ 12 archetypes.
- **Team Rocket (11 Agents):** Psychic + Darkness disruption decks with Rocket synergy.
- **Dragon (12 Agents):** Multi-energy Dragon decks with multi-type attack costs.
- **Autonomous Master & Custom Decks:** Meta champions and user-constructed legal decks.

---

## 4. Autonomous Master Meta-Agents

Master Agents (`MASTER_*`) are autonomously synthesized by the `AutonomousMasterSystem`:
1. **Dynamic Meta Analysis:** Reads match replay records from the `EnrichedReplayBuffer` to identify high-win-rate card synergies via Pointwise Mutual Information (PMI).
2. **Weakness Counter-Breeding:** Synthesizes counter-decks specifically targeting the active meta champion (e.g. Grass energy ramp to counter Fighting/Water champions).
3. **Continuous Survival Tournaments:** Weak master agents losing consistently are automatically pruned, while top performers are promoted to higher generation tiers.

---

## 5. Genetic Algorithm Evolved Agents (`_ga`)

Generated via `python ptcg.py upgrade --agent <ID> [--method {predefined,advanced}]`:
- **Chromosome:** 60-card integer array of card IDs.
- **Genetic Operators:** Uniform and single-point crossover, 1-to-3 card mutation, and elitism preservation.
- **Dual-Tier Optimization:**
  - **Method 1 — Predefined Legal GA** (`--method predefined`): Strict archetype constraints, 100% tournament legal bounds. Always available.
  - **Method 2 — ADVANCED System Optimization** (`--method advanced`): Master-guided fitness, discovery seeding, role-biased mutation. Unlocks when agent has >10,000 simulation games OR metagame has $\ge 3{,}000$ games per energy type across all 10 types. When locked, a Gatekeeper Suggestion Popup guides the user with recommended commands.
- **9-Step Chromosome Legality Repair Pipeline:**
  1. Copy Cap (max 4 per card, max 1 ACE SPEC, Basic Energy unlimited)
  2. Team Rocket Invariant (Mewtwo ex #431 requires 4+ TR Pokémon)
  3. Evolution Line Pruning via `evolves_from` chain tracing (supports Rare Candy Stage 2 skip)
  4. Max 6 Pokémon species / Max 18 Pokémon cards
  5. Basic Pokémon guarantee (at least 1 injected if absent)
  6. Stage 2 Rare Candy Accelerator (inject 2+ copies of #1079 for Stage 2 decks; purge for non-Stage 2)
  7. Special Energy HP-Type Scan (Grow Grass #18 → Grass; Telepath #19 → Psychic; Rock Fighting #20 → Fighting; TR Energy #15 → TR Pokémon; Neo Upper #10 → Stage 2)
  8. Attack Energy Harmonization (replace Pokémon whose attack costs require absent energy types; rainbow energies Legacy #12, Prism #16, Neo Upper #10 provide flexible coverage)
  9. 60-Card Enforcement (pad/trim to exactly 60)
- **HP Energy vs Attack Energy Distinction:**
  - HP Energy = Pokémon's type class (e.g. Dragon, Fire). Indicates card identity only.
  - Attack Energy = actual elemental costs in attack strings (e.g. `{G}{R}` for Applin #42). Deck MUST supply matching basic energies or rainbow energy.
- **Persistent Knowledge Store:** Advanced mode accumulates schemata, synergy pairs, counter strategies, and ecosystem metrics in `ptcg-system/ga_master_knowledge.json` across CLI sessions.
- **Registry Integration:** Saves evolved agents under the `_ga` suffix without overwriting the original base agent.

---

## 6. Interactive Custom Deck Agents (`Custom_*`)

Created via `python ptcg.py deck create`:
- Supports interactive terminal entry or direct `--cards "ID:Qty, ..."` input.
- Automatically determines elemental typing, archetype classification, and combo complexity.
- Directly usable across all simulation, tournament, and training commands (`simulate`, `matchups-simulation`, `train`, `list`).

---

## 7. Decision Engine Architecture (MasterAgent v10.0)

The `MasterAgent` in [`simulation/Decision_Engine.py`](file:///e:/PTCG%20Soverign%20Trainer/ptcg%20sovegin%20trainer/simulation/Decision_Engine.py) is the crash-proof decision brain powering competitive gameplay:

### Key Tactical Priorities
1. **Main Phase Decision Order:**
   $$\text{Ability Activation} \to \text{Supporters} \to \text{Stadium} \to \text{Trainer Items/Tools} \to \text{Evolve} \to \text{Attach Energy} \to \text{Retreat} \to \text{Attack} \to \text{END TURN}$$
2. **Energy Attachment Logic:**
   - Active Pokémon receives primary priority ($+1000$ weight) if within 1 attachment of lethal attack readiness.
   - High-HP bench anchors and EX carries receive secondary priority.
3. **Calculated Retreat Guard:**
   - Retreat is **strictly forbidden** if active HP $\ge 60\%$.
   - Allowed only when active HP $< 30\%$ and a bench Pokémon with 2+ energy is strike-ready.
4. **ACE SPEC & Tactical Item Hold:**
   - *Prime Catcher* is held until guaranteed lethal knockout or vulnerable high-value EX target is benched.
   - *Hero's Cape* and *Heavy Baton* are attached exclusively to primary stage-2 attackers.

---

## 8. AlphaZero PUCT MCTS & HiveMind Network

- **PUCT Upper Confidence Search:** Explores multi-branch decision trees balancing neural policy priors $P(s, a)$ and action-values $Q(s, a)$.
- **PyTorch GPU HiveMind:** Evaluates state tensors $\mathbf{x} \in \mathbb{R}^{256}$ in sub-millisecond forward passes on `cuda:0`.
- **Closed-Loop Learning:** Match trajectories are saved to `EnrichedReplayBuffer` and batched to GPU for real-time loss minimization.
