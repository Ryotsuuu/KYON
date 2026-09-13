# 📚 KYON Sovereign Pokémon TCG AI Platform — Complete CLI Reference Manual

```
██████╗ ████████╗ ██████╗ ██████╗      ███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗╚══██╔══╝██╔════╝██╔════╝      ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║
██████╔╝   ██║   ██║     ██║  ███╗     ███████╗██║   ██║██║   ██║█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗ ██║
██╔═══╝    ██║   ██║     ██║   ██║     ╚════██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚██╗██║
██║        ██║   ╚██████╗╚██████╔╝     ███████║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚████║
╚═╝        ╚═╝    ╚═════╝ ╚═════╝      ╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

---

## 📑 Table of Contents

1. [Overview & Execution Paradigms](#1-overview--execution-paradigms)
2. [Prerequisites & Compute Architecture](#2-prerequisites--compute-architecture)
3. [Quick Start Cheat Sheet](#3-quick-start-cheat-sheet)
4. [Master Command Reference](#4-master-command-reference)
   - [`setup` — Environment Verification & Init](#setup)
   - [`build` — Database Deck Compilation](#build)
   - [`list` — Catalog Filtering & Match Telemetry](#list)
   - [`deck` — Inspection, Rendering & Custom Creation](#deck)
   - [`simulate` — Parallel Statistical Benchmarks](#simulate)
   - [`evolve-simulation` — Closed-Loop Phased Evolution](#evolve-simulation)
   - [`matchups-simulation` — 14-Archetype Meta Gauntlet](#matchups-simulation)
   - [`audit` / `stress-test` — 98-Scenario Integrity Suite](#audit--stress-test)
   - [`sim run` — Step-by-Step Interactive Visualizer](#sim-run)
   - [`train` — External Dataset, MCTS & RL Training](#train)
   - [`master` — Autonomous Metagame Governance & Deep Inspection](#master)
   - [`upgrade` — Genetic Algorithm 60-Card Evolution](#upgrade)
   - [`export` — Kaggle Tarball Packaging](#export)
   - [`validate` — Master Deck Legality Arbiter](#validate)
   - [`gpu` — Accelerator & MemoryGuard Status](#gpu)
   - [`card` — Database Index Query](#card)
5. [Agent Identifier Naming Conventions](#5-agent-identifier-naming-conventions)
6. [System File Layout & Telemetry Stores](#6-system-file-layout--telemetry-stores)
7. [Exit Codes & Error Recovery](#7-exit-codes--error-recovery)

---

## 1. Overview & Execution Paradigms

The KYON Sovereign Pokémon TCG platform operates on two primary interfaces:

| Entry Point | Target Area | Primary Function | Command Invocation |
| :--- | :--- | :--- | :--- |
| [`ptcg.py`](file:///e:/PTCG%20Soverign%20Trainer/ptcg%20sovegin%20trainer/ptcg.py) | Execute | Unified master CLI for simulation, training, discovery, GA evolution, and visual diagnostics | `python ptcg.py <command> [options]` |
| [`main.py`](file:///e:/PTCG%20Soverign%20Trainer/ptcg%20sovegin%20trainer/main.py) | Kaggle Evaluator | Minimal-overhead production agent entry point compliant with Kaggle sub-second response times | Invoked autonomously by sandbox |

---

## 2. Prerequisites & Compute Architecture

- **Python Runtime:** Python 3.10, 3.11, or 3.12 (64-bit)
- **Native Game Engine:** C-Engine binary located in `cg/`:
  - Windows: `cg.dll`
  - Linux x86_64: `libcg.so`
  - Linux ARM64: `libcg-arm64.so`
  - macOS: `libcg.dylib`
- **PyTorch GPU Accelerator:** PyTorch with CUDA 11.8 / 12.1+ (NVIDIA RTX series; fallback to CPU vector operations when GPU is absent).
- **Host Memory:** Minimum 8 GB RAM (Platform enforces a **95% MemoryGuard** limit).

---

## 3. Quick Start Cheat Sheet

```bash
# 1. Inspect hardware locks and PyTorch GPU device status
python ptcg.py gpu

# 2. Run diagnostic health check across all subsystems
python ptcg.py setup check

# 3. Compile all elemental archetype decks from card database
python ptcg.py build

# 4. Create and validate a legal custom deck agent
python ptcg.py deck create --name Custom_Hero_Zygarde --cards "1056:4, 1119:4, 756:4, 1088:1, 1079:4, 1224:4, 1214:4, 1:35"

# 5. Simulate 100 parallel matches with full statistical distribution metrics
python ptcg.py simulate Custom_Hero_Zygarde S_WAT_stage_2_ex --games 100

# 6. Run Kaggle submission readiness gauntlet against 14 elemental archetype champions
python ptcg.py matchups-simulation --agent Custom_Hero_Zygarde --games 25

# 7. Ingest external match datasets into GPU HiveMind Network
python ptcg.py train dataset --path vis.json --epochs 3

# 8. Render step-by-step interactive battle and generate AlphaGo decision matrices
python ptcg.py sim run --agent1 Custom_Hero_Zygarde --agent2 S_GRA_stage_2_ex --mode step

# 9. Review autonomous Master agent rankings and live win rates
python ptcg.py master report

# 10. Execute the 98-scenario system integrity regression audit (it take ~75sec to finish) 
python ptcg.py audit
```

---

## 4. Master Command Reference

---

### 🔹 `setup`
**Purpose:** Initialize directory hierarchies, verify native FFI shared libraries, inspect CSV data files, validate `settings.json`, and benchmark hardware.

#### Syntax & Options
```bash
python ptcg.py setup [action]
```

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `action` (positional) | string | `check` | `check` (fast subsystem verification) or `init` (full directory and asset builder). |

#### Real Terminal Output Example
```
┌───────────────────────── PTCG SYSTEM HEALTH CHECK ──────────────────────────┐
│ C-Engine FFI Core:      [OK] (E:\PTCG Soverign Trainer\cg\cg.dll)           │
│ Card Database:          [OK] (892 Unique Cards Indexed)                     │
│ Agent Catalog:          [OK] (521 Built Agents: 97 Single, 337 Dual,        │
│                              60 Triple, 11 Rocket, 13 Dragon,               │
│                              2 Master/GA/Custom)                            │
│ GPU Accelerator:        [OK] (NVIDIA GeForce RTX 4050 Laptop GPU / cuda:0)  │
│ System Status:          [OK] ALL SYSTEMS FULLY OPERATIONAL                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

> 📌 **NOTE: Understanding the 521 Agent Catalog Count**  
 The system tracks 521 active registered agent decks. This represents the curated and compiled catalog consisting of 97 Single-Type decks, 337 Dual-Type decks, 60 Triple-Type decks, 11 Team Rocket decks, 13 Dragon decks, and autonomous Master/GA/Custom agents. > The card database contains unique cards allowing thousands of permutations result in more than 4k+ agents. 

---

### <a id="build"></a>🔹 `build`
**Purpose:** Compile and register archetype decks from `EN_Card_Data.csv` according to archetype rule definitions in `settings.json`.

#### Syntax & Options
```bash
python ptcg.py build [--combo <TYPE>] [--force]
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--combo` | string | `all` | Filter to build specific energy combination: `all`, `single`, `dual`, `triple`, `team_rocket`, `dragon`, `colorless`. |
| `--force` | flag | `False` | Force rebuild and overwrite of existing archetypes. Without `--force`, already compiled legal 60-card decks are safely skipped to preserve custom and hand-tuned modifications. |

> ⚠️ **IMPORTANT: Smart Rebuild Prevention & Immutability Invariant**  
> When executing `python ptcg.py build`, the engine checks each candidate against existing decks in `agents_registry.json`. If a deck already exists and has a legal 60-card list, it is skipped (`Skipped (Already Built)`). If all candidate decks already exist (`built_count == 0`), `agents_registry.json` is **never touched or written to**, ensuring hand-tuned decks, custom cards, and GA-evolved configurations are completely protected from accidental overwrites.
>
> **Colorless Energy & Decks (`--combo colorless`):**  
> In official Pokémon TCG rules, Colorless (`{C}`) is not a basic elemental energy card—it is satisfied by any elemental energy or Double Colorless Energy. The system indexes 108 Colorless cards as universal partner Pokémon across all archetypes, and allows building pure Colorless decks via `python ptcg.py build --combo colorless`.

---

### 🔹 `list`
**Purpose:** Search, filter, rank, and inspect registered system, custom, GA-evolved, and autonomous Master agents based on simulation volume, win rates, and tactical archetypes.

#### Syntax & Options
```bash
python ptcg.py list [action] [count] [--all] [--top <N>] [--sort <METRIC>] [--sim-type <TYPE>] [--min-games <N>] [--min-wr <FLOAT>] [--custom] [--ga] [--master] [--combo <C>] [--archetype <A>] [--search <Q>] [--limit <N>]
```

| Argument / Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `action` (positional) | string | `None` | Positional ranking/filter action: `top`, `worst`, `winrate`, `wr`, `sims`, `simulations`, `games`, `wins`, `custom`, `ga`, `master`, or specific simulation type (`batch`, `gauntlet`, `1v1`, `matrix`, `benchmark`). |
| `count` (positional) | int | `None` | Optional count when using `top` or `worst` (e.g. `python ptcg.py list worst 10`). |
| `--top` | int | `None` | Show top $N$ agents (e.g. `--top 10`, `--top 25`, `--top 50`). |
| `--worst` | flag / int | `None` | Show worst/underperforming agents sorted ascending by win rate (e.g. `python ptcg.py list --worst 10`). |
| `--order` | string | `desc` | Sort direction: `desc` (descending, e.g. highest win rate first) or `asc` (ascending, e.g. lowest win rate first). |
| `--sort`, `--by` | string | `Alphabetical` | Sort order: `winrate` (highest WR), `sims` (most games), `wins` (most wins), `name`. |
| `--sim-type` | string | `None` | Filter win rate and volume statistics to a specific simulation environment: `sim_run` (1v1 visual), `simulate` (batch), `matchups` (meta gauntlet), `matrix` (round-robin), `benchmark` (continuous ELO). |
| `--min-games`, `--min-sims` | int | `0` | Filter agents that have completed at least $N$ simulated matches. |
| `--min-wr` | float | `None` | Filter agents achieving at least win rate (e.g. `--min-wr 0.60` or `60` for $60\%+$ WR). |
| `--all` | flag | `False` | Display all registered agents without pagination truncation. |
| `--custom` | flag | `False` | Filter exclusively to user-created custom deck agents. |
| `--ga` | flag | `False` | Filter exclusively to Genetic Algorithm evolved agents (`_ga`). |
| `--master` | flag | `False` | Filter exclusively to autonomous Master meta-agents (`MASTER_*`). |
| `--combo` | string | `None` | Filter by combo type: `single`, `dual`, `triple`, `team_rocket`, `dragon`, `colorless`. |
| `--archetype` | string | `None` | Filter by archetype: `stage_2_ex`, `aggro`, `stall`, `prize_rush`, etc. |
| `--search` | string | `None` | Fuzzy keyword search across agent IDs, Pokémon names, or energy types. |
| `--limit` | int | `25` | Maximum number of rows to display in terminal view. |

#### Real Terminal Output Example (Standard View)
```
 PTCG Agent Catalog (519 Matching Agents - Showing Top 10 by Highest Win Rate) 
┌──────┬──────────────────────┬────────┬────────────┬──────────┬──────────┬───────┬──────────┬─────────────┐
│ Rank │ Agent ID             │ Origin │ Archetype  │ Energy   │ Matches  │ Total │ Win Rate │ Status      │
├──────┼──────────────────────┼────────┼────────────┼──────────┼──────────┼───────┼──────────┼─────────────┤
│  1   │ S_FIG_stage_2_ex     │ single │ stage_2_ex │ Fighting │ 4-1-0    │     5 │  80.0%   │ GRANDMASTER │
│  2   │ S_GRA_stage_2_ex     │ single │ stage_2_ex │ Grass    │ 4-1-0    │     5 │  80.0%   │ GRANDMASTER │
│  3   │ DR_damage_counter    │ dragon │ damage_c…  │ Dragon   │ 3-2-0    │     5 │  60.0%   │ META_CHAMP  │
│  4   │ S_FIG_prize_rush     │ single │ prize_rush │ Fighting │ 3-2-0    │     5 │  60.0%   │ META_CHAMP  │
│  5   │ S_FIR_aggro          │ single │ aggro      │ Fire     │ 3-2-0    │     5 │  60.0%   │ META_CHAMP  │
└──────┴──────────────────────┴────────┴────────────┴──────────┴──────────┴───────┴──────────┴─────────────┘
```

#### Real Terminal Output Example (Granular Simulation Breakdown View: `python ptcg.py list sims`)
```
 PTCG Agent Catalog (519 Matching Agents - Showing Top 10 by Simulation Volume) 
┌──────┬──────────────────────┬──────────┬──────────────┬──────────────┬──────────────┬─────────────┬──────────┬─────────────┐
│ Rank │ Agent ID             │ 1v1 (sim)│ Batch (simu) │ Gauntlet(mat)│ Matrix / ELO │ Total Games │ Over. WR │ Status      │
├──────┼──────────────────────┼──────────┼──────────────┼──────────────┼──────────────┼─────────────┼──────────┼─────────────┤
│  1   │ S_FIG_stage_2_ex     │ 10 (80%) │ 7000 (54%)   │ 500 (62%)    │ 500 (58%)    │        8010 │    53.4% │ BALANCED    │
│  2   │ TR_mewtwo_heavy      │ 2 (50%)  │ 5000 (45%)   │ -            │ -            │        5002 │    45.2% │ DEVELOPING  │
│  3   │ S_GRA_stage_2_ex     │ 2 (50%)  │ 3000 (49%)   │ -            │ -            │        3002 │    49.1% │ DEVELOPING  │
│  4   │ S_WAT_stage_2_ex     │ 1 (100%) │ 2000 (51%)   │ -            │ -            │        2001 │    50.6% │ BALANCED    │
│  5   │ DR_stage_2_ex        │ 1 (100%) │ 2000 (49%)   │ -            │ -            │        2001 │    48.6% │ DEVELOPING  │
└──────┴──────────────────────┴──────────┴──────────────┴──────────────┴──────────────┴─────────────┴──────────┴─────────────┘
```

---

### 🧠 Understanding Self-Play Dynamics & Evaluating Grandmaster Agents

#### 1. Why Heavy Self-Play Regresses Win Rate to ~50%
When an agent plays against **itself** (e.g. `python ptcg.py simulate --agent S_FIG_stage_2_ex --self-play --games 500`):
- Because both seats have access to identical deck compositions and tactical decision trees, the theoretical win rate naturally converges to **~50.0%** (governed solely by the slight Turn-1 vs Turn-2 seat tempo differential).
- **The Self-Play Win Rate Regression Warning:** An overall win rate near $50.0\%$ in `ptcg.py list` does NOT indicate a weak agent—it usually reflects heavy self-play training where the agent wins 50% and loses 50% against itself.
- **How to View True Competitive Ability:** To evaluate true strength without self-play dilution, filter by simulation type (e.g., `python ptcg.py list --sim-type matchups` or `python ptcg.py list gauntlet`) or run gauntlet simulations against external champions.
- **The True Value of Self-Play:** Self-play is not designed to inflate win rate percentages; it forces the decision engine and MCTS to eliminate exploitable weaknesses. When the AI plays against itself, any sub-optimal line (e.g. over-committing energy to a vulnerable bench Pokémon) is immediately punished by the opposing seat. Over thousands of self-play games, the system discovers **Nash Equilibrium tactical lines**.

#### 2. What to Look for in a Truly Elite Grandmaster Agent
To evaluate whether an agent is championship-ready:
1. **Cross-Archetype Matrix Win Rate ($\ge 60\%$):** The agent must achieve $> 60\%$ win rate across the entire 14-archetype tournament matrix (`python ptcg.py matchups-matrix`).
2. **Weakness Resilience ($\ge 40\%$ vs Counter-Type):** Even when facing its direct elemental weakness (e.g. Grass vs Fire, Water vs Lightning), a Grandmaster agent should maintain at least $40\%$ win rate through tactical bench shielding, Ace Spec timing, and Prize-trade denial.
3. **Low Setup Variance ($IQR \le 6$ Turns):** The agent should consistently establish its Stage 2 carry and begin lethal attacks between Turns 3–5.
4. **Dynamic ELO Rating ($\ge 1550$):** In continuous matchmaking benchmarks (`python ptcg.py benchmark auto`), the agent should consistently occupy Rank 1–3 on the leaderboard.

---

### 🔹 `deck`
**Purpose:** Inspect card distributions, render rich categorized deck tables, create custom decks with legality verification, or export decks to CSV.

#### Subcommands & Syntax
```bash
# 1. Inspect an agent's deck structure
python ptcg.py deck <AGENT_ID>

# 2. Interactive custom deck creation
python ptcg.py deck create

# 3. Direct programmatic deck creation via ID:Qty string
python ptcg.py deck create --name <AGENT_NAME> --cards "<ID:QTY, ID:QTY, ...>"

# 4. Render visual categorized deck view
python ptcg.py deck render --agent <AGENT_ID>

# 5. Export deck to CSV
python ptcg.py deck <AGENT_ID> --export my_deck.csv
```

| Flag / Param | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `agent` / `subcmd` | string | `None` | Positional agent ID or action keyword (`create`, `render`). |
| `--agent` | string | `None` | Explicit flag specifying agent ID. |
| `--name` | string | `None` | Agent identifier for new deck creation. |
| `--cards` | string | `None` | ID:Quantity string (e.g. `"1056:4, 1119:4, 756:4, 1088:1, 1079:4, 1224:4, 1214:4, 1:35"`). |
| `--export` | string | `None` | Output filepath for CSV export. |
| `--force`, `--overwrite` | flag | `False` | Explicitly permit overwriting an existing agent in `agents_registry.json`. |

> 🚨 **WARNING: Duplicate Agent Overwrite Protection**  
>To prevent accidental loss of hand-crafted or GA-evolved decks, `python ptcg.py deck create` strictly blocks registration if an agent with the requested name already exists. The command aborts with exit code 1 unless `--force` or `--overwrite` is explicitly specified.
>
> It is useful when an existing deck, such as a GA-evolved deck, Master Agent deck, or any agent deck has a minor card-related issue that needs to be corrected or updated. The specified command can be used to modify the deck by replacing, adding, or removing cards as required.

### 🔹 `simulate`
**Purpose:** Execute high-throughput, multi-threaded match simulations between two agents with comprehensive turn statistics, Wilson 95% Confidence Intervals, first/second seat tempo analytics, and automatic GPU replay ingestion.

#### Syntax & Options
```bash
python ptcg.py simulate <AGENT1> [vs] <AGENT2> [--games <N>]
```

| Flag / Param | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `agent1` (positional) | string | `None` | First competitor agent identifier. |
| `agent2` (positional) | string | `None` | Second competitor agent identifier. |
| `--games` | int | `100` | Total match iterations to simulate. |
| `--self-play` | string | `None` | Run self-play iterations for a single agent. |

#### Real Terminal Output Example
```
┌─ Simulation Result & Statistical Telemetry: Custom_Hero_Zygarde vs S_WAT_sta─────┐
│ Completed Games: 100 in 1.412s (70.82 games/sec) | 708.2 in 10s                  │
│                                                                                  │
│ Custom_Hero_Zygarde Wins:         58 (58.0%) | 95% Wilson CI: [0.482, 0.672]     │
│ S_WAT_stage_2_ex Wins:            42 (42.0%) | Draws: 0                          │
│ ──────────────────────────────────────────────────────────────────────────────── │
│ Turn Statistics by Agent:                                                        │
│   • Overall Match Pacing: Mean: 24.12 turns  |  Median: 22.0  |  IQR: 6 (σ: 4.8) │
│   • Custom_Hero_Zygarde Victory Speed: Mean 21.04 turns (Range: [16, 29])        │
│   • S_WAT_stage_2_ex Victory Speed: Mean 28.38 turns (Range: [18, 52])           │
│ ──────────────────────────────────────────────────────────────────────────────── │
│ Seat Advantage Analytics (Going 1st vs 2nd):                                     │
│   • Custom_Hero_Zygarde: 1st WR: 64.0% (32/50) | 2nd WR: 52.0% (26/50)           │
│   • Global 1st-Player Tempo Advantage: 58.0% Win Rate                            │
└──────────────────────────────────────────────────────────────────────────────────┘
```

> **Note**-
> After running a simulation in **1v1 mode**, you can inspect the complete agent-vs-agent battle and evaluate the system's **decision-making process, gameplay behavior, strategic choices, and overall performance**.
>
> In `visualizer.html`, open the generated `vis.json` file to view the **complete 1v1 gameplay**, including the battle progression and actions taken by both agents.
>
> For a deeper view of the system's backend activity and decision-making throughout the battle, open `battle_turn_data.html`. It provides detailed turn-by-turn analysis, including:
>
> 1. **Dual-Agent Search Depth & Possibility Volume Trajectory (Turn-by-Turn)**
> 2. **Real Turn-by-Turn Win-Rate Trajectory & Strategic Turning-Point Marks (Both Agents)**
> 3. **Turn-by-Turn Algorithmic Engine Allocation (MCTS, NN, MCTS×NN Hybrid & OODA Telemetry)**
> 4. **Strategic Cause-and-Effect Decision Possibility Explorer**
> 5. **Per-Turn Decision Tree & Optimal Path Arbitration (Both Agents)**
>
> These visualizations are automatically populated with **new data after every 1v1 simulation run**, allowing you to inspect the latest battle, decision-making behavior, algorithmic activity, and system performance throughout each simulation.

### 🔹 `evolve-simulation`
**Purpose:** Closed-loop phased simulation where batches of matches are played, telemetry is stored in the replay buffer, and the PyTorch GPU HiveMind policy-value network is optimized between phases.

#### Syntax & Options
```bash
python ptcg.py evolve-simulation [--self-play <AGENTS...>] [--games <N>] [--batch-size <B>] [--train-epochs <E>]
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--self-play` | string... | `None` | One or more agents to run through phased self-play loops. |
| `--games` | int | `40` | Total match iterations per agent. |
| `--batch-size` | int | `10` | Match count per training phase. |
| `--train-epochs` | int | `2` | PyTorch GPU training epochs executed after each phase batch. |

---

### 🔹 `matchups-simulation`
**Purpose:** Evaluate candidate agent(s) against the default 14 elemental archetype champions or custom competitor pools (top GA-evolved agents, top Master agents, custom agents), with real-time Wilson 95% CI calculation, automatic replay buffering, PyTorch GPU HiveMind optimization, and structured run persistence in `data/simulation_runs/`.

#### Syntax & Options
```bash
python ptcg.py matchups-simulation --agent <AGENT_ID> [--games <N>] [--train-epochs <E>] [--opponents <AGENTS...>] [--top-ga <N>] [--top-master <N>] [--top-custom <N>]
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--agent` | string | `S_GRA_stage_2_ex` | Candidate agent ID to evaluate across the gauntlet. |
| `--games` | int | `25` | Games played per matchup competitor. |
| `--train-epochs` | int | `2` | GPU NN optimization epochs executed after each matchup segment. |
| `--opponents` | string... | `None` | Custom competitor agent IDs to evaluate against instead of default 14 archetypes. |
| `--top-ga` | int | `None` | Automatically evaluate against the top $N$ highest win-rate GA-evolved agents. |
| `--top-master` | int | `None` | Automatically evaluate against the top $N$ highest win-rate autonomous Master agents. |
| `--top-custom` | int | `None` | Automatically evaluate against the top $N$ highest win-rate custom user agents. |

> 💡 **PRO TIP: Benchmarking Against Top GA & Custom Agents**  
> To pit your candidate agent against elite evolved agents instead of basic archetypes, run:

```bash
python ptcg.py matchups-simulation --agent MyCustomAgent --top-ga 5 --games 25
```

> Every matchup automatically archives structured JSON battle metrics to `data/simulation_runs/matchups_{agent}_{timestamp}_{games}g.json`.

#### Real Terminal Output Example
```
           Meta Gauntlet Telemetry Report: S_WAT_stage_2_ex (11.8s)            
┌───────────────────────┬────────────┬────────────┬───────┬────────────┬────────────┬─────────────┬──────────┐
│ Matchup Archetype (P2)│ P1 Wins    │ P2 Wins    │ Draws │ P1 WR      │ P2 WR      │ Wilson 95%  │ NN Loss  │
├───────────────────────┼────────────┼────────────┼───────┼────────────┼────────────┼─────────────┼──────────┤
│ Grass (Stage 2 ex)    │          1 │          4 │     0 │ 20.0%      │ 80.0%      │ [0.03, 0.62]│ 0.8142   │
│ Fire (Stage 2 ex)     │          3 │          2 │     0 │ 60.0%      │ 40.0%      │ [0.23, 0.88]│ 0.8120   │
│ Psychic (Stage 2 ex)  │          5 │          0 │     0 │ 100.0%     │ 0.0%       │ [0.56, 1.00]│ 0.8015   │
│ Water (Stage 2 ex)    │          4 │          1 │     0 │ 80.0%      │ 20.0%      │ [0.37, 0.96]│ 0.7981   │
├───────────────────────┼────────────┼────────────┼───────┼────────────┼────────────┼─────────────┼──────────┤
│ OVERALL META TOTAL    │         35 │         35 │     0 │ 50.0%      │ 50.0%      │             │          │
└───────────────────────┴────────────┴────────────┴───────┴────────────┴────────────┴─────────────┴──────────┘
┌────────────────── KAGGLE SUBMISSION READINESS EVALUATION ───────────────────┐
│ Agent: S_WAT_stage_2_ex                                                     │
│ Candidate Record: 35 Wins - 35 Losses (Overall Meta WR: 50.0%)              │
│ Status: NEEDS GA OPTIMIZATION BEFORE SUBMISSION                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 🔹 `matchups-matrix` (14x14 Cross-Archetype Tournament Matrix & Heatmap)
**Purpose:** Simulates an exhaustive $14 \times 14$ round-robin tournament across all major archetypes (Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Dual-Type, Team Rocket, Dragon), rendering terminal heatmaps and exporting an interactive HTML heatmap dashboard to `ptcg-system/matchups_matrix.html`.

#### Syntax & Options
```bash
python ptcg.py matchups-matrix [--games <N>]
# Alias:
python ptcg.py matrix --games 10
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--games` | int | `10` | Games simulated for each pairwise matchup combination ($14 \times 14 = 196$ matchup pairs). |

---

### 🔹 `benchmark` (Continuous Automated ELO Matchmaking Benchmark)
**Purpose:** Executes automated round-robin tournament matchmaking using True ELO calculation ($K=32.0$), continuously updating global ELO ratings in `agents_registry.json` and outputting the empirical leaderboard.

#### Syntax & Options
```bash
python ptcg.py benchmark auto [--rounds <N>] [--top <K>] [--games <G>]
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `action` (positional) | string | `auto` | Benchmark mode: `auto` (matchmaking loop), `report`, or `run`. |
| `--rounds` | int | `3` | Number of tournament passes across the candidate pool. |
| `--top` | int | `16` | Number of top registered agents to include in the benchmark pool. |
| `--games` | int | `5` | Games simulated per pairwise match series. |

---

### 🔹 `audit` / `stress-test`
**Purpose:** Execute an automated integrity and mathematical invariant regression audit covering decision logic, prize mapping, Bayesian forecasters, PyTorch GPU tensors, ML Random Forest MAE bounds, 521-deck tournament legality, and the comprehensive 1,457-check Grandmaster Protocol Examiner.

#### Syntax & Subcommands
```bash
# 1. Standard 98-Scenario Sovereign Audit Suite (100% Pass Required)
python ptcg.py audit

# 2. Grandmaster Protocol Examiner (1,457+ Scenario Audit across 1,267 Cards)
python ptcg.py audit protocol examiner
# or via flag:
python ptcg.py audit --protocol

# 3. Deep System Diagnostic Audit
python ptcg.py audit --deep
```

| Command / Flag | Scope | Checks | Description |
| :--- | :---: | :---: | :--- |
| `python ptcg.py audit` | Platform-Wide | 98 Tests | Audits 98 invariant scenarios including Prize Card Tracker, Bayesian Opponent Tracker, HiveMind Attention Net forward passes, MCTS priors, flexible decision latency, and T97 Unified Multi-Component Learning Pipeline. |
| `audit protocol examiner` | Card Catalog & Rules | 1,457 Tests | Exhaustive audit verifying all 1,267 cards in `csv-data/cards.json`, specific ability/attack conditions (TR Mewtwo ex ID 431 *Power Saver* $\ge 4$ TR, Kangaskhan ex, Spidops *Rocket Rush*), special cards (Tools, Stadiums, Special Energy, Dragon dual-energy), energy bench acceleration, and GA playing style utilities. |
| `--deep` | Deep Stack | 12 Subsystems | Deep diagnostic inspection of CUDA accelerators, memory guard thresholds, and C-engine FFI interfaces. |

---

### 🔹 `sim run`
**Purpose:** Execute an interactive, step-by-step or summary visual match between two agents, exporting replay state timelines to `vis.json`, `visualizer.html`, `battle_turn_data.html`, and archiving versioned simulation run data to `data/simulation_runs/`.

#### Syntax & Options
```bash
python ptcg.py sim run --agent1 <A1> --agent2 <A2> [--mode <M>] [--turns <T>] [--html]
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--agent1` | string | `S_FIR_balanced` | Player 1 agent ID. |
| `--agent2` | string | `S_WAT_balanced` | Player 2 agent ID. |
| `--mode` | string | `summary` | Operational mode: `step` (interactive prompt), `fast`, `summary`, or `html`. |
| `--turns` | int | `50` | Maximum battle turn limit. Matches play naturally until terminal victory, deck-out, or reaching turn limit without artificial step truncations. |
| `--html` | flag | `False` | Generate interactive AlphaGo decision matrix HTML dashboard in `battle_turn_data.html`. |

> 📌 **NOTE: Turn Management & Endless Loop Prevention**  
> Unlike raw step limits, `--turns` controls actual Pokémon TCG game turns (`current.turn`). Matches execute flexibly without premature cuts, while automated stalemate protection prevents endless passes. Every match exports an immutable structured replay to `data/simulation_runs/battle_{agent1}_vs_{agent2}_{timestamp}_{turns}t.json`.

---

### 🔹 `train`
**Purpose:** Train all learning subsystems simultaneously on simulation experience, train neural networks on external JSON datasets, run MCTS self-play rollouts, or execute Deep Reinforcement Learning using the native Pokémon TCG environment (`PTCGEnvironment` / `RLTrainer` in `agents.RL`).

#### Syntax & Options
```bash
# 1. Unified multi-component autonomous training (All Subsystems: NN + MCTS + CVM)
python ptcg.py train all [--epochs <N>]
# or via flag:
python ptcg.py train --all [--epochs <N>]

# 2. External dataset ingestion pipeline (supports single JSON or directory of thousands of simulation files)
python ptcg.py train dataset --path <FILE_OR_DIR> [--limit <N>] [--epochs <N>] [--all] [--reset]

# 3. MCTS AlphaZero self-play training
python ptcg.py train --mcts [--iterations <N>]

# 4. Deep Reinforcement Learning training
python ptcg.py train --agent <AGENT_ID> [--episodes <N>]
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `action` (positional) | string | `None` | `all`, `dataset`, `agent`, or `mcts`. |
| `--all` | flag | `False` | Train all learning subsystems simultaneously (PyTorch GPU NN + MCTS AlphaZero + ML CVM). |
| `--path` / `--dataset` | string | `None` | Path to single JSON file or directory containing hundreds/thousands of simulation JSON files. |
| `--limit` | int | `None` | Maximum number of simulation files to process from a folder dataset in this run. If omitted, trains on all remaining files. |
| `--reset` | flag | `False` | Disregard `.train_checkpoint.json` and start folder dataset ingestion from file index 0. |
| `--epochs` | int | `3` | Training epochs for dataset and network optimization. |
| `--agent` | string | `None` | Agent ID for Reinforcement Learning training. |
| `--episodes` | int | `100` | Total RL training episodes. |
| `--mcts` | flag | `False` | Trigger MCTS tree search self-play rollouts. |
| `--iterations` | int | `50` | MCTS simulation rollouts per decision node. |

> 📌 **NOTE: Directory Ingestion & Checkpointing**  
> When `--path` points to a directory (e.g. `python ptcg.py train dataset --path ./replays/ --limit 500`), the engine scans all `*.json` / `*.jsonl` files, displays total discovered files, resumes from `.train_checkpoint.json` if present, parses states and actions, trains the neural network, updates MCTS rollout experiences, and saves the new progress checkpoint. Combine with `--all` to update all learning subsystems simultaneously.
  
> Never train on other format data, the system only take replay_vault (training) data. It is the intended format to train on.

#### Real Terminal Output Example (`train all`)
```
[UNIFIED SUBSYSTEM TRAINING] Training across ALL components on 95 stored replays (1 epochs)...
┌───────── AUTONOMOUS MULTI-COMPONENT TRAINING & LEARNING TELEMETRY ──────────┐
│ PyTorch GPU HiveMind:  Loss: 0.2467 | Epochs: 1 | Batch: 32 (PyTorch GPU    │
│ (cuda:0))                                                                   │
│ MCTS & MCTS_NN (AlphaZero):  2,100 Rollouts Trained | Lookahead: 17 Ply |   │
│ PUCT Shift: 0.0300                                                          │
│ ML Card Value Model (CVM):  RandomForest Regressor | MAE: 0.0315 | Samples: │
│ 1,267                                                                       │
│ Experience Replay Vault:    95 Total Games | 138 States Ingested (4.496s)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 🔹 `master`
**Purpose:** Autonomous metagame control, leaderboards, single-agent deep forensic inspection, dual-agent head-to-head intelligence, counter-strategy synthesis, and causal discovery.

#### Syntax & Options
```bash
# 1. System Metagame Overview (No agents specified)
python ptcg.py master [report]

# 2. Single-Agent Deep Forensic Telemetry Inspection (1 agent specified)
python ptcg.py master <AGENT_ID>
# or explicitly:
python ptcg.py master report <AGENT_ID>
python ptcg.py master --target <AGENT_ID>

# 3. Dual-Agent Comparative & Head-to-Head Intelligence (2 agents specified)
python ptcg.py master <AGENT_1> <AGENT_2>
# or explicitly:
python ptcg.py master report <AGENT_1> <AGENT_2>

# 4. Strategy Discovery across 25,000+ platform battles
python ptcg.py master discover

# 5. Neural card-type learning telemetry & counter-strategy advice
python ptcg.py master learn

# 6. Evolutionary survival cycle (prune bottom 25%, GA-optimize survivors)
python ptcg.py master evolve

# 7. Develop targeted counter-agent to defeat a champion
python ptcg.py master develop --target <AGENT_ID> [--rounds <N>] [--games <G>]

# 8. Multi-round autonomous self-play tournament training
python ptcg.py master train [--rounds <N>] [--games <G>] [--epochs <E>]
```

| Invocation Mode | Description | CLI Example |
| :--- | :--- | :--- |
| `report` (Default) | System Executive Dashboard: 521 Catalog Composition, Elite Top 10 Champions, GA Upgrade Watchlist, Archetype Metagame Balance Table, and Autonomous Master Leaderboards. | `python ptcg.py master` |
| `<AGENT_ID>` | **Single-Agent Deep Forensic Inspection:** Full 60-card roster breakdown with CVM utility scores (RandomForest Regressor), simulation breakdown across all modes (1v1, batch, gauntlet, matrix, benchmark), opponent matchup history, neural card-type learning pacing (Basics, Evolutions, Finishers), and top pairwise card synergies (PMI lift). | `python ptcg.py master S_FIG_stage_2_ex` |
| `<AGENT1> <AGENT2>` | **Dual-Agent Head-to-Head Comparative Intelligence:** Direct head-to-head match history, win rates, Wilson 95% CI, side-by-side architectural comparison, elemental matchup dynamics ($2\times$ weakness interactions), comparative simulation performance across all modes, and decisive matchup swing moments. | `python ptcg.py master S_FIG_stage_2_ex S_WAT_balanced` |
| `discover` | Aggregate system-wide simulations (25,000+ battles across 521 agents), analyze winning games, display concrete pairwise card synergies with names and types, and compare Elite Champions vs Watchlist Candidates. | `python ptcg.py master discover` |
| `learn` | Inspect deep neural learning telemetry, card-type sequencing (Basics, Evolutions, Energy, Supporters, Items), top learned card synergies with WR deltas, and metagame counter strategy advice. | `python ptcg.py master learn` |
| `evolve` | Execute an evolutionary cycle: prune bottom 25%, GA-optimize survivors, and breed counter-decks. | `python ptcg.py master evolve` |
| `develop` | Develop and train an adversarial counter-agent against a target champion. Runs dynamically without premature time cuts until achieving a winning counter agent ($\ge 60\%$ WR) or completing requested rounds. | `python ptcg.py master develop --target S_FIG_stage_2_ex --rounds 5` |
| `train` | Run autonomous tournament loop, update agent ratings, and train the neural policy network. | `python ptcg.py master train --rounds 5 --games 4` |

> 💡 **PRO TIP: Executive Master Report & Telemetry Highlights**  
> 1. **System Ecosystem Overview:** Total platform simulations, total Wins-Losses-Draws across all 521 registered decks, and count of winning games evaluated.  
> 2. **Single-Agent Forensic Telemetry (`ptcg master <AGENT>`):** Renders every card in the deck with its ML CVM Utility Score (0.0 to 1.0), categorizes Pokémon by evolution stage, displays policy execution quality across early/mid/late game phases, and reports exact simulation volume across all 5 game modes.  
> 3. **Dual-Agent Comparative Intelligence (`ptcg master <A1> <A2>`):** Immediately flags 2x elemental weaknesses (e.g. Fighting vs Darkness, Water vs Fire), compares lead carry attackers (HP and Max Damage), contrasts deck composite CVM values, and provides strategic turning point guidance.  
> 4. **Concrete Card Synergies Table:** Renders exact Pokémon/Trainer names and elemental types with empirical Pointwise Mutual Information (PMI) and estimated win rate boost (e.g. `Cynthia's Gabite (#380) + Cynthia's Garchomp ex (#381) -> +22.6% WR`).  
> 5. **Card-Type Learning Pacing:** Analyzes how the AI learns to sequence Basics on early turns, Rare Candy transitions on mid-turns, and ACE SPEC / Boss gusts for game-ending turns.

---

### 🔹 `upgrade`
**Purpose:** Optimize an agent's 60-card deck composition across multiple generations using Genetic Algorithm chromosome crossover and mutation. Supports two optimization tiers.

#### Syntax & Options
```bash
python ptcg.py upgrade --agent <AGENT_ID> [--generations <G>] [--population <P>] [--method {predefined,advanced}]
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--agent` | string | `None` | Target agent ID to evolve. |
| `--generations` | int | `5` | GA evolutionary generations. |
| `--population` | int | `12` | Chromosome population size per generation. |
| `--method` | string | `predefined` | Optimization tier: `predefined` (strict bounds) or `advanced` (master-guided, gated). |

#### Dual-Tier Optimization Methods

| Method | Description | Unlock Condition |
| :--- | :--- | :--- |
| `predefined` | System Predefined Legal GA. Strict archetype constraints, 100% tournament legal bounds, zero out-of-scope card drift. Always available. | None (always available) |
| `advanced` | ADVANCED System Optimization. Master-guided fitness, discovery seeding, role-biased mutation. Uses persistent knowledge store. | Agent has >10,000 simulation games **OR** metagame has ≥3,000 games per energy type across all 10 types. |

When `--method advanced` is requested but the gatekeeper conditions are not met, an interactive **Suggestion Popup** is rendered showing progress bars, deficit numbers, and recommended CLI commands to unlock, then safely falls back to `--method predefined`.

#### 9-Step Chromosome Legality Repair Pipeline

Every GA generation applies this strict repair pipeline to every offspring chromosome:

1. **Copy Cap**: Max 4 copies per non-basic energy card, max 1 ACE SPEC. Basic Energy (IDs 1–8) has no copy limit.
2. **Team Rocket Invariant**: Card #431 (Mewtwo ex) requires 4+ TR Pokémon or is removed.
3. **Evolution Line Pruning**: Orphaned Stage 1/Stage 2 cards whose pre-evolution chain (`evolves_from`) is absent are replaced. Traces 2-level chains (Stage 2 → Stage 1 → Basic) for Rare Candy skip compatibility.
4. **Max 6 Species / Max 18 Pokémon**: Excess species are pruned (keeping highest-scoring lines). Total Pokémon count capped at 18.
5. **Basic Pokémon Guarantee**: At least 1 Basic Pokémon injected if absent.
6. **Stage 2 Rare Candy Accelerator**: If Stage 2 Pokémon exist, 2+ copies of Rare Candy (#1079) are injected. If no Stage 2 Pokémon exist, Rare Candy is purged.
7. **Special Energy HP-Type Scan**: Typed special energies (Grow Grass #18, Telepath Psychic #19, Rock Fighting #20, Team Rocket's #15, Neo Upper #10) are replaced with primary basic energy if the deck lacks matching Pokémon type.
8. **Attack Energy Harmonization**: Pokémon whose attack costs require elemental energies not supplied by the deck (and not covered by rainbow energies Legacy #12, Prism #16, Neo Upper #10) are replaced with compatible Pokémon.
9. **60-Card Enforcement**: Pad with primary basic energy or trim to exactly 60 cards.

### 🔹 `export`
**Purpose:** Package an agent into a Kaggle-compliant standalone submission archive (`submission.tar.gz`).

#### Syntax & Options
```bash
python ptcg.py export --agent <AGENT_ID> [--output <FILE.tar.gz>]
```

---

### 🔹 `validate`
**Purpose:** Verify 60-card tournament legality across **11 competition rules** for all registered decks or a single agent.

#### Syntax & Options
```bash
# Validate single agent
python ptcg.py validate <AGENT_ID>

# Validate entire agent registry
python ptcg.py validate
```

#### Validation Rules (11-Pass Legality Arbiter)

| Rule | Check | Severity |
| :--- | :--- | :--- |
| **Rule 1** | Exactly 60 cards | ERROR |
| **Rule 2** | Max 4 copies per card (Basic Energy exempt); Max 1 ACE SPEC | ERROR |
| **Rule 3** | All card IDs exist in the card database | ERROR |
| **Rule 4** | At least 1 Basic Pokémon | ERROR |
| **Rule 5** | Max 1 ACE SPEC card total | ERROR |
| **Rule 6** | Energy count 20–25 | WARNING |
| **Rule 7** | Max 18 Pokémon cards (max 6 distinct species/lines) | ERROR |
| **Rule 8** | Energy Search (#1119) at least 2 copies | WARNING |
| **Rule 9** | **Stage 2 Evolutionary Accelerator**: Decks with Stage 2 Pokémon MUST contain Rare Candy (#1079). Decks WITHOUT Stage 2 Pokémon must NOT contain Rare Candy. | ERROR |
| **Rule 10** | **Special Energy HP-Type Matching**: Typed special energy cards require matching Pokémon HP type in deck. Grow Grass (#18) → Grass Pokémon; Telepath Psychic (#19) → Psychic Pokémon; Rock Fighting (#20) → Fighting Pokémon; Team Rocket's Energy (#15) → Team Rocket Pokémon; Neo Upper (#10) → Stage 2 Pokémon. | ERROR |
| **Rule 11** | **Attack Energy Coverage**: Every Pokémon's elemental attack cost requirements (e.g. `{G}`, `{R}`, `{W}`) must be satisfiable by the deck's basic energies or flexible rainbow energy (Legacy #12, Prism #16, Neo Upper #10). HP type reflects card class/typing ONLY — not attack requirements. | ERROR |

> **Key Distinction — HP Energy vs Attack Energy:**
> - **HP Energy** (e.g. a Dragon-type Pokémon) reflects the Pokémon's class and type identity. It does NOT determine which basic energies the deck must provide.
> - **Attack Energy** (e.g. Applin #42 costs `{G}{R}`) specifies the exact elemental energies needed to power attacks. If a deck runs Fire & Water but a Pokémon requires Grass to attack, that Pokémon is illegal unless Legacy Energy (#12) or Prism Energy (#16) provides flexible coverage.

---

### 🔹 `gpu`
**Purpose:** Display physical VRAM, CUDA device binding, system RAM allocations, and MemoryGuard headroom.

#### Syntax
```bash
python ptcg.py gpu
```

---

### 🔹 `card`
**Purpose:** Query the card database by numerical identifier, evolution stage, elemental type, or name.

#### Syntax & Options
```bash
python ptcg.py card [CARD_ID] [--stage <S>] [--type <T>] [--ex] [--search <QUERY>]
```

---

## 5. Agent Identifier Naming Conventions

System agent identifiers follow strict architectural naming rules:

```
[PREFIX]_[ENERGY]_[ARCHETYPE]_[MODIFIER]
```

- **Prefixes:**
  - `S_`: Single Elemental Energy Deck (e.g. `S_GRA_stage_2_ex`)
  - `D_`: Dual Elemental Energy Deck (e.g. `D_FIR_WAT_balanced`)
  - `T_`: Triple Elemental Energy Deck (e.g. `T_FIR_WAT_GRA_balanced`)
  - `TR_`: Team Rocket Themed Synergy (e.g. `TR_balanced`)
  - `DR_`: Dragon Type Elemental Deck (e.g. `DR_damage_counter`)
  - `MASTER_`: Autonomous Master Meta-Agent (e.g. `MASTER_FIG_prize_rush`)
  - `Custom_`: User-created deck agent (e.g. `Custom_Hero_Zygarde`)
- **Energy Abbreviations:** `GRA` (Grass), `FIR` (Fire), `WAT` (Water), `LIG` (Lightning), `PSY` (Psychic), `FIG` (Fighting), `DAR` (Darkness), `MET` (Metal).
- **Evolution Modifiers:** `_ga` indicates a Genetic Algorithm optimized variant.

---

## 6. System File Layout & Telemetry Stores

| Path | Purpose | Key File Attributes |
| :--- | :--- | :--- |
| `data/EN_Card_Data.csv` | Card database | Attributes, HP, attacks, energy costs, retreat, categories |
| `ptcg-system/agents_registry.json` | Master registry | Full 60-card integer ID lists, W-L-D match counts, metadata |
| `ptcg-system/agent_catalog.json` | Fast lookup catalog | Lightweight summaries for fast CLI indexing and filtering |
| `ptcg-system/custom_agents/` | User deck directory | CSV deck files created via `ptcg deck create` |
| `data/replay_buffer.json` | Experience store | Multi-dimensional state trajectories and match outcomes |
| `battle_turn_data.html` | Visualizer dashboard | AlphaGo-grade turn metrics, dual-agent search volume, and cause-effect reasoning |
| `visualizer.html` | Playback viewer | Interactive board replay renderer |

---

## 7. Exit Codes & Error Recovery

| Exit Code | Semantic Meaning | Recommended Recovery Action |
| :---: | :--- | :--- |
| `0` | **Success** | Operation completed normally. |
| `1` | **Execution Error** | Check argument syntax, deck legality errors, or file path validity. |
| `130` | **SIGINT / Interrupted** | User aborted process via `Ctrl+C`. State is cleanly preserved. |
