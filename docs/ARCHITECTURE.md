# 🏛️ KYON Sovereign Platform — Comprehensive AI Architecture & Systems Manual

```
 █████╗ ██████╗  ██████╗██╗  ██╗██╗████████╗███████╗ ██████╗████████╗██╗   ██╗██████╗ ███████╗
██╔══██╗██╔══██╗██╔════╝██║  ██║██║╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗██╔════╝
███████║██████╔╝██║     ███████║██║   ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝█████╗  
██╔══██║██╔══██╗██║     ██╔══██║██║   ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗██╔══╝  
██║  ██║██║  ██║╚██████╗██║  ██║██║   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
```

---

## 📑 Table of Contents
1. [Executive Systems Architecture](#1-executive-systems-architecture)
2. [C-Engine FFI Integration Layer](#2-c-engine-ffi-integration-layer)
3. [256-Dimensional State Embedding Tensor](#3-256-dimensional-state-embedding-tensor)
4. [PyTorch GPU HiveMind Policy-Value Network](#4-pytorch-gpu-hivemind-policy-value-network)
5. [AlphaZero PUCT MCTS Decision Pipeline](#5-alphazero-puct-mcts-decision-pipeline)
6. [Bayesian Opponent Hand & Threat Forecaster](#6-bayesian-opponent-hand--threat-forecaster)
7. [Cause-and-Effect Invariant Rules (Decision Engine v10.0)](#7-cause-and-effect-invariant-rules-decision-engine-v100)
8. [Enriched Replay Buffer & Experience Store](#8-enriched-replay-buffer--experience-store)
9. [HardwareManager & MemoryGuard Subsystem](#9-hardwaremanager--memoryguard-subsystem)
10. [Comprehensive File & Directory Hierarchy](#10-comprehensive-file--directory-hierarchy)

---

## 1. Executive Systems Architecture

The KYON Sovereign Pokémon TCG platform is built upon a high-performance, multi-layered architecture designed to bridge low-level C simulation throughput with high-level deep reinforcement learning and Bayesian inference:

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                  Game Observation State                │
                                  │         (Public Board, Discard, Hand, Energy)          │
                                  └──────────────────────────┬─────────────────────────────┘
                                                             │
                                  ┌──────────────────────────▼─────────────────────────────┐
                                  │           256-D Numerical State Encoder                │
                                  └───────────┬────────────────────────────────┬───────────┘
                                              │                                │
                       ┌──────────────────────▼──────────┐         ┌───────────▼──────────────────────┐
                       │     Bayesian Opponent Tracker   │         │    PyTorch GPU HiveMind Net      │
                       │   • Hypergeometric Posterior    │         │   • Shared ResNet Trunk          │
                       │   • Discard & Reveal Memory     │         │   • Policy Head π(a|s)           │
                       │   • Boss / Iono / Prime Catcher │         │   • Value Head V(s) ∈ [-1, +1]   │
                       └──────────────────────┬──────────┘         └───────────┬──────────────────────┘
                                              │                                │
                                              └────────────────┬───────────────┘
                                                               │
                                  ┌────────────────────────────▼───────────────────────────┐
                                  │               PUCT MCTS & Decision Engine              │
                                  │   • AlphaZero Tree Search (PUCT Exploration)           │
                                  │   • 32 Hypothesis Invariant Rules & Safety Guards      │
                                  │   • ACE SPEC Tactical Hold (Prime Catcher / Hero's)    │
                                  └────────────────────────────┬───────────────────────────┘
                                                               │
                                  ┌────────────────────────────▼───────────────────────────┐
                                  │            C-Engine FFI Action Execution               │
                                  │     (80+ Games/Sec, Sub-Millisecond Decision)          │
                                  └────────────────────────────┬───────────────────────────┘
                                                               │
                                  ┌────────────────────────────▼───────────────────────────┐
                                  │        Enriched Replay Buffer & Telemetry MLOps        │
                                  │   • Real-Time GPU Mini-Batch Backpropagation           │
                                  │   • AlphaGo Decision HTML Dashboard Generation         │
                                  └────────────────────────────────────────────────────────┘
```

---

## 2. C-Engine FFI Integration Layer

To satisfy competitive latency budgets ($< 5.0-1000 \text{ ms}$ per decision), the simulation engine interfaces with compiled C libraries (`cg.dll` on Windows, `libcg.so` on Linux, `libcg.dylib` on macOS):
- **C-ABI Bindings (`cg/sim.py`):** Uses Python `ctypes` to execute C-level state mutations and rule validations directly in memory.
- **Thread-Isolated Contexts:** Each simulation worker maintains an independent C-Engine context pointer, eliminating GIL contention and enabling **80+ games per second** on standard multi-core hardware.
- **Zero-Copy Observation Marshalling:** Game state dictionaries are extracted via optimized C structures and converted into compact numeric vectors without intermediate JSON serialization.

---

## 3. 256-Dimensional State Embedding Tensor

Every discrete game observation is mapped into a normalized numerical tensor $\mathbf{x} \in \mathbb{R}^{256}$ before neural evaluation:

| Dimension Range | Feature Category | Description | Normalization / Encoding |
| :--- | :--- | :--- | :--- |
| **0 – 31** | **Active Pokémon State** | Current HP, Max HP, Energy count by type, Evolution stage, Retreat cost, Status conditions. | Normalized $[0, 1]$, One-hot status |
| **32 – 95** | **Bench Reserves (5 Slots)** | Per slot: Pokémon HP ratio, Stage tier, Energy count, Attack-readiness delta, EX/Mega flags. | 12 features per slot $\times 5 = 60$ features |
| **96 – 159** | **Hand Structure & Resources** | Counts of Basic Pokémon, Evolutions, Items, Supporters, Stadiums, ACE SPECs, and Energy cards. | Normalized counts / total hand size |
| **160 – 191** | **Discard & Depletion State** | Supporter discard counts, Energy discard counts, Key item depletion counters. | Normalized against 60-card deck limit |
| **192 – 223** | **Prize & Match Dynamics** | Player prizes remaining ($0..6$), Opponent prizes remaining ($0..6$), Prize margin, Turn number. | Linear scaling in $[0, 1]$ |
| **224 – 255** | **Bayesian Threat Vector** | Posterior probabilities of opponent holding *Boss's Orders*, *Iono*, *Prime Catcher*, energy cards. | Calibrated Bayesian beliefs $[0.0, 1.0]$ |

---

## 4. PyTorch GPU HiveMind Policy-Value Network

The `HiveMindPolicyValueNet` ([`agents/MCTS_NN/policy_value_net.py`](file:///e:/PTCG%20Soverign%20Trainer/ptcg%20sovegin%20trainer/agents/MCTS_NN/policy_value_net.py)) processes the 256-D state tensor through a dual-pathway 4-head attention architecture:

$$\begin{aligned}
\mathbf{t}_{\text{macro}} &= \text{ReLU}(\text{LayerNorm}(\mathbf{W}_{\text{macro}} \mathbf{x} + \mathbf{b}_{\text{macro}})) \in \mathbb{R}^{256} \\
\mathbf{t}_{\text{spatial}} &= \text{LayerNorm}(\mathbf{W}_{\text{proj}} [\mathbf{E}_{\text{card}}(\mathbf{id}_{\text{board}}) \,\|\, \mathbf{E}_{\text{pos}}(\mathbf{pos}_{\text{board}})]) \in \mathbb{R}^{256} \\
\mathbf{T} &= \text{MultiheadAttention}(\mathbf{Q}=\mathbf{T}_{\text{seq}}, \mathbf{K}=\mathbf{T}_{\text{seq}}, \mathbf{V}=\mathbf{T}_{\text{seq}}, \text{heads}=4) \\
\mathbf{h} &= \text{ReLU}(\text{LayerNorm}(\mathbf{W}_{\text{trunk}} \mathbf{T} + \mathbf{b}_{\text{trunk}})) \in \mathbb{R}^{128} \\
\mathbf{p} &= \text{Softmax}(\mathbf{W}_{\text{policy}} \mathbf{h} + \mathbf{b}_{\text{policy}}) \implies \pi(a \mid s) \in \mathbb{R}^{64} \\
v &= \tanh(\mathbf{W}_{\text{value}} \mathbf{h} + \mathbf{b}_{\text{value}}) \implies V(s) \in [-1.0, +1.0]
\end{aligned}$$

### Multi-Task Objective Function with TD($\lambda$) Returns
$$\mathcal{L}(\theta) = \frac{1}{2} (V_\theta(s) - z_{\text{TD}(\lambda)})^2 - \sum_a \hat{\pi}(a \mid s) \log \pi_\theta(a \mid s) + c \|\theta\|_2^2$$
where $z_{\text{TD}(\lambda)}$ is the multi-turn temporal return ($\gamma=0.99, \lambda=0.95$), $\hat{\pi}$ is the MCTS visit distribution, and $c = 10^{-4}$ is $L_2$ regularization.

---

## 5. AlphaZero PUCT MCTS Decision Pipeline

Search tree expansion uses Information-Set MCTS with determinization and a phase-adaptive PUCT selection rule:

$$a^* = \arg\max_a \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]$$

- **Phase-Adaptive Exploration Constant ($c_{\text{puct}}$):**
  - **Early Opening (Turns 1–2):** $c_{\text{puct}} = 1.8$ (broad opening setup search).
  - **Mid-Game Transition (Turns 3–5):** $c_{\text{puct}} = 1.5$ (balanced tactical search).
  - **Late Endgame (Turn 6+ or Prize Gap $\ge 3$):** $c_{\text{puct}} = 1.1$ (sharp prize exploitation).
- **Neural Leaf Evaluation:** Value estimates are computed directly via the HiveMind value head $V(s) \in [-1, 1]$, bypassing noisy random rollouts.
- **Dynamic Branch Pruning:** High-priority moves (lethal attacks, critical evolutions) receive preferential expansion, pruning dead-end branches.

---

## 6. Bayesian Opponent Hand & Threat Forecaster

PTCG is an imperfect-information game where the opponent's hand and 6 prize cards remain concealed. The Bayesian Tracker maintains a probability distribution over the unseen card pool $\mathcal{U}_t$:

$$P(c \in \text{Opponent Hand} \mid \mathcal{O}_t) = \frac{P(\mathcal{O}_t \mid c \in \text{Hand}) \cdot \frac{N_{\text{unseen}}(c)}{|\mathcal{U}_t|}}{\sum_{k} P(\mathcal{O}_t \mid k \in \text{Hand}) \cdot \frac{N_{\text{unseen}}(k)}{|\mathcal{U}_t|}}$$

### Threat Categories Tracked
1. **Gust / Knockout Threats (*Boss's Orders*, *Prime Catcher*):** Likelihood of opponent swapping active Pokémon to take a benched knockout.
2. **Hand Disruption (*Iono*, *Unfair Stamp*):** Likelihood of opponent resetting player's high-card hand.
3. **Energy Acceleration (*Energy Search*, *Basic Energy*):** Probability of opponent powering up an inactive benched carry.

---

## 7. Cause-and-Effect Invariant Rules (Decision Engine v10.0)

To guarantee crash-proof and mathematically sound gameplay, [`simulation/Decision_Engine.py`](file:///e:/PTCG%20Soverign%20Trainer/ptcg%20sovegin%20trainer/simulation/Decision_Engine.py) enforces 32 invariant rules:

| Invariant ID | Rule Category | Operational Guard |
| :--- | :--- | :--- |
| **Rule #1** | Deck Registration | Validates exactly 60 cards are loaded on Step 0. |
| **Rule #2–5** | Energy Attachment | Active gets primary priority; bench energy allocated strictly to ready attackers. |
| **Rule #6–10** | Tactical Retreat | Retreat strictly barred if HP $\ge 60\%$; permitted only when active $< 30\%$ and ready bench exists. |
| **Rule #11–18** | ACE SPEC Preservation | *Prime Catcher* held until lethal knockout; *Hero's Cape* attached only to Stage 2 anchors. |
| **Rule #19–26** | Supporter Sequencing | Ability activations executed before supporters; draw supporters played before items. |
| **Rule #27–32** | Discard Cost Payment | Disallows discarding energy, ACE SPECs, or key evolution cards for cost payment. |

---

## 8. Enriched Replay Buffer & Experience Store

The `EnrichedReplayBuffer` ([`agents/Learning_System/replay_buffer.py`](file:///e:/PTCG%20Soverign%20Trainer/ptcg%20sovegin%20trainer/agents/Learning_System/replay_buffer.py)) stores complete match trajectories:
- **Rolling Window Capacity:** 10,000 matches persisted in `data/replay_buffer.json`.
- **Telemetry Records:** Tracks match duration, turns, winning seat, deck compositions, macro turning points, and state vectors.
- **Card-Level Win Attribution:** Computes empirical win rates and Pointwise Mutual Information (PMI) synergy matrices across all cards.

---

## 9. HardwareManager & MemoryGuard Subsystem

The `HardwareManager` ([`agents/GPU_config.py`](file:///e:/PTCG%20Soverign%20Trainer/ptcg%20sovegin%20trainer/agents/GPU_config.py)) and `MemoryGuard` ([`agents/Resource_Management/memory_guard.py`](file:///e:/PTCG%20Soverign%20Trainer/ptcg%20sovegin%20trainer/agents/Resource_Management/memory_guard.py)) guarantee resource stability:
- **CUDA Device Locking:** Automatically detects and locks `cuda:0` (NVIDIA RTX 4050 / Ampere / Ada Lovelace architecture) while bypassing integrated graphics.
- **95% RAM Ceiling:** Monitors system memory usage; triggers proactive disk flushes and `gc.collect()` when RAM $\ge 88\%$, and yields execution when RAM $\ge 95\%$.

---

## 10. Comprehensive File & Directory Hierarchy

```
ptcg sovegin trainer/
├── ptcg.py                          # Master Unified CLI (all commands)
├── main.py                          # Kaggle Competition Submission Agent
├── deck.csv                         # Default 60-Card Submission Deck
├── visualizer.html                  # Interactive Battle Replay Viewer
├── battle_turn_data.html            # AlphaGo Decision Matrix HTML Dashboard
├── README.md                        # Master Architecture & User Guide
├── cg/                              # Native Simulation C-Engine Core
│   ├── sim.py                       # ctypes Python FFI Bindings
│   ├── api.py                       # SelectContext, OptionType, Observation
│   ├── game.py                      # Game State Data Classes
│   ├── cg.dll                       # Windows x86_64 Native Library
│   ├── libcg.so                     # Linux x86_64 Native Library
│   └── libcg.dylib                  # macOS Native Library
├── simulation/
│   └── Decision_Engine.py           # MasterAgent v10.0 (32 Invariant Rules)
├── agents/
│   ├── csv_data.py                  # CsvDataIndex (Card Attributes & Rankings)
│   ├── csv_deck_builder.py          # Archetype 60-Card Deck Builder
│   ├── deck_validator.py            # MasterDeckValidator (Tournament Legality)
│   ├── archetype_engine.py          # 12 Archetype Strategy Templates
│   ├── GPU_config.py                # HardwareManager & CUDA Auto-Locking
│   ├── NN/                          # PyTorch GPU HiveMind Policy-Value Net
│   ├── MCTS/                        # Monte Carlo Tree Search Engine
│   ├── MCTS_NN/                     # AlphaZero PUCT MCTS Engine
│   ├── RL/                          # OpenAI Gym Environment & RL Trainer
│   ├── Genetic_Algorithm/           # GA Deck Optimizer & Chromosome Mutator
│   ├── Learning_System/             # EnrichedReplayBuffer & PMI Synergy
│   ├── Master_Autonomous/           # System Master Autonomous Agent Engine
│   ├── Evaluation_System/           # Matchups Gauntlet & Audit Suite
│   └── Resource_Management/         # MemoryGuard & Parallel Simulation Runner
├── data/                            # Card Database & Replay Storage
│   ├── EN_Card_Data.csv             # Master Card Database (892 Cards)
│   ├── cards.json                   # Enriched Card JSON Data
│   └── replay_buffer.json           # Persistent Replay Experience Store
├── ptcg-system/                     # Runtime System State & Registries
│   ├── settings.json                # Build Configuration Flags
│   ├── agent_catalog.json           # Fast Agent Index & Summaries
│   ├── agents_registry.json         # Master 519-Agent Deck & Match Registry
│   └── custom_agents/               # User-Created Custom Deck CSVs
└── docs/                            # Comprehensive Documentation Manuals
    ├── CLI-REFERENCE.md             # Complete CLI Command Reference
    ├── AGENTS.md                    # Agent Catalog & Taxonomy Reference
    ├── ARCHITECTURE.md              # AI Systems & Pipeline Specifications
    ├── GPU_GUIDE.md                 # GPU Compute & CUDA Acceleration Manual
    ├── HARDWARE_AND_SIMULATION_GUIDE.md # Simulation & Throughput Architecture
    ├── KAGGLE_DEPLOY.md             # Kaggle Grandmaster Deployment Manual
    ├── MEMORY_AND_GPU_SPECIFICATION.md # RAM Safety & Compute Topology
    └── SETTINGS.md                  # Configuration Settings Reference
```
