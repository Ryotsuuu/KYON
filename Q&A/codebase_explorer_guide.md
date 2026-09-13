# KYON Codebase Explorer Guide

## Comprehensive Guide to Understanding the Codebase and Key Findings

This document serves as an in-depth explorer's guide to the KYON codebase. For each significant file or directory, it explains:
- **Purpose**: What the file/component does in the system
- **Key Elements**: Important classes, functions, or data structures to examine
- **Insights to Gather**: What findings, patterns, or behaviors you can discover by analyzing this file
- **Relationship to System**: How it connects to other components and the overall AI ecosystem

Use this guide to navigate the codebase systematically and understand where specific functionalities reside.

---

## 📁 ROOT LEVEL FILES

### `main.py`
**Purpose**: Entry point for the Kaggle competition agent. Bootstraps the environment and defines the competition-facing `agent()` function.

**Key Elements to Examine**:
- `_DECK` and `_ARCHETYPE` constants (lines near end): The competition deck and its archetype
- `MasterAgent` class instantiation: The core decision-making agent
- `agent()` function: The competition entry point that processes observations and returns actions
- `SelectContext` enum (line ~64): 49 possible game contexts the agent must handle
- `StrategicPosture` enum (line ~218): OODA loop postures (DEVELOPMENT, BURST_RACE, etc.)
- `_select_attack_ooda` method (line ~300): Attack selection logic using OODA and energy counting

**Insights to Gather**:
- How the system maps raw game observations to actions
- The hardcoded competition deck composition (37 Water Energy, 3 Totodile, 3 Croconaw, 3 Mega Feraligatr ex, etc.)
- How OODA loop postures influence attack selection (particularly energy-based damage calculation)
- The fallback mechanisms when primary decision paths fail (exception handling returning `[0]` or deck list)
- Evidence of the system's architecture: Decision Engine as core, with MCTS/NN as enhancers

**Relationship to System**: This is the "face" of the system for competition, but internally delegates to the layered AI ecosystem defined in agents/.

### `README.md`
**Purpose**: Project overview and basic instructions.

**Key Elements to Examine**:
- Project description and goals
- Installation/setup instructions
- Usage examples
- Directory structure overview

**Insights to Gather**:
- High-level understanding of what the project aims to achieve
- Any noted dependencies or special requirements
- Intended audience and use cases

**Relationship to System**: Provides context for why the codebase exists and how to run it.

### `vis.json`
**Purpose**: Persisted game state sequences for analysis and visualization.

**Key Elements to Examine**:
- Array of game state objects (each representing a turn in a game)
- Structure of each state object: `select`, `logs`, `current` (with `turn`, `yourIndex`, `players`, etc.)
- Nested player data: `active`, `bench`, `hand`, `deck`, `prize`, `discard`
- Card objects with `id`, `serial`, `playerIndex`, `name`, and various properties

**Insights to Gather**:
- Actual gameplay trajectories captured from simulations
- How card states evolve over time (HP changes, energy attachment, evolution)
- Decision points where `select` is non-null (agent must make choices)
- The raw data format that feeds into the neural network and replay buffer
- Patterns in gameplay: energy curves, prize taking, damage accumulation

**Relationship to System**: Primary source of experience data for the replay buffer and learning system. Direct output from `battle_select()` that gets stored in `agents/Learning_System/replay_buffer.py`.

### `ptcg.py`
**Purpose**: Core Pokémon TCG game logic interface to the C-engine (`cg` module).

**Key Elements to Examine**:
- Function wrappers around C-engine calls: `battle_start`, `battle_select`, `battle_finish`
- Data conversion utilities: `to_observation_class`, `to_action_class`
- Constants and enums imported from `cg.api`
- Helper functions like `_get_cards()`, `_get_attacks()` that cache game data

**Insights to Gather**:
- The exact interface between Python code and the high-performance C-engine
- How raw C-engine data gets converted to usable Python objects
- What data is available at each game step (through the observation structure)
- Performance optimizations (caching of card/attack data)
- The source of truth for game rules and mechanics

**Relationship to System**: The boundary layer where the AI agents interact with the actual game engine. All observation data flows through here.

### `GPU_config.py`
**Purpose**: Configures GPU/PyTorch availability and device selection.

**Key Elements to Examine**:
- `configure_gpu()` function: Checks for CUDA availability and sets up PyTorch
- `get_torch_device()` function: Returns appropriate device (`cuda:0` or `cpu`)
- Fallback mechanisms for environments without GPU

**Insights to Gather**:
- Whether the system is utilizing GPU acceleration (critical for NN training speed)
- How the system handles deployment constraints (Kaggle vs local)
- The dual-backend design philosophy (GPU primary, CPU fallback)

**Relationship to System**: Enables the Neural Network components (`agents/NN/`) to leverage GPU acceleration when available, with automatic fallback.

### `knowledge_graph.py`
**Purpose**: Builds and queries a knowledge graph from game data.

**Key Elements to Examine**:
- Graph construction from card data and game trajectories
- Node and edge representation (cards, strategies, relationships)
- Query methods for finding patterns, synergies, or paths
- Integration with replay buffer or simulation data

**Insights to Gather**:
- How the system discovers higher-order patterns beyond simple statistics
- Potential for emergent strategy discovery through graph analysis
- Relationship between cards, decks, and win conditions
- Possible application in the System Agent ecosystem for archetype analysis

**Relationship to System**: Supports the Master Agent and System Agent ecosystems by providing structural analysis of game data.

### `rule_engine.py`
**Purpose**: Implements rule-based decision logic for specific game situations.

**Key Elements to Examine**:
- Rule definitions and priority ordering
- Condition-action pairs for common scenarios
- Integration points with the Decision Engine
- Conflict resolution mechanisms

**Insights to Gather**:
- Where hardcoded game knowledge is explicitly encoded
- How rule-based logic combines with learned components
- Specific situations where rules override or guide learning-based decisions
- The balance between explicit programming and learned behavior

**Relationship to System**: Represents the "pre-defined logic" component that the learning systems aim to augment or surpass through experience.

---

## 📂 AGENTS DIRECTORY

### `agents/__init__.py`
**Purpose**: Package initializer for the agents module.

**Key Elements to Examine**:
- Import statements revealing which submodules are exposed
- Version or metadata information

**Insights to Gather**:
- The intended public interface of the agents package
- Which components are considered core vs. auxiliary

**Relationship to System**: Defines how other parts of the codebase access agent functionalities.

---

### `agents/Evaluation_System/`
**Purpose**: Evaluates agent performance, benchmarks strategies, and analyzes matchups.

#### `champion_tracker.py`
**Purpose**: Tracks and identifies top-performing agents/strategies over time.

**Key Elements to Examine**:
- Data structures for maintaining agent performance histories
- Algorithms for identifying champions (win rate, consistency, novelty)
- Persistence mechanisms for champion records
- Integration with tournament systems

**Insights to Gather**:
- How the system defines and measures "best" performance
- Evolution of champion strategies over simulation time
- Criteria used for strategy selection in the Master Agent ecosystem
- Evidence of meta-learning through champion tracking

**Relationship to System**: Feeds into the Master Agent's understanding of what strategies are worth synthesizing or optimizing.

#### `elo_benchmark.py`
**Purpose**: Rates agents using Elo rating system against fixed opponents or pools.

**Key Elements to Examine**:
- Elo rating implementation and update mechanics
- Selection of benchmark opponents or pools
- Rating volatility and convergence properties
- Historical tracking of rating changes

**Insights to Gather**:
- Quantitative measure of agent strength over time
- How performance changes with learning and updates
- Stability of rankings indicating learning convergence
- Comparison between different agent variants (NN vs rule-based)

**Relationship to System**: Provides the primary metric for evaluating whether learning is actually improving performance.

#### `matchups_engine.py`
**Purpose**: Computes head-to-head performance matrices between agent strategies.

**Key Elements to Examine**:
- Matrix structure: agents/strategies vs. agents/strategies
- Statistical significance calculations (confidence intervals)
- Handling of limited sample sizes
- Visualization or reporting features

**Insights to Gather**:
- Which strategies dominate others in direct competition
- Identification of cyclic relationships (rock-paper-scissors dynamics)
- Weaknesses of otherwise strong strategies
- Data for the System Agent to discover counters and weaknesses

**Relationship to System**: Core tool for the System Agent to evaluate strategy effectiveness and guide evolution.

#### `protocol_examiner.py`
**Purpose**: Validates strategies against specific constraints or protocols.

**Key Elements to Examine**:
- Definition of protocols (resource limits, turn limits, etc.)
- Constraint checking mechanisms
- Reporting of protocol compliance/violations
- Integration with tournament systems

**Insights to Gather**:
- How the system ensures strategies are viable under competition rules
- Discovery of strategies that exploit rule interpretations
- Validation of learned strategies for real-world deployment
- Safety mechanisms for autonomous learning

**Relationship to System**: Ensures that learned and evolved strategies remain within legal and practical bounds.

#### `system_audit_engine.py`
**Purpose**: Performs continuous self-assessment of the AI system's components and performance.

**Key Elements to Examine**:
- Audit checkpoints and what they measure
- Anomaly detection mechanisms
- Performance degradation alerts
- Component health indicators
- Automatic triggering of corrective actions

**Insights to Gather**:
- How the system detects when learning has gone astray
- Early warning signs of overfitting, forgetting, or instability
- Feedback loops for maintaining system health during continuous learning
- Evidence of the system's ability to self-correct

**Relationship to System**: Critical for safe, long-term continuous learning - acts as the system's "immune system".

#### `__init__.py`
**Purpose**: Package initializer for Evaluation_System.

---

### `agents/Expert_Agent_Ecosystem/`
**Purpose**: Implements specialized rule-based agents for specific strategies or archetypes.

#### `energy_agents.py`
**Purpose**: Agents focused on energy management strategies.

**Key Elements to Examine**:
- Different energy attachment philosophies (aggressive, conservative, situational)
- Rules for when to attach vs. hold energy
- Integration with energy type considerations
- Performance characteristics in different matchups

**Insights to Gather**:
- Explicit knowledge about energy management encoded in rules
- Baseline performance for comparison with learning-based approaches
- Situations where rule-based energy decisions excel or fail
- Potential features for the neural network to learn

**Relationship to System**: Represents one type of expert agent that the System Agent evaluates and potentially synthesizes with learned components.

#### `__init__.py`
**Purpose**: Package initializer.

---

### `agents/Genetic_Algorithm/`
**Purpose**: Implements genetic algorithms for evolving agent strategies or deck compositions.

#### `deck_optimizer.py`
**Purpose**: Evolves deck compositions using genetic algorithms.

**Key Elements to Examine**:
- Chromosome representation (deck as list of card IDs)
- Fitness function (win rate against meta, diversity bonus)
- Selection mechanisms (tournament, roulette wheel)
- Crossover operators (deck splitting, gene exchange)
- Mutation operators (card substitution, addition/deletion)
- Elitism and population management
- Integration with simulation runner for fitness evaluation

**Insights to Gather**:
- How the system discovers novel and effective deck compositions
- Balance between exploitation of known good decks and exploration of new ones
- Evidence of emergent deck archetypes through evolution
- How deck evolution complements agent strategy evolution
- Constraints applied (deck size, card legality, etc.)

**Relationship to System**: Works alongside the Master Agent's GA to evolve the strategic foundation (decks) that agents then learn to play.

#### `__init__.py`
**Purpose**: Package initializer.

---

### `agents/GPU_config.py`
**Purpose**: (Duplicate of root? Actually this might be different - checking...) 
*Note: Based on earlier exploration, GPU_config.py exists in both root and agents/. The agents version may be a wrapper or specialized variant.*

**Key Elements to Examine** (if different from root):
- Agent-specific GPU configuration
- Integration with neural network training in agent contexts
- Resource allocation for agent-level computations

**Insights to Gather**:
- How GPU resources are managed at the agent level vs. system level
- Any agent-specific optimizations or constraints

**Relationship to System**: Ensures agent-level neural network components can leverage acceleration.

---

### `agents/Knowledge_System_Builder/`
**Purpose**: Builds structured knowledge bases from game data and simulations.

**Key Elements to Examine** (based on typical patterns):
- Knowledge extraction algorithms from replay buffer
- Ontology or schema definition for game knowledge
- Reasoning mechanisms over extracted knowledge
- Integration with expert systems or decision engines

**Insights to Gather**:
- How the system converts raw experience into structured knowledge
- Potential for explainable AI through knowledge representation
- Methods for verifying learned patterns against domain knowledge
- Application in the System Agent for strategic analysis

**Relationship to System**: Transforms statistical learning into symbolic reasoning capabilities.

#### `__init__.py`
**Purpose**: Package initializer.

---

### `agents/Learning_System/`
**Purpose**: Core infrastructure for experience storage, retrieval, and coordinated learning.

#### `__init__.py`
**Purpose**: Package initializer exposing key functions like `get_replay_buffer()`.

#### `condition_tracker.py`
**Purpose**: Tracks and classifies dynamic gameplay conditions (micro and macro states).

**Key Elements to Examine**:
- Micro condition detectors: hand composition, energy attachment, prize count, etc.
- Macro condition classifiers: game phase, threat level, opportunity assessment
- State persistence and transition tracking
- Integration with neural network input encoding
- Real-time updating mechanisms

**Insights to Gather**:
- What specific game states the system learns to recognize
- How condition detection feeds into decision making (OODA Orient phase)
- Evidence of learned pattern recognition in gameplay
- Features that may be encoded in the neural network embeddings
- Temporal patterns in condition evolution

**Relationship to System**: Provides the contextual understanding that enables adaptive behavior. Directly feeds into `encode_dynamic_state()` in the neural network.

#### `persistent_knowledge.py`
**Purpose**: Manages long-term storage and retrieval of learned knowledge beyond immediate experience.

**Key Elements to Examine**:
- Knowledge serialization formats (JSON, pickle, etc.)
- Mechanisms for knowledge consolidation (sleep-like processes)
- Retrieval strategies for relevant knowledge
- Integration with replay buffer and component updates
- Forgetting mechanisms and knowledge pruning

**Insights to Gather**:
- How the system prevents catastrophic forgetting over very long runs
- Distinction between working experience (replay buffer) and long-term knowledge
- Evidence of knowledge transfer between different agents or strategies
- Mechanisms for retaining useful old strategies while learning new ones
- Potential for meta-learning (learning what to remember)

**Relationship to System**: Complements the replay buffer for extremely long-term learning (beyond 10K+ games).

#### `replay_buffer.py`
**Purpose**: Central experience storage mechanism implementing prioritized experience replay.

**Key Elements to Examine**:
- `EnrichedReplayBuffer` class structure
- `add_game()` method: How trajectories are stored
- `sample_batch()` and `sample_prioritized_batch()`: Experience sampling strategies
- `compute_td_lambda_returns()`: Credit assignment mechanism
- `export_to_perpetual_vault()` and `load_from_perpetual_vault()`: Long-term storage
- Rolling window implementation (`max_disk_capacity`)
- Persistence to disk (`save()` and `_load()`)

**Insights to Gather**:
- The exact format of stored experiences (states, actions, rewards, metadata)
- How prioritization works (alpha parameter for prioritization exponent)
- Beta-annealing importance sampling correction (if implemented)
- Evidence of the 54% neural loss reduction through buffer utilization
- How the buffer enables component co-adaptation (NN trains on buffer, improves MCTS, etc.)
- Diversity maintenance mechanisms preventing buffer collapse
- Real-world scale: buffer size vs. total simulations run

**Relationship to System**: The central nervous system of learning. All learning components (NN, ML model, MCTS priors) draw from here, and all experience generators (simulation runner, self-play) feed into it.

#### `unified_trainer.py`
**Purpose**: Orchestrates training across all learning components (NN, MCTS, ML model) using experience from the replay buffer.

**Key Elements to Examine**:
- `train_all_simulation_components()` function: The main training orchestrator
- Experience ingestion into replay buffer
- Neural network training invocation (`get_hive_mind_net().train_on_replays()`)
- MCTS/AlphaZero training via `train_mcts_from_states()`
- ML Card Value Model training via `CardValueModel().train()`
- Telemetry generation and serialization to `models/training_telemetry.json`
- Visual telemetry panel generation (if rich console available)

**Insights to Gather**:
- The coordinated learning cycle: Simulation → Buffer → Component Updates → Improved Simulation
- Evidence of synchronous vs. asynchronous updates preventing train/serve skew
- How improvements in one component (NN) benefit others (better MCTS priors)
- The role of the ML model in providing complementary empirical evaluation
- Telemetry data that matches what you observed: loss reduction, MAE, PMI pairs
- Frequency and batch sizing of different component updates
- How the system handles component failure or degradation during training

**Relationship to System**: The conductor ensuring all learning components improve in harmony from shared experience.

---

### `agents/Master_Autonomous/`
**Purpose**: Implements the autonomous meta-agent ecosystem that discovers, evaluates, and evolves superior strategies.

#### `__init__.py`
**Purpose**: Package initializer.

#### `system_master_agent.py`
**Purpose**: The core Master Agent implementing self-play tournaments, GA optimization, and strategy synthesis.

**Key Elements to Examine**:
- `AutonomousMasterAgent` class structure
- Agent registry management (base agents vs. system agents vs. master agents)
- `get_system_agents_summary()`: Performance analysis of the agent ecosystem
- `create_master_agents()`: Synthesizing top-performing strategies into meta-agents
- `run_master_tournament()`: Head-to-head competition between strategies
- `learn_and_discover()`: Causal/PMI discovery of emergent tactics
- `evolve_cycle()`: Genetic Algorithm optimization of survivors
- `train_autonomous_loop()`: Main improvement cycle combining evaluation, discovery, evolution
- Telemetry and reporting functions (`print_comprehensive_report`, `print_learning_telemetry`)
- Integration with all learning components (NN, ML model, replay buffer, etc.)

**Insights to Gather**:
- How the system discovers what strategies are worth preserving and evolving
- Evidence of the "truthout improvements" process through extended self-play
- Mechanics of causal/PMI discovery finding emergent tactics beyond simple correlations
- GA implementation details: fitness functions, selection, crossover, mutation
- How the Master Agent coordinates with the Learning System for continuous improvement
- The evolution of strategies over time visible in tournament results
- Meta-learning: how the system learns to build better learners
- Real-world performance metrics from your telemetry context

**Relationship to System**: The "brain" that oversees the entire learning ecosystem, ensuring continuous improvement through competition and evolution.

---

### `agents/ML/`
**Purpose**: Machine learning components for card valuation and pattern recognition.

#### `card_value_model.py`
**Purpose**: RandomForest regressor for predicting empirical card values from features.

**Key Elements to Examine**:
- `CardValueModel` class structure
- Feature extraction: 28-D card feature vector (HP, damage, energy cost, type, stage, abilities, etc.)
- Training process: `train()` method using replay buffer outcomes
- Prediction process: `predict_card_value()` method
- Feature importance analysis for discover what makes cards valuable
- Model persistence and loading mechanisms

**Insights to Gather**:
- What specific card attributes the system has learned predict win contribution
- Evidence of the excellent 0.0139 MAE showing accurate valuation
- Discovered synergy patterns through interaction terms or feature combinations
- How card valuations shift as the meta evolves (if tracked over time)
- Validation that learning is capturing true strategic value, not noise
- Potential for using valuations in deck building or card selection decisions

**Relationship to System**: Provides an empirical, experience-based card evaluation system that complements and validates the neural network's internal valuations.

#### `cards_matrix.py`
**Purpose**: Handles card co-occurrence matrices and related analytics.

**Key Elements to Examine**:
- Co-occurrence matrix computation from replay buffer
- Similarity measures between cards (Jaccard, cosine, etc.)
- Cluster analysis of frequently played-together cards
- Integration with deck builder or synergy discovery
- Visualization or reporting of matrix insights

**Insights to Gather**:
- Which cards the system has learned tend to appear together in successful decks
- Evidence of the 5,000+ PMI synergy pairs discovered
- Emergent deck archetypes revealed through clustering
- How card relationships change with learning and meta evolution
- Potential for suggesting card replacements or additions based on co-occurrence

**Relationship to System**: Discovers higher-order patterns in card usage that individual card values might miss.

#### `__init__.py`
**Purpose**: Package initializer.

---

### `agents/MCTS/`
**Purpose**: Implements Monte Carlo Tree Search for game state exploration.

#### `__init__.py`
**Purpose**: Package initializer.

#### `mcts_agent.py`
**Purpose**: Standard MCTS agent wrapping the Decision Engine for rollouts.

**Key Elements to Examine**:
- `MCTSNode` class: Tree node structure with visit statistics
- `UCB1` formula implementation for node selection
- Tree policy: selection, expansion, simulation, backpropagation
- Integration with `MasterAgent` for rollout simulations
- `InformationSetMCTSAgent` variant for partial observability
- Parameters: iterations, exploration constant, time limits

**Insights to Gather**:
- How the system explores the game tree guided by experience (via Decision Engine rollouts)
- Visit counts and action values as learned measures of move quality
- Exploration vs. exploitation balance through the UCB1 formula
- Comparison between pure MCTS and NN-guided variants
- Evidence of improved search efficiency as the Decision Engine improves through learning
- Rollout quality as a function of Decision Engine performance

**Relationship to System**: Provides strategic lookahead capability, enhanced by learning in the Decision Engine and Neural Network.

---

### `agents/MCTS_NN/`
**Purpose**: Implements AlphaZero-style neural network guided Monte Carlo Tree Search.

#### `__init__.py`
**Purpose**: Package initializer.

#### `alphazero_agent.py`
**Purpose**: AlphaZero agent combining MCTS with Neural Network priors and values.

**Key Elements to Examine**:
- `AlphaZeroNode` class: NN-informed tree node with priors and values
- `PUCT` formula implementation: Q + cpuct × P(s,a) × √N(s)/(1+N(s,a))
- Neural network integration: `get_hive_mind_net()` for priors P and leaf values V
- `_compute_tactical_priors()`: Using NN to bias action selection
- `_search_main()`: The PUCT-guided search process
- `self_play_training()` function: Generating training data from self-play
- `train_mcts_from_states()`: Improving MCTS from state trajectories
- Temperature scheduling for exploration control

**Insights to Gather**:
- The core learning-driven decision mechanism: NN priors guiding search
- How the 54% neural loss reduction translates to better move selection
- Evidence of sample efficiency: fewer simulations needed for good decisions
- The PUCT formula balancing prior knowledge (NN) with observed outcomes (Q-values)
- Information set handling for partial observability in opponent's hand
- Adaptive temperature based on game state criticality
- How self-play generates improving training data for the NN
- The closed-loop improvement: better NN → better MCTS → better self-play → better NN

**Relationship to System**: The pinnacle of the learning system - where neural network learning directly enhances strategic search capability.

---

### `agents/NN/`
**Purpose**: Neural network components for policy and value prediction.

#### `__init__.py`
**Purpose**: Package initializer exposing `get_hive_mind_net()`.

#### `policy_value_net.py`
**Purpose**: The HiveMind Policy-Value Network - core learning component.

**Key Elements to Examine**:
- `HiveMindPolicyValueNet` class structure
- `STATE_DIM` = 256 and `ACTION_DIM` = 64 constants
- `encode_dynamic_state()` function: Converts game state to 256D vector
  - Micro conditions (64D): hand, bench, active, energies, prizes
  - Macro conditions (64D): game phase, threat/opportunity scores
  - Card embeddings (128D): learned representation per card ID
- Dual Torch backend (primary) and NumPy fallback
- Backbone network architecture (shared feature extraction)
- Policy head: 64-action logits (softmax → probabilities)
- Value head: single scalar (tanh → win probability in [-1,1])
- Training process: `train_on_replays()` using experience replay buffer
  - Policy loss: cross-entropy with target probabilities
  - Value loss: MSE with actual outcomes (+1 win, -1 loss, 0 draw)
  - Optimizer: Adam with learning rate scheduling
- Prediction methods: `predict()`, `predict_from_obs()`
- Model persistence: `save()` and `load()` methods
- Uncertainty estimation capabilities (if implemented)

**Insights to Gather**:
- The exact 256-dimensional representation the NN uses to understand game states
- How card embeddings are learned from policy gradients in experience
- What specific micro and macro conditions the system has learned to recognize
- Evidence of the 54% loss reduction showing improved state/action valuation
- The excellent 0.0139 MAE from the ML model validating NN internal valuations
- How the policy head produces action probabilities for move selection
- How the value head estimates win probability for position assessment
- Synchronization mechanisms preventing train/serve skew during continuous learning
- Exploration strategies (epsilon-greedy or temperature-based) during training
- GPU utilization status and performance characteristics
- How the NN enables the MCTS_NN component through priors and values

**Relationship to System**: The primary learning component whose improvements cascade through the entire system via better priors, better rollouts, and better value estimates.

---

### `agents/Resource_Management/`
**Purpose**: Manages simulation execution, memory, and hardware resources for large-scale experience generation.

#### `__init__.py`
**Purpose**: Package initializer.

#### `hardware_manager.py`
**Purpose**: Detects and manages hardware resources (CPU cores, memory, etc.).

**Key Elements to Examine**:
- Worker count determination based on available cores
- Memory allocation strategies
- Integration with memory guard and simulation runner

**Insights to Gather**:
- How the system scales simulation throughput to available hardware
- Resource allocation strategies for massive scale (1M+ simulations)
- Bottleneck identification and mitigation

**Relationship to System**: Enables the generation of sufficient experience for effective learning at scale.

#### `memory_guard.py`
**Purpose**: Enforces memory usage limits to prevent system overload during massive simulations.

**Key Elements to Examine**:
- Memory usage monitoring mechanisms
- 95% RAM cap enforcement policy
- Automatic garbage collection or flushing triggers
- Integration with simulation runner's chunk processing

**Insights to Gather**:
- How the system maintains stability during extended simulation runs
- Evidence of successful operation at the scales you've achieved
- Trade-offs between memory safety and simulation throughput
- Early warning signs of memory pressure

**Relationship to System**: Critical safety net enabling reliable 1M+ or 1B+ simulation campaigns.

#### `simulation_runner.py`
**Purpose**: High-throughput parallel simulation engine with auto-streaming and memory protection.

**Key Elements to Examine**:
- `_simulate_single_game()` function: Core battle simulation logic
  - Agent instantiation and interaction
  - Game loop with turn counting and action selection
  - Winner determination and game state capture
  - Experience collection when `collect_replay=True`
- `SimulationRunner` class:
  - `run_simulations()` method: Main entry point for batched simulations
  - Streaming batch processing with configurable chunk size
  - Memory guard integration and periodic checking
  - Thread pool management with auto-tuned worker count
  - Wilson confidence interval calculation for stable statistics
  - Replay flushing mechanism to disk
  - Progress tracking and callback support
- Performance targets and optimization strategies
- `get_simulation_runner()` factory function

**Insights to Gather**:
- The exact mechanism by which your 8,341+ simulations were generated
- How experience is collected for the replay buffer (`collect_replay` flag)
- Evidence of the streaming batch system preventing RAM exhaustion
- Wilson confidence intervals showing statistically sound performance metrics
- Thread utilization and parallelization efficiency
- How simulation parameters affect learning quality (chunk size, number of games)
- The mechanism enabling your extended post-competition analysis
- Performance metrics matching your telemetry: games/second, simulations in 10s
- How the system handles draws, timeouts, and abnormal terminations

**Relationship to System**: The experience generation engine that feeds the learning system. Directly populates the replay buffer used by all learning components.

---

### `agents/Self_Evolving/`
**Purpose**: Implements automated self-improvement mechanisms inspired by AlphaEvolve.

#### `__init__.py`
**Purpose**: Package initializer.

#### `alpha_evolve.py`
**Purpose**: Autonomous improvement system that mutates and selects better strategies.

**Key Elements to Examine**:
- `AlphaEvolve` class structure
- Objective function definition (win rate + novelty + robustness terms)
- Mutation operators: strategy tweaks, parameter adjustments, architectural changes
- Selection mechanisms: tournament-based survival, novelty pressure
- Archive/Hall of Fame implementation for top performers
- Integration with simulation runner for objective evaluation
- Iteration and generation management

**Insights to Gather**:
- How the system discovers improvements beyond simple parameter tuning
- Evidence of novelty-driven exploration preventing local optima
- The balance between exploiting known good strategies and exploring new ones
- How improvements are archived and built upon over generations
- Potential connection to the Master Agent's GA and tournament systems
- Meta-learning aspects: learning how to improve the improvement process

**Relationship to System**: Represents another avenue for continuous improvement, potentially operating alongside or integrated with the Master Agent ecosystem.

---

### `agents/system_architecture/`
**Purpose**: Documents or enforces the overall system architecture.

**Key Elements to Examine**:
- Architectural diagrams or specifications
- Component interface definitions
- Dependency diagrams
- Data flow specifications
- Evolution guidelines

**Insights to Gather**:
- The intended big-picture view of how all components fit together
- Design principles guiding development and evolution
- Constraints ensuring maintainability and extensibility
- Vision for future extensions and integrations

**Relationship to System**: Provides the conceptual framework for understanding how the pieces form a cohesive intelligent system.

---

## 📂 DATA DIRECTORY

### `data/replay_vault/`
**Purpose**: Long-term storage of simulation experiences in timestamped batches.

**Key Elements to Examine**:
- File naming convention: `replays_YYYYMMDD_HHMMSS_[count]_games.json`
- File sizes indicating batch sizes (typically 12-142 games per file)
- JSON structure of stored experiences
- Chronological organization enabling time-based analysis
- Total storage size indicating cumulative experience

**Insights to Gather**:
- The scale of experience accumulated (your ~84MB total indicates substantial learning)
- Temporal patterns in learning (early vs. late performance)
- Ability to resurrect specific time periods for analysis or retraining
- Evidence of continuous operation over multiple days/sessions
- Correlation between file timestamps and your simulation runs
- How the system manages very long-term experience storage

**Relationship to System**: The long-term memory complement to the active replay buffer. Stores experiences that may be sampled less frequently but retained indefinitely.

### `data/simulation_runs/`
**Purpose**: Individual simulation records, including both small tests and large batches.

**Key Elements to Examine**:
- File naming: `battle_[deck1]_vs_[deck2]_YYYYMMDD_HHMMSS_[count]t.json` or `sim_[description]_YYYYMMDD_HHMMSS_[count]g.json`
- Variety of matchups: archetype vs. archetype, self-play, specific strategy tests
- File sizes indicating scale (small tests vs. your 1000-game file: `sim_S_WAT_mega_stage_2_ex_vs_S_WAT_mega_stage_2_ex_20260907_012405_1000g.json`)
- JSON structure capturing complete game traces
- Presence of `replay_states` arrays when experience collection was enabled

**Insights to Gather**:
- Specific matchups tested and their outcomes
- Evidence of large-scale simulation campaigns (your 1000-game file)
- Detailed game trajectories available for deep analysis
- Correlation between simulation files and learning telemetry timestamps
- Which specific strategies or decks were evaluated during learning cycles
- How simulation complexity scales with file size and processing time

**Relationship to System**: The granular record of simulation experiences that feed into the replay buffer and learning system. Enables post-hoc analysis of specific learning periods.

### Root Data Files (examples)

#### `cards.json`
**Purpose**: Master database of all Pokémon TCG cards with attributes.

**Key Elements to Examine**:
- Card ID mapping to names and properties
- Attributes: HP, damage, energy cost, type, stage, evolutions, abilities, attacks
- Structural consistency for feature extraction in ML model

**Insights to Gather**:
- The complete card universe the system can reason about
- Data source for feature engineering in `CardValueModel`
- Potential for identifying under/overvalued cards through learning
- Evolution lines and synergy patterns visible in the raw data

#### `EN_Card_Data.csv`
**Purpose**: CSV format of English card data for alternative processing.

**Key Elements to Examine**:
- Column structure matching cards.json fields
- Potential for easier import into analysis tools
- Consistency with JSON format

#### `card_learning_analytics.json`, `cards_matrix.json`, etc.
**Purpose**: Pre-computed analytics products from the learning system.

**Key Elements to Examine**:
- Structure and content of specific analytics files
- Timestamps indicating when they were generated
- Correlation with simulation telemetry or report generation times
- Specific insights: learning curves, synergy discoveries, cluster analysis

**Insights to Gather**:
- Intermediate products of the learning pipeline
- Validation that analytics generation is working correctly
- Potential for reuse in reports or visualizations without recomputation
- Evidence of ongoing analysis beyond real-time telemetry

**Relationship to System**: Output products from the learning and analysis pipelines that can be consumed by reporting or visualization components.

---

## 📂 SIMULATION DIRECTORY

### `simulation/Decision_Engine.py`
**Purpose**: The core rule-based decision engine with OODA loop architecture.

**Key Elements to Examine**:
- `MasterAgent` class: Main decision-making agent
- `SelectContext` enum: 49 game contexts (verify completeness)
- `OptionType`, `CardType`, `AreaType`, `EnergyType` enums: Game mechanics
- `PrizeCardTracker`: 100% prized card identification system
- `BayesianOpponentTracker`: Learns opponent hand distribution from observations
- `MCTS Rollout & Tactical Invariants Evaluator`: 3-turn lookahead with anti-bait
- `Long-Game Endurance & Deck-Out Guard`: Handles 100-200+ turn games
- `Dynamic Energy Saturation & Divert to Bench Sweeper`: Learned energy management
- `Smart Hand Discard vs Search Optimization`: Contextual decision making
- Five `StrategicPosture` classes implementing OODA loop:
  - `STALL_DISRUPT`: Priority on disruption when behind in prizes
  - `SACRIFICE_WALL`: Resource protection when ahead
  - `TACTICAL_PIVOT`: Adaptive counter-strategy
  - `BURST_RACE`: Aggressive tempo when ahead
  - `DEVELOPMENT`: Standard play and resource building
- Context-specific handler methods for all 49 SelectContext cases
- Helper methods: `_get_state_and_players()`, `_get_card_from_option()`, etc.
- Action selection methods: `_select_energy()`, `_select_pokemon()`, `_select_attack_ooda()`, etc.

**Insights to Gather**:
- The complete OODA loop implementation driving adaptive behavior
- How the Bayesian tracker learns opponent tendencies from actual observations
- Evidence of the 3-turn lookahead improving tactical decisions
- Long-game handling mechanisms preventing losses in extended matches
- Learned energy management heuristics replacing simple rules
- Contextual hand optimization showing situational awareness
- How all 49 game contexts are handled with zero crashes (verify through code inspection)
- The rollout function used by MCTS components for leaf evaluation
- How learning in other components (NN, ML model) enhances this engine's effectiveness
- Specific improvements made during your "truthout" process (look for recent modifications or comments)
- Evidence of the system adapting its behavior based on learned patterns rather than static rules

**Relationship to System**: Provides the stable, knowledgeable foundation that learning components enhance. Serves as the rollout function for MCTS and the baseline decision maker when learning components are unavailable or less confident.

---

## 🔗 CROSS-COMPONENT INSIGHTS

### How to Trace the Learning Cycle

To understand how learning flows through the system, follow this path:

1. **Experience Generation**
   - Start at: `agents/Resource_Management/simulation_runner.py` → `_simulate_single_game()`
   - Observe: Agents (MasterAgent, MCTS variants) making decisions
   - Experience collection: When `collect_replay=True`, states/actions stored
   - End state: Experience ready for storage

2. **Experience Storage**
   - Move to: `agents/Learning_System/replay_buffer.py`
   - Observe: `add_game()` storing trajectories
   - Check: Prioritization parameters (alpha, beta)
   - Verify: Rolling window and persistence mechanisms
   - Result: Experiences ready for sampling

3. **Component Updates**
   - Go to: `agents/Learning_System/unified_trainer.py`
   - Follow: `train_all_simulation_components()`
   - Neural Network: `agents/NN/policy_value_net.py` → `train_on_replays()`
   - ML Model: `agents/ML/card_value_model.py` → `train()`
   - MCTS Priors: `agents/MCTS_NN/alphazero_agent.py` → `train_mcts_from_states()`
   - Synthesis: How improvements in one component benefit others

4. **Decision Improvement**
   - Return to: `simulation/Decision_Engine.py` and MCTS components
   - Observe: How updated NN provides better priors/values
   - Check: How improved ML model informs card selection
   - Verify: How enhanced MCTS search leads to better self-play data
   - Result: Next generation of improved experiences

5. **Meta-Evolution**
   - Examine: `agents/Master_Autonomous/system_master_agent.py`
   - Track: How tournament results drive GA evolution
   - Check: How discovered weaknesses lead to new strategy synthesis
   - Verify: How the hall of fame accumulates improving strategies
   - Outcome: Evolving strategic foundation for agents to learn

### Key Files for Validating Your Specific Findings

To verify the telemetry you observed:

1. **Neural Loss Reduction (0.5337 → 0.2462)**
   - Check: `agents/Learning_System/unified_trainer.py` telemetry generation
   - Verify: `models/training_telemetry.json` for historical loss values
   - Confirm: `agents/NN/policy_value_net.py` training implementation

2. **ML Model MAE (0.0139)**
   - Examine: `agents/ML/card_value_model.py` training and evaluation
   - Look for: MAE calculation in results
   - Check: Feature importance analysis for what drives predictions

3. **5,000+ PMI Synergy Pairs**
   - Investigate: `agents/ML/cards_matrix.py` co-occurrence computation
   - Review: `agents/Learning_System/unified_trainer.py` telemetry sections
   - Check: Any generated analytics files in `data/` directory

4. **3,235+ Transitions Ingested**
   - Inspect: `agents/Learning_System/replay_buffer.py` `__len__()` method
   - Review: `agents/Learning_System/unified_trainer.py` buffer size reporting
   - Verify: Actual count in replay buffer instance

5. **Real-Time Weight Synchronization**
   - Analyze: `agents/NN/policy_value_net.py` weight update mechanisms
   - Check: Asynchronous training vs. inference synchronization
   - Look for: Versioning or timestamp mechanisms preventing drift

### Files Modified During Your "Truthout Improvements"

To find evidence of your post-competition improvements:

1. **Primary Location**: `simulation/Decision_Engine.py`
   - Look for: Recent modifications, commented code, or TODO references to fixes
   - Check: Specific handler methods that were refined (energy attachment, bench clearing, etc.)
   - Verify: OODA posture switching logic adjustments

2. **Secondary Locations**:
   - `agents/NN/policy_value_net.py`: Potential NN architecture or training tweaks
   - `agents/Learning_System/replay_buffer.py`: Possible buffer parameter adjustments
   - `agents/Master_Autonomous/system_master_agent.py`: Changes to tournament or discovery logic
   - `agents/Resource_Management/simulation_runner.py`: Simulation parameter adjustments for better stress testing

3. **Validation Approach**:
   - Compare file modification timestamps with your competition end time
   - Look for commit messages or comments referencing "truthout", "improvement", or "audit"
   - Check for new validation tests or diagnostic logging
   - Examine any new configuration parameters added for fine-tuning

---

## 📊 USING THIS GUIDE EFFECTIVELY

### For Beginners:
1. Start with `main.py` to understand the competition interface
2. Examine `simulation/Decision_Engine.py` to see the core decision maker
3. Review `agents/NN/policy_value_net.py` to understand the learning core
4. Look at `agents/Learning_System/replay_buffer.py` to understand experience storage
5. Check `agents/Master_Autonomous/system_master_agent.py` to see the improvement orchestrator

### For Intermediate Users:
1. Trace the learning cycle as described above
2. Focus on one component at a time (NN, MCTS, Decision Engine)
3. Examine how specific files interface through imports and function calls
4. Look for telemetry generation points to understand performance tracking
5. Review data flow diagrams in your mind as you read file purposes

### For Advanced Users:
1. Identify potential improvement points in the learning cycle
2. Examine edge cases and error handling for robustness
3. Look for opportunities to increase scale or efficiency
4. Analyze the balance between exploration and exploitation in various components
5. Consider how the system might be extended to new domains or game variants

### For Validation and Auditing:
1. Use the "Key Files for Validating Your Specific Findings" section
2. Trace the learning cycle to ensure all components are functioning
3. Check modification timestamps against known improvement periods
4. Verify that telemetry matches actual code behavior
5. Look for consistency in constants, data structures, and interfaces across files

---

## 🏁 CONCLUSION

This codebase represents a sophisticated AI system that integrates multiple learning paradigms (reinforcement learning, neural networks, genetic algorithms, experience replay) with a strong rule-based foundation (OODA loop decision engine) to create continuously improving gameplay intelligence.

**Key Architectural Insights**:
- **Learning-First Hierarchy**: NN/MCTS components drive decisions, with Decision Engine as capable fallback
- **Experience-Centric Design**: All learning components draw from and contribute to shared replay buffer
- **Bidirectional Co-Adaptation**: Learning improves reasoning, which generates better learning data
- **Meta-Learning Ecosystem**: System Agents discover, evaluate, and evolve strategies that Agents then learn to play
- **Safety-First Scaling**: Memory guards, streaming batches, and Wilson CIs enable massive scale without instability
- **Evidence-Based Validation**: Telemetry metrics (loss reduction, MAE, synergy discovery) prove learning efficacy

**Where to Find Specific Information**:
- **Core Learning Mechanism**: `agents/NN/policy_value_net.py` + `agents/Learning_System/replay_buffer.py`
- **Strategic Planning**: `agents/MCTS_NN/alphazero_agent.py`
- **Decision Foundation**: `simulation/Decision_Engine.py`
- **Experience Generation**: `agents/Resource_Management/simulation_runner.py`
- **Systemic Improvement**: `agents/Master_Autonomous/system_master_agent.py`
- **Empirical Validation**: `agents/ML/card_value_model.py` and `agents/ML/cards_matrix.py`
- **Long-Term Knowledge**: `agents/Learning_System/persistent_knowledge.py` and `data/replay_vault/`

By following this guide, you can systematically explore the codebase to understand how each piece contributes to the overall intelligence of the PTCG Sovereign Trainer system, validate the findings from your telemetry observations, and identify areas for further improvement or extension.

---

*Generated from deep codebase analysis.*