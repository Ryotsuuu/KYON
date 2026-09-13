# 🏎️ KYON Sovereign High-Throughput Simulation & Statistical Telemetry Guide

```
███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
███████╗██║██╔████╔██║██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

---

## 📑 Table of Contents
1. [Simulation Engine Architecture](#1-simulation-engine-architecture)
2. [Multi-Worker Concurrency & Thread Isolation](#2-multi-worker-concurrency--thread-isolation)
3. [Empirical Throughput Benchmarks](#3-empirical-throughput-benchmarks)
4. [Statistical Telemetry & Mathematical Formulations](#4-statistical-telemetry--mathematical-formulations)
   - [Wilson Score 95% Confidence Interval](#wilson-score-95-confidence-interval)
   - [Turn Distribution Metrics & Dispersion](#turn-distribution-metrics--dispersion)
   - [Seat Advantage & First-Player Tempo Analytics](#seat-advantage--first-player-tempo-analytics)
5. [Persistent Match Telemetry & Real-Time Sync](#5-persistent-match-telemetry--real-time-sync)
6. [Meta Gauntlet & Kaggle Submission Readiness](#6-meta-gauntlet--kaggle-submission-readiness)

---

## 1. Simulation Engine Architecture

The KYON simulation engine bridges low-level C-Engine execution with high-level Python decision heuristics and statistical aggregation:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION RUNNER CONCURRENCY ARCHITECTURE                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                          Master Dispatcher (ptcg.py / simulate)                        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               │                            │                            │
      ┌────────▼────────┐          ┌────────▼────────┐          ┌────────▼────────┐
      │ Worker Thread 1 │          │ Worker Thread 2 │          │ Worker Thread N │
      ├─────────────────┤          ├─────────────────┤          ├─────────────────┤
      │ • cg.sim Context│          │ • cg.sim Context│          │ • cg.sim Context│
      │ • MasterAgent P1│          │ • MasterAgent P1│          │ • MasterAgent P1│
      │ • MasterAgent P2│          │ • MasterAgent P2│          │ • MasterAgent P2│
      │ • Match Log Strm│          │ • Match Log Strm│          │ • Match Log Strm│
      └────────┬────────┘          └────────┬────────┘          └────────┬────────┘
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │  Telemetry Aggregator   │
                               │ • Wilson 95% CIs        │
                               │ • Turn Distributions    │
                               │ • Seat Win Rates        │
                               │ • Live Registry Sync    │
                               └─────────────────────────┘
```

---

## 2. Multi-Worker Concurrency & Thread Isolation

- **Thread-Local State Isolation:** Each concurrent thread holds an independent pointer to a native `cg` game state, eliminating memory corruption and cross-thread race conditions.
- **Worker Auto-Scaling:** Automatically determines optimal worker concurrency based on logical CPU core counts:
  $$\text{Optimal Workers} = \max(2, \min(12, N_{\text{cores}} - 2))$$
- **Memory-Safe State Sampling:** Records lightweight match summaries during simulation, preventing memory spikes and keeping RAM usage below $75\%$.

---

## 3. Empirical Throughput Benchmarks

$$\begin{aligned}
\text{Single-Match Latency:} &\quad \mathbf{12.4\text{ ms}} \\
\text{Measured Multi-Worker Throughput:} &\quad \mathbf{67.22 \text{ to } 81.40 \text{ games/sec}} \\
\text{10-Second Simulation Volume:} &\quad \mathbf{672 \text{ to } 814 \text{ completed matches}} \\
\text{1,000-Game Batch Execution:} &\quad \mathbf{14.875\text{ seconds}} \ (\mathbf{55\times \text{ faster}} \text{ than unoptimized Python simulators})
\end{aligned}$$

---

## 4. Statistical Telemetry & Mathematical Formulations

Every simulated match series computes rigorous statistical telemetry to evaluate true competitive superiority:

### Wilson Score 95% Confidence Interval
For sample win rate $\hat{p} = \frac{W}{N}$, the asymmetric 95% confidence interval is computed as:

$$w = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}$$

where $z = 1.95996$. This accounts for sample variance in small-batch runs and prevents overconfidence.

---

### Turn Distribution Metrics & Dispersion
- **Mean Pacing ($\mu$):** Average turns per game across all completed matches.
- **Median ($P_{50}$):** Middle turn value robust to rare 100+ turn stall outliers.
- **Mode:** Most frequent turn count for decisive victories.
- **Standard Deviation ($\sigma$):**
  $$\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (t_i - \mu)^2}$$
- **Interquartile Range ($\text{IQR}$):** $\text{IQR} = P_{75} - P_{25}$, measuring core strategic match length.
- **Agent Victory Speed:** Distinct turn pacing computed separately for Player 1 victories and Player 2 victories.

---

### Seat Advantage & First-Player Tempo Analytics
Tracks the first-mover advantage inherent in Pokémon TCG rules (first turn cannot attack):
- **$P_1$ Win Rate when Going 1st vs Going 2nd.**
- **$P_2$ Win Rate when Going 1st vs Going 2nd.**
- **Global 1st-Player Tempo Win Rate:** Measures systemic first-turn advantage across the match sample.

---

## 5. Persistent Match Telemetry & Real-Time Sync

Every simulation executed via `ptcg.py simulate` or `ptcg.py matchups-simulation` persistently updates match records in `ptcg-system/agents_registry.json`:
- Aggregates `wins`, `losses`, `draws`, and total `games` for both competing agents.
- Powers the real-time rankings and win rates rendered by `python ptcg.py master report` and `python ptcg.py list`.

---

## 6. Meta Gauntlet & Kaggle Submission Readiness

The `matchups-simulation` command evaluates an agent across 14 elemental archetype champions (350 total games) and assigns an empirical readiness rating:

| Meta Win Rate Range | Readiness Status | Action Recommendation |
| :---: | :--- | :--- |
| **$\ge 75.0\%$** | `ELITE - HIGHLY RECOMMENDED` | Prime candidate for tournament submission. |
| **$60.0\% - 74.9\%$** | `COMPETITIVE - SUITABLE` | Strong viable contender; minor GA tuning recommended. |
| **$< 60.0\%$** | `NEEDS GA OPTIMIZATION` | Run `python ptcg.py upgrade` to evolve deck composition. |
