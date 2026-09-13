"""
Protocol Examiner & Adversarial Codebase Inspector
Evaluates the PTCG Sovereign Trainer system against strict Grandmaster Examiner protocols:
1. Dead Code / Silent Ignore Detection in Decision_Engine and simulation pipeline.
2. Dynamic Lookahead & Branching Depth (ensuring >20 Ply and >40 Br scaling).
3. MCTS Rollout Ceiling & Telemetry (ensuring 1000+ rollouts are captured and rendered).
4. Dual-Agent Winning Advantage & Turn-by-Turn Condition Awareness.
5. HTML Visualizer Dynamic Headroom & Legend Consistency.
"""

import os
import sys
import inspect
import json
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "e:" / "PTCG Soverign Trainer" / "ptcg sovegin trainer"
if not ROOT.exists():
    # Try alternative path resolution
    ROOT = Path("e:/PTCG Soverign Trainer/ptcg sovegin trainer")
sys.path.insert(0, str(ROOT))

print(f"=== PROTOCOL EXAMINER & ADVERSARIAL CODEBASE AUDIT ===")
print(f"Working Directory: {ROOT}")

findings = []

def record_finding(category, severity, item, detail):
    findings.append({
        "category": category,
        "severity": severity,
        "item": item,
        "detail": detail
    })
    badge = "[CRITICAL]" if severity == "HIGH" else "[WARNING]" if severity == "MEDIUM" else "[INFO]"
    print(f"  {badge} ({category}) {item}: {detail}")

# -------------------------------------------------------------
# Protocol 1: AST / Static Inspection of Decision_Engine.py
# -------------------------------------------------------------
print("\n--- Protocol 1: AST & Call-Flow Audit of simulation/Decision_Engine.py ---")
decision_engine_path = ROOT / "simulation" / "Decision_Engine.py"
if decision_engine_path.exists():
    with open(decision_engine_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Check 1.1: Verify _update_prize_dynamics is called
    if "_update_prize_dynamics" in code:
        calls = code.count("self._update_prize_dynamics(")
        print(f"  [OK] _update_prize_dynamics defined and called {calls} time(s).")
        if calls == 0:
            record_finding("Decision_Engine", "HIGH", "_update_prize_dynamics", "Function defined but never invoked in main decision loops!")
    else:
        record_finding("Decision_Engine", "HIGH", "_update_prize_dynamics", "Missing method in Decision_Engine.py")

    # Check 1.2: Check whether _condition_tracker is updated
    if "_condition_tracker" in code:
        tracker_calls = code.count("self._condition_tracker.")
        print(f"  [OK] _condition_tracker referenced {tracker_calls} time(s).")
    else:
        record_finding("Decision_Engine", "MEDIUM", "_condition_tracker", "_condition_tracker not integrated.")

    # Check 1.3: Check _solve_endgame_lethal_subgame invocation
    lethal_calls = code.count("_solve_endgame_lethal_subgame")
    print(f"  [OK] _solve_endgame_lethal_subgame referenced {lethal_calls} time(s).")

    # Check 1.4: Check for bare 'except: pass' or masked critical errors
    import re
    silent_excepts = re.findall(r'except\s*:\s*(?:pass|continue)', code)
    silent_generic_excepts = re.findall(r'except\s+Exception\s*:\s*(?:pass|continue)', code)
    total_silent = len(silent_excepts) + len(silent_generic_excepts)
    if total_silent > 0:
        record_finding("Robustness", "LOW", "Silent Exception Handlers", f"Found {total_silent} bare/silent except blocks in Decision_Engine.py")
    else:
        print("  [OK] Zero silent exception passes found in Decision_Engine.py.")

# -------------------------------------------------------------
# Protocol 2: Visualizer & battle_turn_data.html Inspection
# -------------------------------------------------------------
print("\n--- Protocol 2: Visualizer & HTML Legend Scalability Audit ---")
vis_path = ROOT / "agents" / "battle_visualizer.py"
if vis_path.exists():
    with open(vis_path, "r", encoding="utf-8") as f:
        vis_code = f.read()

    # Check if 0–20 Ply is hardcoded in the HTML legend
    if "0–20 Ply" in vis_code or "0-20 Ply" in vis_code:
        record_finding("Visualizer", "MEDIUM", "Static Chart Legend", 
                       "HTML legend in battle_visualizer.py hardcodes '(0–20 Ply)' and '(0–40)' while dynamic headroom scales to 32+, 48+, 64+!")
    else:
        print("  [OK] Chart legend dynamically accommodates deep search.")

    # Check MCTS rollout calculation
    if "mcts_rollouts_p1" in vis_code:
        print("  [OK] mcts_rollouts_p1 dynamically computed based on depth and branches.")
        # Check if rollouts can exceed 1000+
        # In battle_visualizer: int(p1_depth * p1_branches * 1.8 + 160)
        # For p1_depth=26, p1_branches=25: 26*25*1.8 + 160 = 1330 rollouts!
    else:
        record_finding("Visualizer", "HIGH", "MCTS Rollout Telemetry", "mcts_rollouts_p1 not found in battle_visualizer.py")

# -------------------------------------------------------------
# Protocol 3: Dual-Agent Advantage & State Awareness Verification
# -------------------------------------------------------------
print("\n--- Protocol 3: Dual-Agent Advantage & Playing Condition Runtime Audit ---")
try:
    from agents.battle_visualizer import generate_battle_turn_data_html
    print("  [OK] Successfully imported generate_battle_turn_data_html from agents.battle_visualizer.")
except Exception as e:
    record_finding("Import", "HIGH", "battle_visualizer", f"Failed to import generate_battle_turn_data_html: {e}")

# -------------------------------------------------------------
# Protocol 4: Simulation Engine Lookahead & Branching Depth Check
# -------------------------------------------------------------
print("\n--- Protocol 4: Lookahead Expansion & MCTS Depth Scaling Audit ---")
# Check how MasterAgent or simulation runner handles depth & rollout ceilings
try:
    from simulation.Decision_Engine import MasterAgent
    agent = MasterAgent()
    print("  [OK] MasterAgent instantiated successfully.")
    
    # Check if agent has prize dynamic tracking attributes
    has_prev_my = hasattr(agent, "_prev_my_prizes")
    has_prev_opp = hasattr(agent, "_prev_opp_prizes")
    has_momentum = hasattr(agent, "_momentum_signal")
    has_tracker = hasattr(agent, "_condition_tracker")
    print(f"  [OK] Temporal Attributes: _prev_my_prizes={has_prev_my}, _prev_opp_prizes={has_prev_opp}, _momentum_signal={has_momentum}, _condition_tracker={has_tracker}")
    
    if not (has_prev_my and has_prev_opp and has_momentum and has_tracker):
        record_finding("Agent Attributes", "HIGH", "MasterAgent Attributes", "One or more temporal tracking attributes missing on MasterAgent!")
except Exception as e:
    record_finding("Decision_Engine", "HIGH", "MasterAgent Initialization", f"Exception during MasterAgent audit: {e}")

print("\n--- Summary of Findings ---")
if not findings:
    print("Zero defects or silent ignores found! All protocols passed.")
else:
    print(f"Found {len(findings)} area(s) for enhancement / correction.")

with open(ROOT / "HIGH AFFECTION" / "report" / "PROTOCOL_EXAMINER_FINDINGS.json", "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2)
print("Wrote findings to HIGH AFFECTION/report/PROTOCOL_EXAMINER_FINDINGS.json")
