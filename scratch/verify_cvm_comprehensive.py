#!/usr/bin/env python3
"""
Comprehensive verification script to check CVM usage across all decision methods.
This script DOES NOT MODIFY any code - it only reads and analyzes the existing files.
"""

import os
import re

def check_cvm_usage_in_method(method_name, method_content):
    """Check if CVM is used for predictions in a given method content."""
    # Look for calls to cvm.predict_* methods
    predict_patterns = [
        r'cvm\.predict_contextual_value\(',
        r'cvm\.predict_value\(',
        r'cvm\.predict\('
    ]

    for pattern in predict_patterns:
        if re.search(pattern, method_content):
            return True
    return False

def extract_method_content(lines, method_name):
    """Extract the content of a method from lines."""
    method_start = -1
    for i, line in enumerate(lines):
        if f"def {method_name}" in line:
            method_start = i
            break

    if method_start == -1:
        return None

    # Find the end of the method (next method or class definition, or end of file)
    method_end = len(lines)
    for i in range(method_start + 1, len(lines)):
        line = lines[i]
        if line.strip().startswith("def ") or line.strip().startswith("class "):
            method_end = i
            break

    return '\n'.join(lines[method_start:method_end])

def main():
    print("=" * 70)
    print("COMPREHENSIVE CVM USAGE VERIFICATION")
    print("Checking all major decision methods in Decision Engine")
    print("=" * 70)

    # Read the Decision Engine source file
    decision_engine_path = os.path.join("simulation", "Decision_Engine.py")

    try:
        with open(decision_engine_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print("[] Successfully read Decision Engine source file")
    except Exception as e:
        print(f"[x] Failed to read Decision Engine source file: {e}")
        return False

    lines = content.split('\n')

    # Check if cvm is instantiated at all
    cvm_instantiated = False
    for line in lines:
        if "from agents.ML.card_value_model import get_card_value_model" in line:
            cvm_instantiated = True
            break
        if "cvm = get_card_value_model()" in line:
            cvm_instantiated = True
            break

    if not cvm_instantiated:
        print("[x] CardValueModel is NOT instantiated at all")
        return False
    else:
        print("[] CardValueModel IS INSTANTIATED")

    # Define the major decision methods to check
    methods_to_check = [
        "_select_evolve",           # EVOLUTIONS
        "_select_attach_to_ooda",   # ATTACHMENT
        "_select_cards_search",     # PRIZE CARDS / TRAINER ITEMS
        "_select_attach_from",      # ATTACH FROM BENCH
        "_select_attack_ooda",      # ATTACKS
        "_select_best_basic",       # BEST BASIC
        "_bench_all",               # BENCH ALL
        "_pick_best_bench_ooda",    # PICK BEST BENCH
    ]

    results = {}

    for method_name in methods_to_check:
        method_content = extract_method_content(lines, method_name)
        if method_content is None:
            results[method_name] = "NOT_FOUND"
            continue

        cvm_used = check_cvm_usage_in_method(method_name, method_content)
        results[method_name] = "USED" if cvm_used else "NOT_USED"

    # Print results
    print("\n" + "-" * 70)
    print("CVM USAGE BY METHOD:")
    print("-" * 70)

    used_count = 0
    total_count = 0

    for method_name, result in results.items():
        if result == "NOT_FOUND":
            print(f"[{method_name:<25}] : METHOD NOT FOUND")
            continue

        total_count += 1
        if result == "USED":
            used_count += 1
            print(f"[{method_name:<25}] : [] USED - CVM predictions utilized")
        else:
            print(f"[{method_name:<25}] : [x] NOT USED - CVM NOT utilized")

    print("-" * 70)
    print(f"SUMMARY: {used_count}/{total_count} methods use CVM predictions")

    # Specific analysis for the key methods we know about
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF KEY METHODS")
    print("=" * 70)

    # Check EVOLUTIONS method
    ev_content = extract_method_content(lines, "_select_evolve")
    if ev_content and check_cvm_usage_in_method("_select_evolve", ev_content):
        print("[] EVOLUTIONS (_select_evolve): CVM USED")
        # Show the usage
        lines_ev = ev_content.split('\n')
        for i, line in enumerate(lines_ev):
            if 'cvm.predict_contextual_value' in line:
                print(f"    Line {i+1} in method: {line.strip()}")
    else:
        print("[x] EVOLUTIONS (_select_evolve): CVM NOT USED")

    # Check ATTACHMENT method
    att_content = extract_method_content(lines, "_select_attach_to_ooda")
    if att_content and check_cvm_usage_in_method("_select_attach_to_ooda", att_content):
        print("[] ATTACHMENT (_select_attach_to_ooda): CVM USED")
        # Show the usage
        lines_att = att_content.split('\n')
        for i, line in enumerate(lines_att):
            if 'cvm.predict_contextual_value' in line:
                print(f"    Line {i+1} in method: {line.strip()}")
    else:
        print("[x] ATTACHMENT (_select_attach_to_ooda): CVM NOT USED")

    # Check SUPPORTERS method (this is in _select_cards_search)
    search_content = extract_method_content(lines, "_select_cards_search")
    if search_content:
        supp_used = check_cvm_usage_in_method("_select_cards_search", search_content)
        if supp_used:
            print("[] SUPPORTERS (in _select_cards_search): CVM USED")
            # Show the usage
            lines_search = search_content.split('\n')
            for i, line in enumerate(lines_search):
                if 'cvm.predict_contextual_value' in line:
                    print(f"    Line {i+1} in method: {line.strip()}")
        else:
            print("[x] SUPPORTERS (in _select_cards_search): CVM NOT USED")

            # But check if it's used in the supporters subsection within this method
            # Look for the supporters section specifically
            if "SUPPORTERS (Draw Engine, Strategic Gust KO, Healing & Acceleration)" in search_content:
                print("    [*] Supporters section FOUND in method")
                # Extract supporters section
                lines_search = search_content.split('\n')
                in_supporters = False
                supporters_lines = []
                for line in lines_search:
                    if "SUPPORTERS (Draw Engine, Strategic Gust KO, Healing & Acceleration)" in line:
                        in_supporters = True
                        supporters_lines.append(line)
                        continue
                    if in_supporters:
                        if line.strip().startswith("# ") and "TRAINER ITEMS" in line:
                            break
                        supporters_lines.append(line)

                supporters_content = '\n'.join(supporters_lines)
                if check_cvm_usage_in_method("supporters section", supporters_content):
                    print("    [] CVM USED in supporters subsection")
                    lines_supp = supporters_content.split('\n')
                    for i, line in enumerate(lines_supp):
                        if 'cvm.predict_contextual_value' in line:
                            print(f"       Line {i+1} in supporters: {line.strip()}")
                else:
                    print("    [x] CVM NOT USED in supporters subsection")
    else:
        print("[x] _select_cards_search method NOT FOUND")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if used_count == 0:
        print("[x] CRITICAL: NO METHODS USE CVM PREDICTIONS")
        print("    The CardValueModel is instantiated but completely ignored")
    elif used_count == total_count:
        print("[] ALL METHODS USE CVM PREDICTIONS")
        print("    The CardValueModel is properly integrated throughout")
    else:
        print("[!] PARTIAL INTEGRATION: Some methods use CVM, others don't")
        print("    This creates inconsistent decision-making")
        print("    Methods using CVM:", [m for m, r in results.items() if r == "USED"])
        print("    Methods NOT using CVM:", [m for m, r in results.items() if r == "NOT_USED"])

        # Specific finding
        search_result = results.get("_select_cards_search", "ERROR")
        if search_result == "NOT_USED":
            print("\n[x] KEY FINDING: PRIZE CARD/TRAINER ITEM SELECTION IGNORES CVM")
            print("    The _select_cards_search method (handles TO_HAND, TO_DECK, TO_PRIZE, etc.)")
            print("    does NOT use CVM predictions for utility scaling, despite using:")
            print("    - PMI matrix for synergy boosting")
            print("    - Hardcoded bonuses for card types, stages, energy")
            print("    - Prize trap filtering")
            print("    This creates a blind spot in card valuation for retrieval/prize contexts")

    # Return True if issue is present (useful for automated testing)
    # Issue is present if ANY method doesn't use CVM
    issue_present = any(result == "NOT_USED" for result in results.values() if result not in ["NOT_FOUND", "ERROR"])
    return issue_present

if __name__ == "__main__":
    issue_present = main()
    exit(1 if issue_present else 0)  # Exit with error code if issue present