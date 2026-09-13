#!/usr/bin/env python3
"""
Verification script to check if CardValueModel is actually used in Decision Engine.
This script DOES NOT MODIFY any code - it only reads and analyzes the existing files.
"""

import os
import re

def test_cvm_usage_in_decision_engine():
    """Test if CardValueModel predictions are actually used to score actions."""

    print("=" * 60)
    print("CVM USAGE VERIFICATION TEST")
    print("=" * 60)

    # Read the Decision Engine source file
    decision_engine_path = os.path.join("simulation", "Decision_Engine.py")

    try:
        with open(decision_engine_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print("[] Successfully read Decision Engine source file")
    except Exception as e:
        print(f"[x] Failed to read Decision Engine source file: {e}")
        return False

    # Check if cvm is instantiated
    cvm_instantiated = False
    cvm_instantiation_lines = []

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "from agents.ML.card_value_model import get_card_value_model" in line:
            cvm_instantiated = True
            cvm_instantiation_lines.append(f"Line {i+1}: {line.strip()}")
        if "cvm = get_card_value_model()" in line:
            cvm_instantiation_lines.append(f"Line {i+1}: {line.strip()}")

    if cvm_instantiated:
        print("[] CardValueModel IS INSTANTIATED:")
        for line in cvm_instantiation_lines:
            print(f"    {line}")
    else:
        print("[x] CardValueModel is NOT instantiated")
        return False

    # Check if cvm is actually USED to call predict methods
    cvm_used_for_prediction = False
    cvm_usage_lines = []

    # Look for calls to cvm.predict_* methods
    predict_patterns = [
        r'cvm\.predict_contextual_value\(',
        r'cvm\.predict_value\(',
        r'cvm\.predict\('
    ]

    for i, line in enumerate(lines):
        for pattern in predict_patterns:
            if re.search(pattern, line):
                cvm_used_for_prediction = True
                cvm_usage_lines.append(f"Line {i+1}: {line.strip()}")

    if cvm_used_for_prediction:
        print("\n[] CardValueModel IS USED FOR PREDICTIONS:")
        for line in cvm_usage_lines:
            print(f"    {line}")
        print("\n[] VERIFICATION RESULT: PASS - CVM predictions are being used")
        return True
    else:
        print("\n[x] CardValueModel is INSTANTIATED BUT NEVER USED FOR PREDICTIONS")
        print("    This confirms the silent issue: ML model computed but ignored")

        # Show where cvm is defined but not used
        print("\n[] CVM INSTANTIATION LOCATIONS (NOT FOLLOWED BY USAGE):")
        for line in cvm_instantiation_lines:
            print(f"    {line}")

        print("\n[] EXPECTED USAGE PATTERNS (NOT FOUND):")
        print("    - cvm.predict_contextual_value(...)")
        print("    - cvm.predict_value(...)")
        print("    - cvm.predict(...)")

        return False

def test_pmi_usage_for_comparison():
    """Test PMI usage as a positive control to show what proper usage looks like."""

    print("\n" + "=" * 60)
    print("PMI USAGE VERIFICATION (POSITIVE CONTROL)")
    print("=" * 60)

    decision_engine_path = os.path.join("simulation", "Decision_Engine.py")

    try:
        with open(decision_engine_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"[x] Failed to read Decision Engine source file: {e}")
        return False

    lines = content.split('\n')

    # Check if pmi_matrix is instantiated
    pmi_instantiated = False
    pmi_instantiation_lines = []

    for i, line in enumerate(lines):
        if "from agents.ML.cards_matrix import get_cards_matrix" in line:
            pmi_instantiated = True
            pmi_instantiation_lines.append(f"Line {i+1}: {line.strip()}")
        if "pmi_matrix = get_cards_matrix()" in line:
            pmi_instantiation_lines.append(f"Line {i+1}: {line.strip()}")

    if pmi_instantiated:
        print("[] PMI Matrix IS INSTANTIATED:")
        for line in pmi_instantiation_lines:
            print(f"    {line}")
    else:
        print("[x] PMI Matrix is NOT instantiated")
        return False

    # Check if pmi_matrix is actually USED
    pmi_used = False
    pmi_usage_lines = []

    # Look for calls to pmi_matrix.get_synergy or similar methods
    pmi_patterns = [
        r'pmi_matrix\.get_synergy\(',
        r'\.get_synergy\('  # More general in case of different variable name
    ]

    for i, line in enumerate(lines):
        for pattern in pmi_patterns:
            if re.search(pattern, line):
                pmi_used = True
                pmi_usage_lines.append(f"Line {i+1}: {line.strip()}")

    if pmi_used:
        print("\n[] PMI Matrix IS USED:")
        for line in pmi_usage_lines:
            print(f"    {line}")
        print("\n[] CONTROL TEST RESULT: PASS - PMI is properly used")
        return True
    else:
        print("\n[x] PMI Matrix is INSTANTIATED BUT NEVER USED")
        return False

def main():
    """Run all verification tests."""
    print("PTCG SOVEREIGN TRAINER - CVM USAGE VERIFICATION")
    print("This script verifies whether the CardValueModel is actually used\n")

    # Test CVM usage (should fail - demonstrating the issue)
    cvm_result = test_cvm_usage_in_decision_engine()

    # Test PMI usage as control (should pass)
    pmi_result = test_pmi_usage_for_comparison()

    print("\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)

    if cvm_result:
        print("[] CVM USAGE: VERIFIED - CardValueModel predictions are being used")
    else:
        print("[x] CVM USAGE: FAILED - CardValueModel is instantiated but NOT used")
        print("    THIS CONFIRMS THE SILENT ISSUE DESCRIBED IN SILENT_ISSUE_REPORT.md")

    if pmi_result:
        print("[] PMI USAGE: VERIFIED - PMI matrix is properly used (control test)")
    else:
        print("[x] PMI USAGE: FAILED - Unexpected: control test should pass")

    print("\n" + "=" * 60)
    if not cvm_result and pmi_result:
        print("CONCLUSION: Silent issue confirmed - CVM trained but not used")
        print("            While PMI is properly integrated, CVM predictions are ignored")
    elif cvm_result and pmi_result:
        print("CONCLUSION: Both systems properly integrated - issue may be fixed")
    else:
        print("CONCLUSION: Unexpected test results - investigate further")
    print("=" * 60)

    # Return True if issue is present (useful for automated testing)
    return not cvm_result  # True if CVM is NOT used (issue present)

if __name__ == "__main__":
    issue_present = main()
    exit(1 if issue_present else 0)  # Exit with error code if issue present