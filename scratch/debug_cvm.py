#!/usr/bin/env python3
"""
Debug script to check CVM usage detection.
"""

import os
import re

def main():
    print("DEBUGGING CVM USAGE DETECTION")
    print("=" * 50)

    # Read the Decision Engine source file
    decision_engine_path = os.path.join("simulation", "Decision_Engine.py")

    try:
        with open(decision_engine_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print("[] Successfully read Decision Engine source file")
    except Exception as e:
        print(f"[x] Failed to read Decision Engine source file: {e}")
        return

    # Check for cvm instantiation
    if "cvm = get_card_value_model()" in content:
        print("[] FOUND: cvm = get_card_value_model()")
    else:
        print("[x] NOT FOUND: cvm = get_card_value_model()")

    # Check for cvm.predict_contextual_value
    if "cvm.predict_contextual_value" in content:
        print("[] FOUND: cvm.predict_contextual_value")
        # Show all occurrences
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "cvm.predict_contextual_value" in line:
                print(f"    Line {i+1}: {line.strip()}")
    else:
        print("[x] NOT FOUND: cvm.predict_contextual_value")

    # Test regex
    pattern = r'cvm\.predict_contextual_value\('
    matches = re.findall(pattern, content)
    print(f"[] Regex matches for pattern '{pattern}': {len(matches)}")
    if matches:
        print(f"    Matches: {matches}")

    # Test line by line
    lines = content.split('\n')
    print("\n[] LINE BY LINE CHECK:")
    for i, line in enumerate(lines):
        if 'cvm.predict_contextual_value' in line:
            print(f"    Line {i+1}: MATCH - {line.strip()}")
        elif 'cvm' in line and 'predict' in line:
            print(f"    Line {i+1}: POSSIBLE - {line.strip()}")

if __name__ == "__main__":
    main()