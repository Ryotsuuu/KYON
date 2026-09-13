#!/usr/bin/env python3
"""Package Kaggle submission tar.gz.

Includes: main.py, deck.csv, cg/, simulation/, models/, agents/, csv-data/, ptcg-system/
"""
import os
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent.parent / "download"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _add_directory_recursively(tar, dir_path: Path, arc_prefix: str):
    if not dir_path.is_dir():
        return
    for item in sorted(dir_path.rglob("*")):
        if item.is_file():
            # Skip python cache, logs, checkpoints, and temporary working files
            if '__pycache__' in item.parts or item.suffix in ('.pyc', '.log', '.tmp'):
                continue
            if 'checkpoints' in item.parts or '.git' in item.parts:
                continue
            rel = item.relative_to(dir_path)
            arc_name = f"{arc_prefix}/{rel.as_posix()}"
            tar.add(item, arcname=arc_name)
            print(f"  [OK] {arc_name}")


def main():
    out_path = OUT_DIR / "kaggle_submission.tar.gz"
    root_submission = ROOT / "submission.tar.gz"
    print(f"Packaging Kaggle submission to {out_path} and {root_submission}")

    for target_tar in (out_path, root_submission):
        with tarfile.open(target_tar, "w:gz") as tar:
            # 1. main.py
            main_py = ROOT / "main.py"
            if main_py.exists():
                tar.add(main_py, arcname="main.py")
                print(f"  [OK] main.py")
            else:
                sim_path = ROOT / "simulation" / "Decision_Engine.py"
                if sim_path.exists():
                    tar.add(sim_path, arcname="main.py")
                    print(f"  [OK] main.py (from simulation/Decision_Engine.py)")

            # 2. deck.csv
            deck_csv = ROOT / "deck.csv"
            if deck_csv.exists():
                tar.add(deck_csv, arcname="deck.csv")
                print(f"  [OK] deck.csv")
            else:
                print(f"  [WARN] deck.csv not found")

            # 3. cg/ directory
            _add_directory_recursively(tar, ROOT / "cg", "cg")

            # 4. simulation/ directory
            _add_directory_recursively(tar, ROOT / "simulation", "simulation")

            # 5. models/ directory (PyTorch GPU Weights & ML Value Model)
            _add_directory_recursively(tar, ROOT / "models", "models")

            # 6. csv-data/ directory
            _add_directory_recursively(tar, ROOT / "csv-data", "csv-data")

            # 7. agents/ directory
            _add_directory_recursively(tar, ROOT / "agents", "agents")

            # 8. ptcg-system/ directory
            _add_directory_recursively(tar, ROOT / "ptcg-system", "ptcg-system")

        size_mb = target_tar.stat().st_size / (1024 * 1024)
        print(f"\nDone! {target_tar.name}: {size_mb:.2f} MB ({target_tar})")


if __name__ == '__main__':
    main()

