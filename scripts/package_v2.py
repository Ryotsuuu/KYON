#!/usr/bin/env python3
"""
Package KYON v2 into a correct zip archive.

Includes ALL files needed:
- ptcg.py (CLI)
- main.py (Kaggle entry)
- cg/ (native libraries + Python bindings)
- simulation/ (Decision Engine)
- agents/ (all modules)
- data/ (csv-data JSON files)
- ptcg-system/ (settings, catalog, registry)
- docs/ (documentation)
- scripts/ (utility scripts)
- deck.csv (default deck)

Excludes: __pycache__, .pyc, .tar.gz, tool-results/
"""
import zipfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent  # /home/z/my-project/v2_build/
ZIP_NAME = "KYON_v2.zip"
ZIP_PATH = OUT_DIR / ZIP_NAME

# Files/dirs to include
INCLUDE_DIRS = [
    "cg", "simulation", "agents", "data", "ptcg-system", "docs", "scripts",
]

INCLUDE_FILES = [
    "ptcg.py", "main.py", "deck.csv",
]

# Patterns to exclude
EXCLUDE_PATTERNS = [
    "__pycache__", ".pyc", ".tar.gz", ".zip",
    "tool-results", ".ipynb_checkpoints",
]


def should_include(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    return not any(excl in parts for excl in EXCLUDE_PATTERNS)


def main():
    print(f"Packaging KYON v2 from {ROOT}")
    print(f"Output: {ZIP_PATH}")
    
    file_count = 0
    
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include individual files
        for fname in INCLUDE_FILES:
            fpath = ROOT / fname
            if fpath.exists():
                arcname = f"ptcg_v2/{fname}"
                zf.write(fpath, arcname)
                file_count += 1
                print(f"  + {arcname}")
        
        # Include directories
        for dirname in INCLUDE_DIRS:
            dirpath = ROOT / dirname
            if not dirpath.is_dir():
                print(f"  [SKIP] {dirname}/ (not found)")
                continue
            
            for f in sorted(dirpath.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(ROOT)
                rel_str = str(rel).replace(os.sep, "/")
                if not should_include(rel_str):
                    continue
                arcname = f"ptcg_v2/{rel_str}"
                zf.write(f, arcname)
                file_count += 1
    
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDone: {file_count} files, {size_mb:.1f} MB -> {ZIP_PATH}")
    return ZIP_PATH


if __name__ == "__main__":
    main()
