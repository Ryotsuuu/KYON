#!/usr/bin/env python3
"""Package KYON v2 into a complete zip with ALL files.

CRITICAL: This script ensures cg/ folder includes ALL platform binaries:
  - cg.dll (Windows)
  - libcg.so (Linux x86_64)
  - libcg-arm64.so (Linux ARM64)
  - libcg.dylib (macOS)

Also includes: data/, agents/, simulation/, ptcg-system/, ptcg.py, main.py, deck.csv
"""
import os
import sys
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent.parent / "download"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_NAME = "KYON_v2.zip"
ZIP_PATH = OUT_DIR / ZIP_NAME


def should_include(fpath: Path) -> bool:
    """Decide if a file should be included in the zip."""
    name = fpath.name
    # Skip pycache, __pycache__, .pyc
    if '__pycache__' in str(fpath):
        return False
    if name.endswith('.pyc'):
        return False
    # Skip graphify-out, tests, notebooks
    for skip_dir in ['graphify-out', 'tests', '.git', 'node_modules', 'charts', 'logs']:
        if f'/{skip_dir}/' in str(fpath).replace('\\', '/'):
            return False
    # Skip oversized files (>50MB)
    try:
        if fpath.stat().st_size > 50 * 1024 * 1024:
            return False
    except Exception:
        pass
    return True


def main():
    print(f"Packaging KYON v2 from {ROOT}")
    print(f"Output: {ZIP_PATH}")
    print()

    # Verify cg binaries exist
    cg_dir = ROOT / "cg"
    cg_libs = ['cg.dll', 'libcg.so', 'libcg.dylib', 'libcg-arm64.so']
    print("Checking cg native libraries:")
    for lib in cg_libs:
        lib_path = cg_dir / lib
        if lib_path.exists():
            size_kb = lib_path.stat().st_size / 1024
            print(f"  [OK] {lib} ({size_kb:.0f} KB)")
        else:
            print(f"  [MISSING] {lib} - WILL NOT BE IN ZIP!")
    print()

    # Collect all files to include
    files_to_add = []

    # 1. Top-level Python files
    for f in ROOT.glob("*.py"):
        if should_include(f):
            files_to_add.append((f, f"ptcg_v2/{f.name}"))

    # 2. deck.csv
    deck_csv = ROOT / "deck.csv"
    if deck_csv.exists():
        files_to_add.append((deck_csv, "ptcg_v2/deck.csv"))

    # 3. cg/ directory - ALL files (especially binaries)
    if cg_dir.is_dir():
        for f in cg_dir.iterdir():
            if f.is_file() and should_include(f):
                files_to_add.append((f, f"ptcg_v2/cg/{f.name}"))

    # 4. agents/ directory - all .py files recursively
    agents_dir = ROOT / "agents"
    if agents_dir.is_dir():
        for f in agents_dir.rglob("*.py"):
            if should_include(f):
                rel = f.relative_to(ROOT)
                files_to_add.append((f, f"ptcg_v2/{rel}"))

    # 5. simulation/ directory
    sim_dir = ROOT / "simulation"
    if sim_dir.is_dir():
        for f in sim_dir.iterdir():
            if f.is_file() and should_include(f):
                files_to_add.append((f, f"ptcg_v2/simulation/{f.name}"))

    # 6. data/ directory - json and csv
    data_dir = ROOT / "data"
    if data_dir.is_dir():
        for f in data_dir.iterdir():
            if f.is_file() and f.suffix in ('.json', '.csv') and should_include(f):
                files_to_add.append((f, f"ptcg_v2/data/{f.name}"))

    # 7. ptcg-system/ directory - json files
    ps_dir = ROOT / "ptcg-system"
    if ps_dir.is_dir():
        for f in ps_dir.iterdir():
            if f.is_file() and f.suffix == '.json':
                files_to_add.append((f, f"ptcg_v2/ptcg-system/{f.name}"))

    # 8. scripts/ directory
    scripts_dir = ROOT / "scripts"
    if scripts_dir.is_dir():
        for f in scripts_dir.iterdir():
            if f.is_file() and f.suffix == '.py':
                files_to_add.append((f, f"ptcg_v2/scripts/{f.name}"))

    # Write zip
    print(f"Writing {len(files_to_add)} files to {ZIP_NAME}...")
    total_size = 0
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files_to_add:
            zf.write(src, arcname)
            total_size += src.stat().st_size

    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDone! {ZIP_NAME}: {zip_size_mb:.1f} MB ({len(files_to_add)} files, {total_size/(1024*1024):.1f} MB uncompressed)")

    # Verify cg binaries are in the zip
    print(f"\nVerifying cg binaries in zip:")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        for lib in cg_libs:
            entry = f"ptcg_v2/cg/{lib}"
            if entry in zf.namelist():
                info = zf.getinfo(entry)
                print(f"  [OK] {lib} ({info.file_size/1024:.0f} KB compressed)")
            else:
                print(f"  [MISSING] {lib} - NOT IN ZIP!")

    return ZIP_PATH


if __name__ == '__main__':
    main()
