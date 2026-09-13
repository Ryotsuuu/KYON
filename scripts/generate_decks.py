#!/usr/bin/env python3
"""Generate all agent decks using csv-data powered builder."""
import json
import sys
import os
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load settings
with open(ROOT / "ptcg-system" / "settings.json") as f:
    settings = json.load(f)

# Import builder (requires cg module for API types, but csv_data works standalone)
from agents.csv_data import CsvDataIndex
from agents.csv_deck_builder import CsvDeckBuilder
from agents.archetype_engine import (
    get_all_archetypes_for_combo, ARCHETYPES_SINGLE_DUAL,
    ARCHETYPES_TRIPLE, ARCHETYPES_TEAM_ROCKET, ARCHETYPES_DRAGON,
)

# Energy types available
ENERGY_TYPES = [
    'Grass', 'Fire', 'Water', 'Lightning',
    'Psychic', 'Fighting', 'Darkness', 'Metal',
]

DATA_DIR = ROOT / "data"

def build_all_agents():
    print("Loading csv-data...")
    idx = CsvDataIndex(DATA_DIR)
    print(f"Loaded {len(idx.cards)} cards")

    builder = CsvDeckBuilder(idx)

    all_agents = {}
    agent_count = 0
    legal_count = 0
    illegal_agents = []

    # === SINGLE ENERGY COMBOS ===
    if settings["energy_combos"].get("single", False):
        print("\n=== BUILDING SINGLE ENERGY AGENTS ===")
        enabled_archetypes = {k: v for k, v in settings["archetypes"].items() if v}
        for etype in ENERGY_TYPES:
            for arch_name, enabled in enabled_archetypes.items():
                if not enabled:
                    continue
                agent_id = f"S_{etype[:3].upper()}_{arch_name}"
                try:
                    deck, info = builder.build_deck([etype], arch_name, 'single', seed=42 + agent_count)
                    all_agents[agent_id] = {
                        'deck': deck,
                        'info': info,
                        'energy_types': [etype],
                        'archetype': arch_name,
                        'combo_type': 'single',
                    }
                    agent_count += 1
                    if len(deck) == 60:
                        legal_count += 1
                    else:
                        illegal_agents.append((agent_id, len(deck)))
                except Exception as e:
                    print(f"  ERROR {agent_id}: {e}")

    # === DUAL ENERGY COMBOS ===
    if settings["energy_combos"].get("dual", False):
        print("\n=== BUILDING DUAL ENERGY AGENTS ===")
        enabled_archetypes = {k: v for k, v in settings["archetypes"].items() if v}
        for i, et1 in enumerate(ENERGY_TYPES):
            for et2 in ENERGY_TYPES[i+1:]:
                for arch_name, enabled in enabled_archetypes.items():
                    if not enabled:
                        continue
                    agent_id = f"D_{et1[:3].upper()}+{et2[:3].upper()}_{arch_name}"
                    try:
                        deck, info = builder.build_deck([et1, et2], arch_name, 'single', seed=42 + agent_count)
                        all_agents[agent_id] = {
                            'deck': deck,
                            'info': info,
                            'energy_types': [et1, et2],
                            'archetype': arch_name,
                            'combo_type': 'dual',
                        }
                        agent_count += 1
                        if len(deck) == 60:
                            legal_count += 1
                        else:
                            illegal_agents.append((agent_id, len(deck)))
                    except Exception as e:
                        print(f"  ERROR {agent_id}: {e}")

    # === TRIPLE ENERGY COMBOS ===
    if settings["energy_combos"].get("triple", False):
        print("\n=== BUILDING TRIPLE ENERGY AGENTS ===")
        enabled_archetypes = {k: v for k, v in settings.get("archetypes_for_triple", {}).items() if v}
        # Use top triple energy candidates from csv-data
        triple_combos = [
            ('Grass', 'Fire', 'Water'),
            ('Lightning', 'Psychic', 'Fighting'),
            ('Darkness', 'Metal', 'Colorless'),
            ('Grass', 'Lightning', 'Fighting'),
            ('Fire', 'Water', 'Psychic'),
        ]
        for combo in triple_combos:
            for arch_name, enabled in enabled_archetypes.items():
                if not enabled:
                    continue
                e_short = '+'.join(e[:3].upper() for e in combo)
                agent_id = f"T_{e_short}_{arch_name}"
                try:
                    deck, info = builder.build_deck(list(combo), arch_name, 'triple', seed=42 + agent_count)
                    all_agents[agent_id] = {
                        'deck': deck,
                        'info': info,
                        'energy_types': list(combo),
                        'archetype': arch_name,
                        'combo_type': 'triple',
                    }
                    agent_count += 1
                    if len(deck) == 60:
                        legal_count += 1
                    else:
                        illegal_agents.append((agent_id, len(deck)))
                except Exception as e:
                    print(f"  ERROR {agent_id}: {e}")

    # === TEAM ROCKET COMBOS ===
    if settings["energy_combos"].get("team_rocket", False):
        print("\n=== BUILDING TEAM ROCKET AGENTS ===")
        enabled_archetypes = {k: v for k, v in settings.get("archetypes_for_team_rocket", {}).items() if v}
        for arch_name, enabled in enabled_archetypes.items():
            if not enabled:
                continue
            agent_id = f"TR_{arch_name}"
            try:
                deck, info = builder.build_deck(['Psychic', 'Darkness'], arch_name, 'team_rocket', seed=42 + agent_count)
                all_agents[agent_id] = {
                    'deck': deck,
                    'info': info,
                    'energy_types': ['Psychic', 'Darkness'],
                    'archetype': arch_name,
                    'combo_type': 'team_rocket',
                }
                agent_count += 1
                if len(deck) == 60:
                    legal_count += 1
                else:
                    illegal_agents.append((agent_id, len(deck)))
            except Exception as e:
                print(f"  ERROR {agent_id}: {e}")

    # === DRAGON COMBOS ===
    if settings["energy_combos"].get("dragon", False):
        print("\n=== BUILDING DRAGON AGENTS ===")
        enabled_archetypes = {k: v for k, v in settings.get("archetypes_for_dragon", {}).items() if v}
        for arch_name, enabled in enabled_archetypes.items():
            if not enabled:
                continue
            agent_id = f"DR_{arch_name}"
            try:
                deck, info = builder.build_deck(['Dragon'], arch_name, 'dragon', seed=42 + agent_count)
                all_agents[agent_id] = {
                    'deck': deck,
                    'info': info,
                    'energy_types': ['Dragon'],
                    'archetype': arch_name,
                    'combo_type': 'dragon',
                }
                agent_count += 1
                if len(deck) == 60:
                    legal_count += 1
                else:
                    illegal_agents.append((agent_id, len(deck)))
            except Exception as e:
                print(f"  ERROR {agent_id}: {e}")

    # === Save Results ===
    print(f"\n{'='*60}")
    print(f"Total agents built: {agent_count}")
    print(f"Legal 60-card decks: {legal_count}")
    print(f"Illegal decks: {len(illegal_agents)}")
    if illegal_agents:
        for aid, size in illegal_agents[:10]:
            print(f"  {aid}: {size} cards")

    # Save agent registry
    registry_path = ROOT / "ptcg-system" / "agents_registry.json"
    with open(registry_path, 'w') as f:
        json.dump(all_agents, f, indent=2, default=str)
    print(f"\nAgent registry saved to {registry_path}")

    # Save default deck.csv (first legal agent)
    for aid, adata in all_agents.items():
        if len(adata['deck']) == 60:
            deck_path = ROOT / "deck.csv"
            with open(deck_path, 'w') as f:
                for cid in adata['deck']:
                    f.write(f"{cid}\n")
            print(f"Default deck.csv saved ({aid}, {len(adata['deck'])} cards)")
            break

    # Save agent catalog (lightweight summary)
    catalog = {}
    for aid, adata in all_agents.items():
        catalog[aid] = {
            'combo_type': adata['combo_type'],
            'energy_types': adata['energy_types'],
            'archetype': adata['archetype'],
            'deck_size': len(adata['deck']),
            'species': [s['name'] for s in adata['info'].get('pokemon', {}).get('species', [])],
        }
    catalog_path = ROOT / "ptcg-system" / "agent_catalog.json"
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    print(f"Agent catalog saved to {catalog_path}")

    return all_agents


if __name__ == '__main__':
    build_all_agents()
