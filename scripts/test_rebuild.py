import sys
import json
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.csv_data import get_csv_index
from agents.csv_deck_builder import CsvDeckBuilder, is_pokemon_attack_compatible

idx = get_csv_index(ROOT / "data")
builder = CsvDeckBuilder(idx)

reg_path = ROOT / "ptcg-system" / "agents_registry.json"
cat_path = ROOT / "ptcg-system" / "agent_catalog.json"

with open(reg_path, encoding='utf-8') as f:
    reg = json.load(f)
with open(cat_path, encoding='utf-8') as f:
    cat = json.load(f)

print(f"Starting rebuild of {len(reg)} agents...")
rebuilt_reg = {}
rebuilt_cat = {}
bad_deck_count = 0

for aid, data in reg.items():
    combo = data.get('combo_type', 'single')
    arch = data.get('archetype', 'balanced')
    etypes = data.get('energy_types', [])
    seed = data.get('seed', 42)
    
    deck, info = builder.build_deck(etypes, arch, combo, seed=seed)
    
    has_prism = (16 in deck)
    deck_etypes = set(etypes)
    for cid in deck:
        c = idx.get_card(cid)
        if c and (c.is_basic_energy or c.is_special_energy) and c.type and c.type != '{A}':
            deck_etypes.add(c.type)
    
    bad_cards = [cid for cid in set(deck) if not is_pokemon_attack_compatible(idx.get_card(cid), list(deck_etypes), has_prism)]
    if bad_cards:
        bad_deck_count += 1
        print(f"WARNING: {aid} still has bad cards: {bad_cards}")
    
    rebuilt_reg[aid] = {
        'deck': deck,
        'combo_type': combo,
        'archetype': arch,
        'energy_types': etypes,
        'deck_size': len(deck),
        'build_info': info
    }
    rebuilt_cat[aid] = {
        'combo_type': combo,
        'archetype': arch,
        'energy_types': etypes,
        'deck_size': len(deck)
    }

print(f"Rebuild completed! Total decks with errors: {bad_deck_count} / {len(rebuilt_reg)}")

# Inspect specific user-mentioned agents
for target in ['S_GRA_balanced', 'D_WAT+LIG_mega_stage_2_ex', 'DR_damage_counter', 'D_PSY+DAR_mega_stage_2_ex']:
    if target in rebuilt_reg:
        d = rebuilt_reg[target]['deck']
        print(f"\n=== {target} (Deck: {len(d)} cards) ===")
        counts = {}
        for c in d: counts[c] = counts.get(c, 0) + 1
        for cid, cnt in counts.items():
            card = idx.get_card(cid)
            if card and card.is_pokemon:
                atks = [f"{a.get('name')}: {a.get('cost')}" for a in card.attacks]
                print(f"  Pokemon x{cnt} #{cid:>4} {card.name:<25} Type: {card.type:<10} Attacks: {atks}")
            elif card and (card.is_basic_energy or card.is_special_energy):
                print(f"  Energy  x{cnt} #{cid:>4} {card.name:<25} Type: {card.type:<10}")

# Save updated files
with open(reg_path, 'w', encoding='utf-8') as f:
    json.dump(rebuilt_reg, f, indent=2)
with open(cat_path, 'w', encoding='utf-8') as f:
    json.dump(rebuilt_cat, f, indent=2)
print("\n[SUCCESS] Saved updated agents_registry.json and agent_catalog.json!")
