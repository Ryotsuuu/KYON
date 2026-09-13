import json

with open('csv-data/cards.json', 'r', encoding='utf-8') as f:
    cards = json.load(f)

for c in cards:
    name = c.get('name', '')
    if 'mewtwo' in name.lower() or 'rocket' in name.lower():
        cid = c.get('card_id') or c.get('id')
        attacks = c.get('attacks', [])
        for a in attacks:
            txt = a.get('text', '')
            if 'rocket' in txt.lower() or 'in play' in txt.lower() or '4' in txt:
                print(f"Match: Card {cid}: {name.encode('ascii', 'ignore').decode()}")
                print(f"  Attack: {str(a.get('name')).encode('ascii', 'ignore').decode()} | Dmg: {a.get('damage')}")
                print(f"  Text: {txt.encode('ascii', 'ignore').decode()}")
