with open("battle_turn_data.html", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

print("Dynamic 32+ to 64+ Ply Scale:", "Dynamic 32+ to 64+ Ply Scale" in text)
print("Dynamic 1k–3.5k+ Rollouts/turn:", "Dynamic 1k–3.5k+ Rollouts/turn" in text)
print("Static 0-20 Ply present:", ("0–20 Ply" in text or "0-20 Ply" in text))
print("Static 100-450 present:", "100–450/turn" in text)
