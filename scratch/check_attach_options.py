import json

with open('vis.json', 'r', encoding='utf-8') as f:
    vis = json.load(f)

for i, frame in enumerate(vis):
    obs = frame.get('obs', {})
    sel = obs.get('select', {}) if isinstance(obs, dict) else {}
    if sel.get('context') == 0:
        opts = sel.get('option', [])
        attach_opts = [(j, o) for j, o in enumerate(opts) if o.get('type') == 8]
        if attach_opts:
            chosen = frame.get('action', [[]])[0]
            turn = obs.get('current', {}).get('turn')
            p_idx = sel.get('playerIndex')
            print(f"Frame {i} (Turn {turn} P{p_idx}):")
            for idx, opt in attach_opts:
                print(f"   Opt {idx}: {opt}")
            print(f"   Chosen: {chosen}")
