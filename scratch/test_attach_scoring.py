import json
from simulation.Decision_Engine import MasterAgent, _get_cards, _get_attacks

with open('vis.json', 'r', encoding='utf-8') as f:
    vis = json.load(f)

# Find an attach frame
for i, frame in enumerate(vis):
    obs = frame.get('obs')
    if isinstance(obs, str):
        try: obs = json.loads(obs)
        except: continue
    if not isinstance(obs, dict): continue
    sel = obs.get('select', {})
    if sel.get('context') == 0:
        opts = sel.get('option', [])
        attach_opts = [j for j, o in enumerate(opts) if o.get('type') == 8]
        if attach_opts:
            agent = MasterAgent()
            cards = _get_cards()
            attacks = _get_attacks()
            obs_data = {'turn': obs.get('current', {}).get('turn', 1)}
            posture, weights = agent.ooda.orient(obs_data, 'balanced', {})
            
            print(f"\n--- Testing Attachment at Frame {i} (Turn {obs_data['turn']}) ---")
            cur = obs.get('current', {})
            p0 = cur.get('players', [{}])[0]
            act = p0.get('active', [{}])[0] if p0.get('active') else {}
            bench = p0.get('bench', [])
            print(f"Active: ID={act.get('id')} HP={act.get('hp')} Energies={len(act.get('energies', []))}")
            print(f"Bench Count: {len(bench)}")
            for b_i, b in enumerate(bench):
                print(f"  Bench[{b_i}]: ID={b.get('id')} HP={b.get('hp')} Energies={len(b.get('energies', []))}")
            
            for opt_idx in attach_opts:
                opt = opts[opt_idx]
                area = opt.get('inPlayArea')
                in_play_idx = opt.get('inPlayIndex')
                area_name = "ACTIVE" if area == 4 else f"BENCH[{in_play_idx}]"
                # Let's see what score_attach produces
                # We can call score_attach internally
                # or inspect _energy_priority_attach_ooda
            
            res = agent._energy_priority_attach_ooda(obs, opts, attach_opts, cards, attacks, obs_data, posture, weights)
            print(f"Selected option index: {res} -> Opt: {opts[res[0]]}")
            break
