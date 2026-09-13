"""
agents/Evaluation_System/matchups_matrix.py
===========================================
Cross-Archetype 14x14 Matchup Matrix & Heatmap Dashboard Generator.

Simulates and visualizes empirical win rates across all 14 elemental and tactical archetypes,
identifying elemental advantages, hard counters, and meta equilibrium states.
"""
import os
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SYSTEM_DIR = ROOT_DIR / "ptcg-system"
MATRIX_HTML_FILE = SYSTEM_DIR / "matchups_matrix.html"

ARCHETYPE_CHAMPIONS = [
    "S_GRA_stage_2_ex",
    "S_FIR_aggro",
    "S_WAT_stage_2_ex",
    "S_LIG_stage_2_ex",
    "S_PSY_stage_2_ex",
    "S_FIG_stage_2_ex",
    "S_DAR_stage_2_ex",
    "S_MET_stage_2_ex",
    "D_GRA+PSY_balanced",
    "D_FIR+WAT_aggro",
    "D_PSY+DAR_mega_stage_2_ex",
    "T_GRA+FIR+WAT_balanced",
    "TR_stage_2_ex",
    "DR_damage_counter"
]


class MatchupsMatrixEngine:
    """Computes and renders full cross-archetype win rate matrices and HTML heatmaps."""

    def __init__(self, archetypes: Optional[List[str]] = None):
        self.archetypes = archetypes or ARCHETYPE_CHAMPIONS
        self.matrix_cache_file = SYSTEM_DIR / "matchups_matrix_cache.json"

    def compute_matrix(self, games_per_pair: int = 10, use_cache: bool = True) -> Dict[str, Any]:
        from ptcg import _get_agent_deck, _record_batch_results
        from agents.Resource_Management.simulation_runner import get_simulation_runner

        runner = get_simulation_runner()
        n = len(self.archetypes)
        results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Load cache if available
        if use_cache and self.matrix_cache_file.exists():
            try:
                with open(self.matrix_cache_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except Exception:
                results = {}

        total_pairs = n * (n - 1) // 2
        completed_pairs = 0

        for i in range(n):
            a1 = self.archetypes[i]
            if a1 not in results:
                results[a1] = {}
            for j in range(n):
                a2 = self.archetypes[j]
                if a2 not in results:
                    results[a2] = {}

                if a1 == a2:
                    results[a1][a2] = {
                        "wins_p1": games_per_pair // 2,
                        "wins_p2": games_per_pair // 2,
                        "draws": 0,
                        "games": games_per_pair,
                        "win_rate": 0.50
                    }
                    continue

                # Check if already computed
                if a2 in results[a1] and results[a1][a2].get("games", 0) >= games_per_pair:
                    continue

                deck1 = _get_agent_deck(a1)
                deck2 = _get_agent_deck(a2)

                if not deck1 or not deck2:
                    continue

                res = runner.run_simulations(
                    deck1, deck2, total_games=games_per_pair, collect_replays=False,
                    agent1_config={'name': a1}, agent2_config={'name': a2}
                )

                w1, w2, dr = res.get('wins_p1', 0), res.get('wins_p2', 0), res.get('draws', 0)
                tot = max(1, res.get('completed_games', games_per_pair))
                wr1 = round(w1 / tot, 4)
                wr2 = round(w2 / tot, 4)

                results[a1][a2] = {
                    "wins_p1": w1, "wins_p2": w2, "draws": dr,
                    "games": tot, "win_rate": wr1
                }
                results[a2][a1] = {
                    "wins_p1": w2, "wins_p2": w1, "draws": dr,
                    "games": tot, "win_rate": wr2
                }

                _record_batch_results(a1, a2, w1, w2, dr, sim_type="matrix")
                completed_pairs += 1

        # Save cache
        try:
            with open(self.matrix_cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass

        return {
            "archetypes": self.archetypes,
            "grid": results,
            "games_per_pair": games_per_pair,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def export_html(self, data: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        out = output_path or MATRIX_HTML_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        archs = data["archetypes"]
        grid = data["grid"]

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KYON Sovereign — Cross-Archetype Matchup Matrix</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0b0f19; color: #f3f4f6; margin: 0; padding: 24px; }}
  h1 {{ color: #38bdf8; text-align: center; margin-bottom: 8px; }}
  .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 24px; font-size: 14px; }}
  .container {{ max-width: 1400px; margin: 0 auto; overflow-x: auto; background: #131d31; border-radius: 12px; padding: 20px; box-shadow: 0 8px 30px rgba(0,0,0,0.5); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ border: 1px solid #1e293b; padding: 8px 6px; text-align: center; }}
  th {{ background: #1e293b; color: #38bdf8; position: sticky; top: 0; }}
  th.row-header {{ text-align: left; background: #1a243a; color: #cbd5e1; font-weight: bold; width: 180px; }}
  .cell-elite {{ background: #065f46; color: #a7f3d0; font-weight: bold; }}
  .cell-good {{ background: #047857; color: #d1fae5; }}
  .cell-even {{ background: #334155; color: #f1f5f9; }}
  .cell-weak {{ background: #881337; color: #fecdd3; }}
  .cell-bad {{ background: #4c0519; color: #fda4af; font-weight: bold; }}
  .legend {{ display: flex; justify-content: center; gap: 16px; margin-top: 20px; font-size: 13px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .color-box {{ width: 16px; height: 16px; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>⚡ KYON 14×14 Cross-Archetype Matchup Matrix</h1>
  <div class="subtitle">Empirical Win Rates Across Tournament Archetypes ({data.get('games_per_pair', 10)} games/cell) | Generated: {data.get('timestamp')}</div>
  <div class="container">
    <table>
      <thead>
        <tr>
          <th>Candidate vs Opponent</th>
"""
        for a in archs:
            short_name = a.replace('S_', '').replace('D_', '').replace('T_', '')
            html_content += f"          <th>{short_name}</th>\n"

        html_content += """        </tr>
      </thead>
      <tbody>
"""
        for a1 in archs:
            html_content += f"        <tr>\n          <td class='row-header'>{a1}</td>\n"
            for a2 in archs:
                cell = grid.get(a1, {}).get(a2, {})
                wr = cell.get("win_rate", 0.50)
                wr_pct = wr * 100
                if wr >= 0.75:
                    c_class = "cell-elite"
                elif wr >= 0.60:
                    c_class = "cell-good"
                elif wr >= 0.45:
                    c_class = "cell-even"
                elif wr >= 0.30:
                    c_class = "cell-weak"
                else:
                    c_class = "cell-bad"

                html_content += f"          <td class='{c_class}' title='{a1} vs {a2}: {cell.get('wins_p1',0)}W - {cell.get('wins_p2',0)}L'>{wr_pct:.0f}%</td>\n"
            html_content += "        </tr>\n"

        html_content += """      </tbody>
    </table>
    <div class="legend">
      <div class="legend-item"><div class="color-box" style="background:#065f46"></div> ≥75% Dominant Advantage</div>
      <div class="legend-item"><div class="color-box" style="background:#047857"></div> 60–74% Favorable Matchup</div>
      <div class="legend-item"><div class="color-box" style="background:#334155"></div> 45–59% Even / Skill-Based</div>
      <div class="legend-item"><div class="color-box" style="background:#881337"></div> 30–44% Unfavorable Matchup</div>
      <div class="legend-item"><div class="color-box" style="background:#4c0519"></div> <30% Severe Weakness</div>
    </div>
  </div>
</body>
</html>
"""
        with open(out, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return out


_matchups_matrix_engine: Optional[MatchupsMatrixEngine] = None


def get_matchups_matrix_engine() -> MatchupsMatrixEngine:
    global _matchups_matrix_engine
    if _matchups_matrix_engine is None:
        _matchups_matrix_engine = MatchupsMatrixEngine()
    return _matchups_matrix_engine
