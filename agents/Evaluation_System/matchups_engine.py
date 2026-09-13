"""
agents/Evaluation_System/matchups_engine.py
===========================================
Comprehensive Meta-Matchup Simulation Engine & Kaggle Readiness Validator.

Capabilities:
1. Evaluates candidate agent(s) against a gauntlet of elemental meta archetypes:
   - Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Dragon, Team Rocket, Dual-Type
2. High-throughput parallel simulation with Wilson 95% Confidence Intervals
3. Continuous closed-loop GPU Policy-Gradient + MCTS HiveMind NN training on match replays
4. Calculates Meta Win Rate, P1 & P2 win rates, draw rates, average turns, and speed
5. Generates Rich TUI dashboard and Kaggle Submission Readiness recommendation
"""
import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None

from agents.Resource_Management.simulation_runner import get_simulation_runner
from agents.Learning_System import get_replay_buffer
from agents.NN import get_hive_mind_net
from agents.ML import CardValueModel, CardsMatrix

META_BENCHMARK_AGENTS = [
    ("Grass (Stage 2 ex)", "S_GRA_stage_2_ex"),
    ("Fire (Stage 2 ex)", "S_FIR_stage_2_ex"),
    ("Water (Mega ex)", "S_WAT_mega_stage_1_ex"),
    ("Lightning (Stage 2 ex)", "S_LIG_stage_2_ex"),
    ("Psychic (Stage 2 ex)", "S_PSY_stage_2_ex"),
    ("Fighting (Prize Rush)", "S_FIG_prize_rush"),
    ("Darkness (Stage 2 ex)", "S_DAR_stage_2_ex"),
    ("Metal (Stage 2 ex)", "S_MET_stage_2_ex"),
    ("Dragon (Counter)", "DR_damage_counter"),
    ("Team Rocket (Stall)", "TR_stall"),
    ("Dual Psy/Dar (Mega)", "D_PSY+DAR_mega_stage_2_ex"),
    ("Fire (Aggro)", "S_FIR_aggro"),
    ("Fighting (Stage 2 ex)", "S_FIG_stage_2_ex"),
    ("Water (Stage 2 ex)", "S_WAT_stage_2_ex"),
]


class MatchupsSimulationEngine:
    """Executes multi-archetype meta gauntlets and validates Kaggle submission readiness."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (ROOT / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runner = get_simulation_runner(self.data_dir)
        self.buffer = get_replay_buffer()
        self.hive_mind = get_hive_mind_net()
        self.card_model = CardValueModel()
        self.pmi_matrix = CardsMatrix()

    def run_matchups_gauntlet(
        self,
        candidate_agents: List[str],
        games_per_matchup: int = 25,
        train_epochs: int = 2,
        resolve_fn: Optional[Any] = None,
        get_deck_fn: Optional[Any] = None,
        opponents: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Run candidate agent(s) across all elemental archetypes or custom opponents with closed-loop training."""
        from ptcg import _resolve_agent, _get_agent_deck
        _res_fn = resolve_fn or _resolve_agent
        _deck_fn = get_deck_fn or _get_agent_deck
        active_opponents = opponents or META_BENCHMARK_AGENTS

        overall_reports = {}

        for cand_raw in candidate_agents:
            cand_id = _res_fn(cand_raw)
            if not cand_id:
                print(f"[ERROR] Candidate agent '{cand_raw}' could not be resolved.")
                continue

            cand_deck = _deck_fn(cand_id)
            if not cand_deck or len(cand_deck) != 60:
                print(f"[ERROR] Invalid 60-card deck for candidate '{cand_id}'.")
                continue

            if HAS_RICH:
                console.print(Panel(
                    f"Candidate Agent: [bold green]{cand_id}[/bold green]\n"
                    f"Meta Gauntlet: [bold cyan]{len(active_opponents)} Competitor Opponents[/bold cyan]\n"
                    f"Games per Matchup: [bold yellow]{games_per_matchup}[/bold yellow] (Total Matches: {len(active_opponents)*games_per_matchup})\n"
                    f"Coupled Training: [bold magenta]GPU Policy-Gradient + MCTS HiveMind NN ({train_epochs} epochs/matchup)[/bold magenta]",
                    title="[bold white]PTCG COMPREHENSIVE MATCHUPS-SIMULATION GAUNTLET[/bold white]",
                    border_style="magenta"
                ))
            else:
                print(f"\n=== MATCHUPS-SIMULATION GAUNTLET: {cand_id} ===")
                print(f"Testing against {len(active_opponents)} opponents ({games_per_matchup} games each)")

            matchup_results = []
            total_cand_wins = 0
            total_cand_losses = 0
            total_draws = 0
            total_games_played = 0
            t_gauntlet_start = time.perf_counter()

            for opp_label, opp_id_raw in active_opponents:
                opp_id = _res_fn(opp_id_raw)
                opp_deck = _deck_fn(opp_id)
                if not opp_deck or len(opp_deck) != 60:
                    continue

                if HAS_RICH:
                    with Progress(SpinnerColumn(), TextColumn(f"[bold cyan]Matchup: {cand_id} vs {opp_label}..."), BarColumn(), TimeElapsedColumn()) as prog:
                        task = prog.add_task("sim", total=None)
                        sim_res = self.runner.run_simulations(cand_deck, opp_deck, total_games=games_per_matchup, collect_replays=True)
                        prog.update(task, completed=True)
                else:
                    print(f"  Simulating {games_per_matchup} games vs {opp_label} ({opp_id})...")
                    sim_res = self.runner.run_simulations(cand_deck, opp_deck, total_games=games_per_matchup, collect_replays=True)

                w1, w2, dr = sim_res['wins_p1'], sim_res['wins_p2'], sim_res['draws']
                wr1 = sim_res['win_rate_p1']
                wr2 = sim_res['win_rate_p2']
                ci = sim_res.get('wilson_ci_95_p1', (0.0, 0.0))

                total_cand_wins += w1
                total_cand_losses += w2
                total_draws += dr
                total_games_played += sim_res['completed_games']

                # Train NN & ML models on newly accumulated experience
                nn_res = self.hive_mind.train_on_replays(epochs=train_epochs)
                ml_res = self.card_model.train(self.buffer)
                self.pmi_matrix.compute_from_replays(self.buffer)

                # Record simulation telemetry
                try:
                    from ptcg import _record_batch_results
                    _record_batch_results(cand_id, opp_id, w1, w2, dr, sim_type="matchups")
                except Exception:
                    pass

                matchup_record = {
                    'opponent_label': opp_label,
                    'opponent_id': opp_id,
                    'p1_wins': w1,
                    'p2_wins': w2,
                    'draws': dr,
                    'p1_win_rate': wr1,
                    'p2_win_rate': wr2,
                    'ci_95': ci,
                    'avg_turns': sim_res.get('avg_turns', 0.0),
                    'games_per_sec': sim_res.get('games_per_second', 0.0),
                    'nn_loss': nn_res.get('avg_loss', 0.0),
                    'ml_mae': ml_res.get('mae', 0.0)
                }
                matchup_results.append(matchup_record)

                if not HAS_RICH:
                    print(f"    Result: {w1}W ({cand_id}) - {w2}L ({opp_label}) - {dr}D | Candidate WR: {wr1*100:.1f}% | CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

            dur_total = round(time.perf_counter() - t_gauntlet_start, 2)
            meta_wr = total_cand_wins / max(1, total_games_played)

            # Display Comprehensive Matchups Table
            if HAS_RICH:
                table = Table(
                    title=f"Meta Gauntlet Telemetry Report: {cand_id} ({dur_total}s)",
                    border_style="cyan",
                    show_footer=True
                )
                table.add_column("Matchup Archetype (P2)", style="bold white", footer="OVERALL META TOTAL")
                table.add_column(f"P1 Wins ({cand_id})", justify="right", style="green", footer=str(total_cand_wins))
                table.add_column("P2 Wins (Archetype)", justify="right", style="red", footer=str(total_cand_losses))
                table.add_column("Draws", justify="right", style="white", footer=str(total_draws))
                table.add_column("Candidate (P1) WR", justify="right", style="bold green", footer=f"{meta_wr*100:.1f}%")
                table.add_column("Archetype (P2) WR", justify="right", style="bold red", footer=f"{(total_cand_losses/max(1,total_games_played))*100:.1f}%")
                table.add_column("Wilson 95% CI", justify="center", style="yellow")
                table.add_column("Avg Turns", justify="right", style="cyan")
                table.add_column("Speed (g/s)", justify="right")
                table.add_column("NN Loss", justify="right", style="magenta")

                for m in matchup_results:
                    ci_s = f"[{m['ci_95'][0]:.2f}, {m['ci_95'][1]:.2f}]"
                    table.add_row(
                        m['opponent_label'],
                        str(m['p1_wins']),
                        str(m['p2_wins']),
                        str(m['draws']),
                        f"{m['p1_win_rate']*100:.1f}%",
                        f"{m['p2_win_rate']*100:.1f}%",
                        ci_s,
                        f"{m['avg_turns']:.1f}",
                        f"{m['games_per_sec']:.1f}",
                        f"{m['nn_loss']:.4f}"
                    )
                console.print(table)

                # Kaggle Readiness Verdict Banner
                readiness_color = "green" if meta_wr >= 0.70 else ("yellow" if meta_wr >= 0.50 else "red")
                verdict = "ELITE - HIGHLY RECOMMENDED FOR KAGGLE SUBMISSION" if meta_wr >= 0.75 else (
                    "COMPETITIVE - SUITABLE FOR KAGGLE SUBMISSION" if meta_wr >= 0.60 else "NEEDS GA OPTIMIZATION BEFORE SUBMISSION"
                )

                console.print(Panel(
                    f"Agent: [bold white]{cand_id}[/bold white]\n"
                    f"Total Matches Simulated: [bold]{total_games_played}[/bold] ({dur_total}s)\n"
                    f"Candidate Record: [bold green]{total_cand_wins} Wins[/bold green] - [bold red]{total_cand_losses} Losses[/bold red] - [bold white]{total_draws} Draws[/bold white]\n"
                    f"Overall Meta Win Rate: [bold {readiness_color}]{meta_wr*100:.1f}%[/bold {readiness_color}]\n"
                    f"Status: [bold {readiness_color}]{verdict}[/bold {readiness_color}]",
                    title="[bold white]KAGGLE SUBMISSION READINESS EVALUATION[/bold white]",
                    border_style=readiness_color
                ))

            # Save detailed report
            out_file = self.data_dir / f"matchups_report_{cand_id}.json"
            rep_data = {
                'candidate_agent': cand_id,
                'total_games': total_games_played,
                'candidate_wins': total_cand_wins,
                'candidate_losses': total_cand_losses,
                'draws': total_draws,
                'meta_win_rate': round(meta_wr, 4),
                'duration_sec': dur_total,
                'matchups': matchup_results
            }
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(rep_data, f, indent=2)

            # Also persist an immutable versioned record to data/simulation_runs/
            try:
                run_dir = self.data_dir / "simulation_runs"
                run_dir.mkdir(parents=True, exist_ok=True)
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                ver_file = run_dir / f"matchups_{cand_id}_{timestamp_str}_{total_games_played}g.json"
                with open(ver_file, 'w', encoding='utf-8') as f:
                    json.dump(rep_data, f, indent=2)
            except Exception:
                pass

            # Automated Champion Auto-Checkpoint & Hot-Swap
            try:
                from agents.Evaluation_System.champion_tracker import get_champion_tracker
                tracker = get_champion_tracker()
                promoted = tracker.check_and_promote(cand_id, meta_wr, total_games_played, context="matchups-simulation", deck=cand_deck)
                if promoted and HAS_RICH:
                    console.print(Panel(
                        f"[bold gold1]🏆 NEW ALL-TIME CHAMPION PROMOTED:[/bold gold1] [bold green]{cand_id}[/bold green] ([bold cyan]{meta_wr*100:.1f}%[/bold cyan] Meta WR)\n"
                        f"[bold white]Automated Action:[/bold white] Saved checkpoint to ptcg-system/checkpoints/ and hot-swapped [bold green]submission.tar.gz[/bold green]!",
                        title="[bold gold1]AUTONOMOUS CHAMPION AUTO-HOTSWAP TRIGGERED[/bold gold1]",
                        border_style="gold1"
                    ))
            except Exception:
                pass

            print(f"[SUCCESS] Matchups report saved -> {out_file}\n")
            overall_reports[cand_id] = rep_data

        return overall_reports


_matchups_engine: Optional[MatchupsSimulationEngine] = None

def get_matchups_engine() -> MatchupsSimulationEngine:
    global _matchups_engine
    if _matchups_engine is None:
        _matchups_engine = MatchupsSimulationEngine()
    return _matchups_engine
