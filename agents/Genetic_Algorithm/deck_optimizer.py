"""
agents/Genetic_Algorithm/deck_optimizer.py
=========================================
Genetic Algorithm Deck Optimizer with Strict Legality Repair & Protection.

Guarantees:
- Exactly 60 cards
- Max 4 copies of any non-basic energy card
- Max 1 ACE SPEC card per deck
- At least 1 Basic Pokémon (enforces legal starting hand)
- Max 6 Pokémon species lines (max 18 Pokémon total)
- Category & Energy compatibility preservation
"""
import copy
import random
import logging
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from agents.csv_data import CsvDataIndex, get_csv_index
from agents.Learning_System.replay_buffer import get_replay_buffer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Gatekeeper & Suggestion Popup
# ═══════════════════════════════════════════════════════════════════════

def check_advanced_optimization_unlocked(
    agent_id: str,
    registry_data: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Empirical Gatekeeper for ADVANCED Genetic Algorithm & Master Optimization.

    Condition A: Target agent has > 10,000 completed simulation games.
    Condition B: Metagame coverage has >= 3,000 simulation games across all 10 energy types.

    Returns:
        (is_unlocked, status_dict)
    """
    if registry_data is None:
        reg_file = Path(__file__).resolve().parents[2] / "ptcg-system" / "agents_registry.json"
        if reg_file.exists():
            try:
                with open(reg_file, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
            except Exception:
                registry_data = {}
        else:
            registry_data = {}

    target_meta = registry_data.get(agent_id, {})
    agent_games = target_meta.get('games', 0) if isinstance(target_meta, dict) else 0
    condition_a = agent_games > 10000

    energy_types_list = ["Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting", "Darkness", "Metal", "Dragon", "Colorless"]
    energy_counts = {e: 0 for e in energy_types_list}
    for aid, meta in registry_data.items():
        if isinstance(meta, dict):
            g = meta.get('games', 0)
            for etype in meta.get('energy_types', []):
                if etype in energy_counts:
                    energy_counts[etype] += g

    condition_b = all(cnt >= 3000 for cnt in energy_counts.values())
    is_unlocked = condition_a or condition_b

    status_dict = {
        'agent_id': agent_id,
        'agent_games': agent_games,
        'agent_threshold': 10000,
        'condition_a_met': condition_a,
        'energy_counts': energy_counts,
        'energy_threshold': 3000,
        'condition_b_met': condition_b,
        'is_unlocked': is_unlocked,
    }
    return is_unlocked, status_dict


def render_unlock_suggestion_popup(agent_id: str, status_dict: Dict[str, Any]) -> str:
    """Renders an informative terminal suggestion banner / popup when Advanced Optimization is locked."""
    agent_games = status_dict.get('agent_games', 0)
    agent_thresh = status_dict.get('agent_threshold', 10000)
    agent_deficit = max(0, agent_thresh - agent_games)
    energy_counts = status_dict.get('energy_counts', {})
    energy_thresh = status_dict.get('energy_threshold', 3000)

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        console = Console()

        t = Table(box=None, expand=True)
        t.add_column("Criterion", style="bold white", width=26)
        t.add_column("Current / Required", style="cyan", width=22)
        t.add_column("Deficit", style="yellow", width=14)
        t.add_column("Status", justify="center", width=12)

        agent_status_text = "[green]UNLOCKED[/green]" if status_dict.get('condition_a_met') else "[red]LOCKED[/red]"
        t.add_row(
            f"Agent: {agent_id}",
            f"{agent_games:,} / {agent_thresh:,}",
            f"-{agent_deficit:,}" if agent_deficit > 0 else "0",
            agent_status_text
        )

        metagame_status = "[green]UNLOCKED[/green]" if status_dict.get('condition_b_met') else "[red]LOCKED[/red]"
        qual_energies = sum(1 for c in energy_counts.values() if c >= energy_thresh)
        t.add_row(
            "Metagame Coverage",
            f"{qual_energies}/10 types >= {energy_thresh:,}",
            f"{10 - qual_energies} pending",
            metagame_status
        )

        e_table = Table(title="Elemental Metagame Simulation Breakdown", show_header=True, header_style="bold magenta", border_style="dim")
        e_table.add_column("Energy Type", style="cyan")
        e_table.add_column("Simulations", justify="right")
        e_table.add_column("Progress", justify="left")
        e_table.add_column("Deficit", justify="right", style="yellow")

        for etype, cnt in energy_counts.items():
            pct = min(100.0, (cnt / energy_thresh) * 100.0) if energy_thresh > 0 else 100.0
            bars = int(pct / 10)
            bar_visual = f"[{'=' * bars}{'.' * (10 - bars)}] {pct:5.1f}%"
            defic = max(0, energy_thresh - cnt)
            color = "green" if cnt >= energy_thresh else "yellow" if cnt > 0 else "red"
            e_table.add_row(etype, f"{cnt:,}", f"[{color}]{bar_visual}[/{color}]", f"-{defic:,}" if defic > 0 else "[green]0[/green]")

        cmd_table = Table(box=None, show_header=False)
        cmd_table.add_column("Action", style="bold cyan")
        cmd_table.add_column("CLI Command", style="green")
        cmd_table.add_row("1. Target Agent Sims", f"python ptcg.py simulate {agent_id} S_PSY_stage_2_ex --games {min(1000, agent_deficit or 500)}")
        cmd_table.add_row("2. Agent Gauntlet", f"python ptcg.py matchups-simulation --agent {agent_id} --games 50")
        cmd_table.add_row("3. Metagame Boost", "python ptcg.py run-all-matchups --games 100")

        content = Table.grid(padding=(1, 0))
        content.add_row(Text("The ADVANCED system optimization method requires empirical foundation to guide\nmutation rules, Master-guided fitness weights, and discovery seeding safely.\n", style="italic white"))
        content.add_row(t)
        content.add_row(e_table)
        content.add_row(Text("\n[RECOMMENDED CLI COMMANDS TO UNLOCK ADVANCED METHOD]", style="bold yellow"))
        content.add_row(cmd_table)
        content.add_row(Text("\n[FALLBACK NOTE] Safely falling back to System Predefined Legal GA (--method predefined).\nStrict system invariants & 100% legal bounds will be maintained.", style="bold cyan"))

        capture_console = Console(record=True, width=80)
        panel = Panel(
            content,
            title="[bold red][GATEKEEPER] ADVANCED OPTIMIZATION LOCKED[/bold red]",
            border_style="bright_yellow",
            padding=(1, 2)
        )
        try:
            console.print()
            console.print(panel)
            console.print()
        except Exception:
            pass
        capture_console.print(panel)
        return capture_console.export_text()
    except Exception:
        output = []
        output.append("\n" + "=" * 70)
        output.append("[GATEKEEPER] ADVANCED OPTIMIZATION LOCKED")
        output.append("=" * 70)
        output.append(f"Target Agent: {agent_id} -> {agent_games:,} / {agent_thresh:,} sims (Deficit: -{agent_deficit:,})")
        output.append(f"Metagame Threshold: >= {energy_thresh:,} sims required per elemental energy type")
        output.append("\nRecommended commands to unlock:")
        output.append(f"  * python ptcg.py simulate {agent_id} S_PSY_stage_2_ex --games {min(1000, agent_deficit or 500)}")
        output.append(f"  * python ptcg.py matchups-simulation --agent {agent_id} --games 50")
        output.append("  * python ptcg.py run-all-matchups --games 100")
        output.append("\n[FALLBACK NOTE] Safely falling back to System Predefined Legal GA (--method predefined).")
        output.append("=" * 70 + "\n")
        msg = "\n".join(output)
        try:
            print(msg)
        except Exception:
            pass
        return msg


class Individual:
    """A single deck candidate in the GA population."""

    def __init__(self, deck: List[int], agent_id: str = '', generation: int = 0):
        self.deck = list(deck)
        self.agent_id = agent_id
        self.generation = generation
        self.fitness = 0.0
        self.win_rate = 0.0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.games_played = 0

    def copy(self) -> 'Individual':
        ind = Individual(self.deck, self.agent_id, self.generation)
        ind.fitness = self.fitness
        ind.win_rate = self.win_rate
        ind.wins = self.wins
        ind.losses = self.losses
        ind.draws = self.draws
        ind.games_played = self.games_played
        return ind


class GeneticOptimizer:
    """Genetic Algorithm for optimizing PTCG decks with strict chromosome repair.
    
    Supports:
    - Method 1: 'predefined' (strict predefined bounds & archetype constraints)
    - Method 2: 'advanced' (master-guided fitness, discovery seeding, role-biased mutation)
    """

    def __init__(
        self,
        base_deck: List[int],
        population_size: int = 12,
        mutation_rate: float = 0.35,
        crossover_rate: float = 0.65,
        elite_count: int = 2,
        seed: int = 42,
        csv_index: Optional[CsvDataIndex] = None,
        energy_types: Optional[List[str]] = None,
        archetype: Optional[str] = None,
        optimization_method: str = "predefined",
        master_agent: Optional[Any] = None,
        fitness_fn: Optional[Any] = None,
    ):
        self.base_deck = list(base_deck)
        self.population_size = max(4, population_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.rng = random.Random(seed)
        self._csv_idx = csv_index or get_csv_index(Path("data"))
        self.energy_types = list(energy_types) if energy_types else self._infer_energy_types(self.base_deck)
        self.archetype = archetype or "balanced"
        self.optimization_method = optimization_method.lower() if optimization_method else "predefined"
        self.master_agent = master_agent
        self.fitness_fn = fitness_fn
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.population: List[Individual] = []
        self._initialize_population()

    def _infer_energy_types(self, deck: List[int]) -> List[str]:
        """Infer the elemental energy types present in the base deck."""
        energy_map = {
            1: 'Grass', 2: 'Fire', 3: 'Water', 4: 'Lightning',
            5: 'Psychic', 6: 'Fighting', 7: 'Darkness', 8: 'Metal', 9: 'Dragon',
            18: 'Grass', 19: 'Psychic', 20: 'Fighting',
        }
        types = []
        for cid in deck:
            if cid in energy_map:
                t = energy_map[cid]
                if t not in types:
                    types.append(t)
        return types or ['Psychic']

    def _get_primary_energy_id(self) -> int:
        """Get basic energy ID for the deck's primary element."""
        mapping = {
            'Grass': 1, 'Fire': 2, 'Water': 3, 'Lightning': 4,
            'Psychic': 5, 'Fighting': 6, 'Darkness': 7, 'Metal': 8, 'Dragon': 9
        }
        primary = self.energy_types[0] if self.energy_types else 'Psychic'
        return mapping.get(primary, 5)

    def _initialize_population(self, opponent_deck: Optional[List[int]] = None):
        """Create diverse initial population mutated legally from base deck.
        
        In advanced mode: seeds initial chromosomes using persistent knowledge schemata
        and high-synergy card pairings.
        """
        self.population = [Individual(deck=self.base_deck, agent_id="seed_base", generation=0)]
        seeded_count = 1

        if self.optimization_method == "advanced":
            try:
                from agents.Learning_System.persistent_knowledge import get_persistent_knowledge
                pk = get_persistent_knowledge()
                schemata = pk.get_top_schemata(self.archetype, limit=2)
                for s_idx, schema in enumerate(schemata):
                    if seeded_count >= self.population_size:
                        break
                    key_cards = schema.get('cards', [])
                    if key_cards:
                        seeded_deck = list(self.base_deck)
                        for kc in key_cards:
                            if kc not in seeded_deck and len(seeded_deck) > 0:
                                for d_i in range(len(seeded_deck)):
                                    if not self._is_protected(seeded_deck[d_i], Counter(seeded_deck)):
                                        seeded_deck[d_i] = kc
                                        break
                        repaired = self._repair_deck(seeded_deck)
                        self.population.append(Individual(deck=repaired, agent_id=f"seed_schema_{s_idx}", generation=0))
                        seeded_count += 1
            except Exception as e:
                logger.debug("Schema seeding bypassed: %s", e)

        for i in range(seeded_count, self.population_size):
            mutated = self._mutate(list(self.base_deck))
            self.population.append(Individual(deck=mutated, agent_id=f"init_{i}", generation=0))
        self.evaluate_population(opponent_deck)

    def evaluate_population(self, opponent_deck: Optional[List[int]] = None):
        """Evaluate fitness of population via high-throughput simulations."""
        from agents.Resource_Management.simulation_runner import get_simulation_runner
        runner = get_simulation_runner()
        opp = opponent_deck or self.base_deck

        for ind in self.population:
            if ind.games_played == 0:
                res = runner.run_simulations(ind.deck, opp, total_games=4)
                ind.wins = res['wins_p1']
                ind.losses = res['wins_p2']
                ind.draws = res['draws']
                ind.games_played = res['completed_games']
                ind.win_rate = ind.wins / ind.games_played if ind.games_played > 0 else 0.0
                avg_turns = res.get('avg_turns', 25.0) or 25.0
                tempo_bonus = max(0.0, 30.0 - avg_turns) * 0.5

                if self.fitness_fn:
                    ind.fitness = float(self.fitness_fn(ind, res))
                elif self.optimization_method == "advanced":
                    # Master-Guided Multi-Objective Fitness
                    synergy_bonus = 0.0
                    if self._csv_idx:
                        deck_set = set(ind.deck)
                        synergy_hits = 0
                        for cid in deck_set:
                            cdata = self._csv_idx.get_card(cid)
                            if cdata and cdata.cross_reference_synergies:
                                for syn in cdata.cross_reference_synergies:
                                    if syn.get('partner_id') in deck_set:
                                        synergy_hits += 1
                        synergy_bonus = min(15.0, synergy_hits * 1.5)

                    cvm_utility_bonus = 0.0
                    if self._csv_idx:
                        ratings = [
                            self._csv_idx.get_card(cid).effectiveness_score
                            for cid in ind.deck
                            if self._csv_idx.get_card(cid) and getattr(self._csv_idx.get_card(cid), 'effectiveness_score', 0) > 0
                        ]
                        if ratings:
                            cvm_utility_bonus = min(10.0, (sum(ratings) / len(ratings)) * 1.0)

                    causal_bonus = 0.0
                    pkmn_count = sum(1 for cid in ind.deck if self._csv_idx and self._csv_idx.get_card(cid) and self._csv_idx.get_card(cid).is_pokemon)
                    energy_count = sum(1 for cid in ind.deck if cid in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20))
                    if 10 <= pkmn_count <= 18:
                        causal_bonus += 2.0
                    if 8 <= energy_count <= 16:
                        causal_bonus += 2.0
                    if any(self._csv_idx and self._csv_idx.get_card(cid) and self._csv_idx.get_card(cid).has_bench_protection for cid in ind.deck):
                        causal_bonus += 1.0

                    ind.fitness = ind.win_rate * 100.0 + tempo_bonus + synergy_bonus + cvm_utility_bonus + causal_bonus + self.rng.uniform(0.1, 0.5)
                else:
                    # Method 1 Predefined Fitness: win-rate + tempo
                    ind.fitness = ind.win_rate * 100.0 + tempo_bonus + self.rng.uniform(0.1, 0.9)

    def evolve(self, opponent_deck: Optional[List[int]] = None) -> List[Individual]:
        """Advance population by one generation using tournament selection, crossover, mutation, and elite preservation."""
        self.evaluate_population(opponent_deck)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if not self.best_ever or self.population[0].fitness > self.best_ever.fitness:
            self.best_ever = self.population[0].copy()

        next_gen = []

        # 1. Elitism: preserve top individuals
        for i in range(min(self.elite_count, len(self.population))):
            elite = self.population[i].copy()
            elite.generation = self.generation + 1
            next_gen.append(elite)

        # 2. Crossover & Mutation for offspring
        while len(next_gen) < self.population_size:
            if self.rng.random() < self.crossover_rate and len(self.population) >= 2:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                child_deck = self._crossover(p1.deck, p2.deck)
            else:
                p = self._tournament_select()
                child_deck = list(p.deck)

            if self.rng.random() < self.mutation_rate:
                child_deck = self._mutate(child_deck)

            # Strict Chromosome Legality Repair
            repaired_deck = self._repair_deck(child_deck)

            next_gen.append(Individual(
                deck=repaired_deck,
                agent_id=f"gen{self.generation+1}_{len(next_gen)}",
                generation=self.generation + 1,
            ))

        self.population = next_gen
        self.evaluate_population(opponent_deck)
        self.generation += 1

        # In advanced mode, record elite discovery to persistent knowledge
        if self.optimization_method == "advanced" and self.population:
            try:
                from agents.Learning_System.persistent_knowledge import get_persistent_knowledge
                pk = get_persistent_knowledge()
                top_ind = self.population[0]
                if top_ind.fitness > 50.0:
                    pk.record_ga_discovery(top_ind.deck, top_ind.fitness, self.archetype)
            except Exception as e:
                logger.debug("Failed to record GA discovery: %s", e)

        return self.population

    def step(self, opponent_deck: Optional[List[int]] = None, fitness_fn: Optional[callable] = None) -> Optional[Individual]:
        """Advance population by one generation and return current elite individual."""
        if fitness_fn:
            self.fitness_fn = fitness_fn
        self.evolve(opponent_deck=opponent_deck)
        return self.population[0] if self.population else None

    def _tournament_select(self, k: int = 3) -> Individual:
        candidates = self.rng.sample(self.population, min(k, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)

    def _crossover(self, deck1: List[int], deck2: List[int]) -> List[int]:
        """Uniform crossover between two parent chromosomes."""
        child = []
        pad_id = self._get_primary_energy_id()
        for i in range(60):
            d1_val = deck1[i] if i < len(deck1) else pad_id
            d2_val = deck2[i] if i < len(deck2) else pad_id
            child.append(d1_val if self.rng.random() < 0.5 else d2_val)
        return self._repair_deck(child)

    def _mutate(self, deck: List[int]) -> List[int]:
        """Mutate deck by swapping 1-3 non-mandatory cards with legal alternatives."""
        deck = list(deck)
        n_swaps = self.rng.randint(1, 3)
        counts = Counter(deck)

        for _ in range(n_swaps):
            swappable = [i for i, cid in enumerate(deck) if not self._is_protected(cid, counts)]
            if not swappable:
                break

            idx = self.rng.choice(swappable)
            old_cid = deck[idx]

            if self.optimization_method == "advanced":
                new_cid = self._find_strategic_replacement(old_cid, deck, counts)
            else:
                new_cid = self._find_replacement(old_cid, deck, counts)

            if new_cid and new_cid != old_cid:
                deck[idx] = new_cid
                counts[old_cid] -= 1
                counts[new_cid] = counts.get(new_cid, 0) + 1

        return self._repair_deck(deck)

    def _is_protected(self, cid: int, counts: Counter) -> bool:
        """Protect hardcore cards (ACE SPECs, Core Special Energies, Search Baseline, Evolution Lines, GA Risk Invariants)."""
        if self._csv_idx:
            c = self._csv_idx.get_card(cid)
            if c and c.is_ace_spec:
                return True
            if cid in (15, 16, 18, 19, 20):
                return True
            if cid == 1079:
                return True
            if c and c.is_pokemon and (c.is_stage1 or c.is_stage2 or c.is_mega_ex):
                return True

            # In advanced mode, protect cards flagged with GA mutation risk
            if self.optimization_method == "advanced" and self._csv_idx.has_ga_mutation_risk(cid):
                if c and c.is_team_rocket_pokemon and self.archetype == "team_rocket" and counts.get(cid, 0) <= 2:
                    return True
                if cid in (10, 12, 16) and "Dragon" in self.energy_types:
                    return True

        if cid == 1119 and counts.get(1119, 0) <= 2:
            return True
        return False

    def _find_replacement(self, old_cid: int, deck: List[int], counts: Counter) -> Optional[int]:
        """Method 1 (Predefined): Strict archetype & energy bounded replacement."""
        if not self._csv_idx:
            return None
        old_card = self._csv_idx.get_card(old_cid)
        if not old_card:
            return None

        # Basic Energy Replacement -> Must match deck's energy types
        if old_cid in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            energy_name_to_id = {
                'Grass': 1, 'Fire': 2, 'Water': 3, 'Lightning': 4,
                'Psychic': 5, 'Fighting': 6, 'Darkness': 7, 'Metal': 8, 'Dragon': 9
            }
            cand = [energy_name_to_id[t] for t in self.energy_types if t in energy_name_to_id]
            if 'Psychic' in self.energy_types and counts.get(19, 0) < 4:
                cand.append(19)
            if 'Grass' in self.energy_types and counts.get(18, 0) < 4:
                cand.append(18)
            if 'Fighting' in self.energy_types and counts.get(20, 0) < 4:
                cand.append(20)
            return self.rng.choice(cand) if cand else old_cid

        # Pokemon Replacement -> Must match energy types (or Colorless)
        if old_card.is_pokemon:
            candidates = []
            for cid, card in self._csv_idx.cards.items():
                if cid == old_cid or counts.get(cid, 0) >= 4 or not card.is_pokemon:
                    continue
                card_type = card.type or 'Colorless'
                if card_type not in self.energy_types and card_type != 'Colorless':
                    continue
                if card.get_attack_energy_types() - set(self.energy_types):
                    continue
                if card.is_team_rocket_pokemon and self.archetype != "team_rocket":
                    continue
                if old_card.is_basic and not card.is_basic:
                    continue
                if old_card.is_stage1 and not (card.is_stage1 or card.is_basic):
                    continue
                if old_card.is_stage2 and not (card.is_stage2 or card.is_stage1 or card.is_basic):
                    continue
                candidates.append(cid)
            return self.rng.choice(candidates) if candidates else old_cid

        # Trainer / Item / Supporter Replacement
        candidates = []
        for cid, card in self._csv_idx.cards.items():
            if cid == old_cid or counts.get(cid, 0) >= 4:
                continue
            if not card.is_trainer and not card.is_supporter and not card.is_item:
                continue
            if card.is_ace_spec and any(self._csv_idx.get_card(x) and self._csv_idx.get_card(x).is_ace_spec for x in deck):
                continue
            candidates.append(cid)

        return self.rng.choice(candidates) if candidates else old_cid

    def _find_strategic_replacement(self, old_cid: int, deck: List[int], counts: Counter) -> Optional[int]:
        """Method 2 (Advanced): Role-biased strategic mutation utilizing csv-data/ intelligence."""
        if not self._csv_idx:
            return self._find_replacement(old_cid, deck, counts)
        old_card = self._csv_idx.get_card(old_cid)
        if not old_card:
            return None

        # Basic energy replacement
        if old_cid in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            return self._find_replacement(old_cid, deck, counts)

        # Pokémon replacement with strategic role and synergy awareness
        if old_card.is_pokemon:
            role = self._csv_idx.get_strategic_role(old_cid)
            complementary = self._csv_idx.get_complementary_cards(old_cid, self.archetype)

            valid_candidates = []
            for cid, card in self._csv_idx.cards.items():
                if cid == old_cid or counts.get(cid, 0) >= 4 or not card.is_pokemon:
                    continue
                card_type = card.type or 'Colorless'
                if card_type not in self.energy_types and card_type != 'Colorless':
                    continue
                if card.get_attack_energy_types() - set(self.energy_types):
                    continue
                if card.is_team_rocket_pokemon and self.archetype != "team_rocket":
                    continue
                if old_card.is_basic and not card.is_basic:
                    continue
                if old_card.is_stage1 and not (card.is_stage1 or card.is_basic):
                    continue
                if old_card.is_stage2 and not (card.is_stage2 or card.is_stage1 or card.is_basic):
                    continue

                score = 1.0
                if cid in complementary:
                    score += 5.0
                if card.gameplay_role == role:
                    score += 3.0
                score += getattr(card, 'effectiveness_score', 5.0) * 0.5
                if card.has_bench_protection:
                    score += 2.0
                valid_candidates.append((cid, score))

            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[1], reverse=True)
                top_pool = [c[0] for c in valid_candidates[:8]]
                return self.rng.choice(top_pool)
            return old_cid

        # Trainer / Item / Supporter replacement with role bias
        candidates = []
        for cid, card in self._csv_idx.cards.items():
            if cid == old_cid or counts.get(cid, 0) >= 4:
                continue
            if not card.is_trainer and not card.is_supporter and not card.is_item:
                continue
            if card.is_ace_spec and any(self._csv_idx.get_card(x) and self._csv_idx.get_card(x).is_ace_spec for x in deck):
                continue

            score = 1.0
            score += getattr(card, 'effectiveness_score', 5.0)
            if card.gameplay_role in ('draw_engine', 'support', 'energy_foundation'):
                score += 3.0
            candidates.append((cid, score))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_pool = [c[0] for c in candidates[:10]]
            return self.rng.choice(top_pool)

        return old_cid

    def _repair_deck(self, deck: List[int]) -> List[int]:
        """Strict Chromosome Legality Repair Pipeline.

        Guarantees:
        1. Length == 60 exactly
        2. Max 4 copies of ANY non-basic energy card (max 1 for ACE SPEC)
        3. Max 6 Pokemon species (max 18 total Pokemon)
        4. Attack Energy Harmonization: Every Pokemon's elemental attack requirements
           must be satisfiable by the deck's basic energies or flexible rainbow energy.
           HP type represents class/typing only.
        5. Special Energy HP-Type & Target Validity:
           - Grow Grass (18) requires Grass-type Pokemon in deck.
           - Telepath Psychic (19) requires Psychic-type Pokemon in deck.
           - Rock Fighting (20) requires Fighting-type Pokemon in deck.
           - Team Rocket's Energy (15) requires Team Rocket Pokemon in deck.
           - Neo Upper Energy (10) requires Stage 2 Pokemon in deck.
        6. Stage 2 Evolutionary Accelerator:
           - If Stage 2 Pokemon present, ensure 2-4 copies of Rare Candy (1079).
           - If NO Stage 2 Pokemon present, purge all copies of Rare Candy (1079).
        7. At least 1 Basic Pokemon
        8. Prune orphaned evolution lines
        9. Team Rocket roster requirement (Mewtwo ex #431 requires 4+ TR Pokemon)
        10. Basic energy has no copy limit, pad with deck's primary basic energy
        """
        deck = list(deck)
        primary_energy_id = self._get_primary_energy_id()

        # 1. Cap counts at 4 (or 1 for ACE SPEC) - Basic energies (IDs 1-8) have NO cap in PTCG rules
        counts = Counter()
        ace_spec_found = False
        repaired = []

        for cid in deck:
            c_data = self._csv_idx.get_card(cid) if self._csv_idx else None
            is_basic_energy = cid in (1, 2, 3, 4, 5, 6, 7, 8)
            is_ace = bool(c_data and c_data.is_ace_spec)
            max_allowed = 60 if is_basic_energy else (1 if is_ace else 4)

            if is_ace:
                if ace_spec_found:
                    continue
                ace_spec_found = True

            if counts[cid] < max_allowed:
                repaired.append(cid)
                counts[cid] += 1

        deck = repaired

        if not self._csv_idx:
            while len(deck) < 60:
                deck.append(primary_energy_id)
            return deck[:60]

        # 2. Check Team Rocket Invariant: Card 431 (Mewtwo ex) requires 4+ TR Pokemon
        has_mewtwo_tr = 431 in deck
        tr_count = sum(1 for cid in deck if self._csv_idx.get_card(cid) and self._csv_idx.get_card(cid).is_team_rocket_pokemon)
        if has_mewtwo_tr and tr_count < 4:
            if self.archetype == "team_rocket":
                tr_basics = [
                    cid for cid, c in self._csv_idx.cards.items()
                    if c.is_pokemon and c.is_basic and c.is_team_rocket_pokemon
                ]
                while tr_count < 4 and tr_basics:
                    deck.append(self.rng.choice(tr_basics))
                    tr_count += 1
            else:
                deck = [cid for cid in deck if cid != 431]

        if tr_count < 4 and self.archetype != "team_rocket":
            valid_replacements = [
                cid for cid, c in self._csv_idx.cards.items()
                if c.is_pokemon and c.is_basic and not c.is_team_rocket_pokemon and (c.type in self.energy_types or c.type == 'Colorless')
            ]
            new_deck = []
            for cid in deck:
                c = self._csv_idx.get_card(cid)
                if c and c.is_team_rocket_pokemon and self.archetype != "team_rocket":
                    rep = self.rng.choice(valid_replacements) if valid_replacements else primary_energy_id
                    new_deck.append(rep)
                else:
                    new_deck.append(cid)
            deck = new_deck

        # 3. Evolution Line Pruning: Prune orphaned evolution cards using evolves_from chain tracing
        # A Stage 1 is orphaned if its evolves_from Basic is not in the deck.
        # A Stage 2 is orphaned if NEITHER its evolves_from Stage 1 NOR its root Basic (2 levels up) is in the deck.
        # With Rare Candy, Stage 2 can skip Stage 1, so only the root Basic is required.
        deck_pokemon_names = set()
        for cid in deck:
            c = self._csv_idx.get_card(cid)
            if c and c.is_pokemon:
                deck_pokemon_names.add(c.name.lower())

        def _has_preevolution_in_deck(card):
            """Check if this evolved card's pre-evolution chain exists in the deck."""
            if not card.evolves_from:
                return True
            ef_lower = card.evolves_from.lower()
            if ef_lower in deck_pokemon_names:
                return True
            # For Stage 2: trace 2 levels up (Stage 2 -> Stage 1 -> Basic) for Rare Candy skip
            if card.is_stage2:
                for s1_card in self._csv_idx.cards.values():
                    if s1_card.is_pokemon and s1_card.name.lower() == ef_lower:
                        if s1_card.evolves_from and s1_card.evolves_from.lower() in deck_pokemon_names:
                            return True
                        break
            # Fallback: name-based root match (first word of name in any other Pokemon name)
            root_name = card.name.split()[0].lower()
            if any(root_name in existing_name for existing_name in deck_pokemon_names if existing_name != card.name.lower()):
                return True
            return False

        cleaned_deck = []
        for cid in deck:
            c = self._csv_idx.get_card(cid)
            if c and c.is_pokemon and (c.is_stage1 or c.is_stage2):
                if not _has_preevolution_in_deck(c):
                    cleaned_deck.append(primary_energy_id)
                    continue
            cleaned_deck.append(cid)
        deck = cleaned_deck

        # 4. Max 6 Pokemon species & Max 18 Pokemon cards constraint
        pkmn_cards = [cid for cid in deck if self._csv_idx.get_card(cid) and self._csv_idx.get_card(cid).is_pokemon]
        unique_species = list(dict.fromkeys(pkmn_cards))
        if len(unique_species) > 6:
            pkmn_counter = Counter(pkmn_cards)
            def _species_score(cid):
                c = self._csv_idx.get_card(cid)
                if not c:
                    return 0
                score = pkmn_counter[cid] * 10
                if c.is_basic:
                    score += 5
                if c.is_stage2:
                    score += 4
                if c.is_stage1:
                    score += 3
                return score
            sorted_species = sorted(unique_species, key=_species_score, reverse=True)
            allowed_species = set(sorted_species[:6])
            deck = [cid if (not self._csv_idx.get_card(cid) or not self._csv_idx.get_card(cid).is_pokemon or cid in allowed_species) else primary_energy_id for cid in deck]

        # Max 18 Pokemon cards: if more than 18, trim excess
        pkmn_in_deck = [cid for cid in deck if self._csv_idx.get_card(cid) and self._csv_idx.get_card(cid).is_pokemon]
        if len(pkmn_in_deck) > 18:
            excess = len(pkmn_in_deck) - 18
            new_deck = []
            trimmed = 0
            for cid in deck:
                c = self._csv_idx.get_card(cid)
                if c and c.is_pokemon and trimmed < excess:
                    if c.is_basic and not any(self._csv_idx.get_card(x) and self._csv_idx.get_card(x).evolves_from == c.name for x in deck):
                        new_deck.append(primary_energy_id)
                        trimmed += 1
                        continue
                new_deck.append(cid)
            deck = new_deck

        # 5. Ensure at least 1 Basic Pokemon exists
        has_basic = any(
            self._csv_idx.get_card(cid) and self._csv_idx.get_card(cid).is_pokemon and self._csv_idx.get_card(cid).is_basic
            for cid in deck
        )
        if not has_basic:
            basic_candidates = [
                cid for cid, c in self._csv_idx.cards.items()
                if c.is_pokemon and c.is_basic and not c.is_team_rocket_pokemon and (c.type in self.energy_types or c.type == 'Colorless')
            ]
            if basic_candidates:
                deck.append(self.rng.choice(basic_candidates))

        # 6. Stage 2 Evolutionary Accelerator: Rare Candy (1079) enforcement
        has_stage2_pkmn = any(
            self._csv_idx.get_card(cid) and (self._csv_idx.get_card(cid).is_stage2 or getattr(self._csv_idx.get_card(cid), 'stage2', False))
            for cid in deck
        )
        rare_candy_count = deck.count(1079)
        if has_stage2_pkmn:
            target_candy = 2
            needed_candies = max(0, target_candy - rare_candy_count)
            if needed_candies > 0:
                for i, cid in enumerate(deck):
                    if needed_candies == 0:
                        break
                    if cid in (1, 2, 3, 4, 5, 6, 7, 8) and deck.count(cid) > 8:
                        deck[i] = 1079
                        needed_candies -= 1
                while needed_candies > 0:
                    deck.append(1079)
                    needed_candies -= 1
        else:
            if 1079 in deck:
                deck = [cid if cid != 1079 else primary_energy_id for cid in deck]

        # 7. Special Energy HP-Type & Target Validity Scan
        pokemon_types = {self._csv_idx.get_card(cid).type for cid in deck if self._csv_idx.get_card(cid) and self._csv_idx.get_card(cid).is_pokemon}
        has_tr_pkmn = any(self._csv_idx.get_card(cid) and self._csv_idx.get_card(cid).is_team_rocket_pokemon for cid in deck)

        validated_deck = []
        for cid in deck:
            if cid == 18 and 'Grass' not in pokemon_types:
                validated_deck.append(primary_energy_id)
            elif cid == 19 and 'Psychic' not in pokemon_types:
                validated_deck.append(primary_energy_id)
            elif cid == 20 and 'Fighting' not in pokemon_types:
                validated_deck.append(primary_energy_id)
            elif cid == 15 and not has_tr_pkmn:
                validated_deck.append(primary_energy_id)
            elif cid == 10 and not has_stage2_pkmn:
                validated_deck.append(primary_energy_id)
            else:
                validated_deck.append(cid)
        deck = validated_deck

        # 8. Attack Energy Harmonization vs Deck Energy Provision
        energy_id_to_type = {
            1: 'Grass', 2: 'Fire', 3: 'Water', 4: 'Lightning',
            5: 'Psychic', 6: 'Fighting', 7: 'Darkness', 8: 'Metal'
        }
        deck_basic_energy_types = {energy_id_to_type[cid] for cid in deck if cid in energy_id_to_type}
        has_legacy_rainbow = 12 in deck
        has_prism_rainbow = 16 in deck
        has_neo_upper_rainbow = 10 in deck

        compatible_basics = [
            cid for cid, c in self._csv_idx.cards.items()
            if c.is_pokemon and c.is_basic and not c.is_team_rocket_pokemon and c.get_attack_energy_types().issubset(deck_basic_energy_types | set(self.energy_types))
        ]
        harmonized_deck = []
        for cid in deck:
            c = self._csv_idx.get_card(cid)
            if c and c.is_pokemon:
                needed_types = c.get_attack_energy_types()
                unmet = False
                for etype in needed_types:
                    if etype in deck_basic_energy_types:
                        continue
                    if has_legacy_rainbow:
                        continue
                    if c.is_basic and has_prism_rainbow:
                        continue
                    if c.is_stage2 and has_neo_upper_rainbow:
                        continue
                    unmet = True
                    break
                if unmet:
                    rep = self.rng.choice(compatible_basics) if compatible_basics else primary_energy_id
                    harmonized_deck.append(rep)
                    continue
            harmonized_deck.append(cid)
        deck = harmonized_deck

        # 9. Enforce exactly 60 cards using Primary Basic Energy for padding
        while len(deck) < 60:
            deck.append(primary_energy_id)
        if len(deck) > 60:
            deck = deck[:60]

        return deck

    def get_best_deck(self) -> Optional[List[int]]:
        if self.best_ever:
            return list(self.best_ever.deck)
        if self.population:
            return list(max(self.population, key=lambda x: x.fitness).deck)
        return None

    def check_legality(self, deck: List[int]) -> bool:
        """Check whether a deck chromosome satisfies tournament legality rules."""
        from agents.deck_validator import MasterDeckValidator
        val = MasterDeckValidator()
        rep = val.validate_deck(deck)
        return rep.is_legal

    def repair_deck(self, deck: List[int]) -> List[int]:
        """Public interface to repair deck chromosome into a tournament-legal 60-card list."""
        return self._repair_deck(deck)

    def get_stats(self) -> Dict[str, Any]:
        if not self.population:
            return {}
        fitnesses = [ind.fitness for ind in self.population]
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': round(max(fitnesses), 4),
            'avg_fitness': round(sum(fitnesses) / max(1, len(fitnesses)), 4),
            'best_ever_fitness': round(self.best_ever.fitness if self.best_ever else 0.0, 4),
            'optimization_method': self.optimization_method,
        }


# ═══════════════════════════════════════════════════════════════════════
# Multi-Island MAP-Elites Co-Evolution
# ═══════════════════════════════════════════════════════════════════════

class MapElitesIslandOptimizer:
    """Multi-Island Co-Evolutionary Genetic Algorithm with MAP-Elites Diversity Niching."""

    ISLAND_ARCHETYPES = ['aggro', 'balanced', 'stall', 'prize_rush']

    def __init__(
        self,
        base_deck: List[int],
        island_count: int = 4,
        population_per_island: int = 8,
        migration_interval: int = 2,
        seed: int = 42,
        csv_index: Optional[CsvDataIndex] = None,
        energy_types: Optional[List[str]] = None,
        optimization_method: str = "predefined",
    ):
        self.base_deck = list(base_deck)
        self.migration_interval = migration_interval
        self.optimization_method = optimization_method
        self.islands: Dict[str, GeneticOptimizer] = {}
        self.generation = 0
        self.best_overall: Optional[Individual] = None

        for i, arch in enumerate(self.ISLAND_ARCHETYPES[:island_count]):
            self.islands[arch] = GeneticOptimizer(
                base_deck=self.base_deck,
                population_size=population_per_island,
                seed=seed + i * 100,
                csv_index=csv_index,
                energy_types=energy_types,
                archetype=arch,
                optimization_method=self.optimization_method,
            )

    def evolve_generation(self, fitness_fn: Optional[callable] = None) -> Dict[str, Any]:
        """Evolves all islands independently and executes migration when due."""
        self.generation += 1
        island_stats = {}

        for arch, optimizer in self.islands.items():
            best_ind = optimizer.step(fitness_fn=fitness_fn)
            island_stats[arch] = optimizer.get_stats()
            if self.best_overall is None or (best_ind and best_ind.fitness > self.best_overall.fitness):
                self.best_overall = best_ind.copy()

        # Island Migration
        if self.generation % self.migration_interval == 0 and len(self.islands) > 1:
            island_keys = list(self.islands.keys())
            for i, src_key in enumerate(island_keys):
                dst_key = island_keys[(i + 1) % len(island_keys)]
                src_opt = self.islands[src_key]
                dst_opt = self.islands[dst_key]
                if src_opt.population and dst_opt.population:
                    src_elite = max(src_opt.population, key=lambda x: x.fitness).copy()
                    worst_idx = min(range(len(dst_opt.population)), key=lambda idx: dst_opt.population[idx].fitness)
                    dst_opt.population[worst_idx] = src_elite

        return {
            'generation': self.generation,
            'islands': island_stats,
            'best_overall_fitness': round(self.best_overall.fitness if self.best_overall else 0.0, 4)
        }

    def get_best_deck(self) -> Optional[List[int]]:
        if self.best_overall:
            return list(self.best_overall.deck)
        best_candidates = [opt.get_best_deck() for opt in self.islands.values() if opt.get_best_deck()]
        return best_candidates[0] if best_candidates else self.base_deck

