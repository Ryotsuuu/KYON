u"""agents/Expert_Agent_Ecosystem/energy_agents.py
===============================================
Energy Agent Ecosystem: Selects optimal energy attachment targets.

v2 Fix: Priority to active Pokemon, then best bench (next to take active).
Stacks energy in one turn if possible.
"""
from typing import List, Dict, Optional, Tuple


class EnergyAgentEcosystem:
    """Orchestrates energy attachment decisions."""

    def __init__(self):
        self.agents = {}

    def build_for_deck(self, deck: List[int], energy_types: List[str]):
        """Build energy agent profiles for a deck."""
        self.agents['primary'] = PrimaryEnergyAgent(energy_types)
        self.agents['priority'] = PriorityEnergyAgent()

    def select_target(self, active_pokemon, bench_pokemon, available_energy) -> Tuple[str, int, float]:
        """Select best target for energy attachment.
        
        Returns: (position, index, score) - ('active', 0, score) or ('bench', bench_index, score)
        """
        # Priority 1: Active Pokemon (if it needs energy)
        if active_pokemon:
            return self.agents['priority'].score_target(active_pokemon, 'active', 0)
        
        # Priority 2: Best bench Pokemon (closest to being able to attack)
        best = None
        best_score = -1.0
        for idx, bp in enumerate(bench_pokemon or []):
            score_data = self.agents['priority'].score_target(bp, 'bench', idx)
            if score_data[2] > best_score:
                best_score = score_data[2]
                best = score_data
        
        return best or ('active', 0, 0.0)


class PrimaryEnergyAgent:
    """Selects which energy type to attach."""

    def __init__(self, energy_types: List[str]):
        self.energy_types = energy_types


class PriorityEnergyAgent:
    """Scores Pokemon for energy attachment priority.
    
    v2 Fix: Active Pokemon gets highest priority.
    Energy stacking: attach all needed energy, don't wait.
    """

    def score_target(self, pokemon, position: str, index: int) -> Tuple[str, int, float]:
        """Score a Pokemon for energy priority.
        
        Returns: (position, index, score)
        """
        hp = getattr(pokemon, 'hp', 0) or 0
        max_hp = getattr(pokemon, 'maxHp', 1) or 1
        energy_count = len(getattr(pokemon, 'energies', []) or [])
        hp_ratio = hp / max_hp

        score = 0.0

        # Active Pokemon gets massive priority boost
        if position == 'active':
            score += 1000.0
            # If close to being able to attack, even higher
            if energy_count >= 1:
                score += 500.0
        else:
            # Bench: prioritize Pokemon close to being ready
            if energy_count >= 1:
                score += 300.0

        # HP factor
        score += hp_ratio * 100.0

        # EX/Mega bonus (they're the damage dealers)
        card = None
        try:
            from agents.csv_data import get_csv_index
            idx = get_csv_index()
            card = idx.get_card(pokemon.id if hasattr(pokemon, 'id') else 0)
        except Exception:
            pass
        if card:
            if card.is_ex:
                score += 50.0
            if card.is_mega_ex:
                score += 80.0

        return (position, index, score)
