"""
agents/RL/ptcg_env.py
=====================
Gymnasium-compatible Pokémon TCG Reinforcement Learning Environment.

Interfaces directly with ctypes C-engine (`cg.sim.BattleLocal`) for deterministic,
high-speed step transitions and rich potential-based reward shaping.
"""
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cg.api import to_observation_class, SelectContext, OptionType
from agents.NN import encode_dynamic_state, STATE_DIM, ACTION_DIM
from agents.Resource_Management.simulation_runner import battle_start, battle_select, battle_finish
from simulation.Decision_Engine import MasterAgent

logger = logging.getLogger(__name__)


class PTCGEnvironment:
    """Standard Gymnasium-style environment wrapping cg simulation."""

    def __init__(self, deck: Optional[List[int]] = None, opponent_deck: Optional[List[int]] = None):
        self.deck = deck or [1] * 60
        self.opponent_deck = opponent_deck or [1] * 60
        self.opponent_agent = MasterAgent(deck=self.opponent_deck)
        self.current_obs = None
        self.raw_obs = None
        self.done = False
        self.steps = 0
        self.prev_my_prizes = 6
        self.prev_opp_prizes = 6
        self.prev_my_hp = 100
        self.prev_opp_hp = 100

    @property
    def observation_space_dim(self) -> int:
        return STATE_DIM

    @property
    def action_space_dim(self) -> int:
        return ACTION_DIM

    def reset(self) -> np.ndarray:
        """Reset battle environment and initialize game."""
        battle_finish()
        self.raw_obs, _ = battle_start(self.deck, self.opponent_deck)
        self.current_obs = to_observation_class(self.raw_obs)
        self.done = False
        self.steps = 0
        self.prev_my_prizes = 6
        self.prev_opp_prizes = 6
        self.prev_my_hp = 100
        self.prev_opp_hp = 100

        # Step through until it is player 0's turn to act
        self._advance_to_player_turn()

        return encode_dynamic_state(self.current_obs)

    def _advance_to_player_turn(self):
        """Execute automated steps or opponent agent decisions until player 0 has a decision."""
        while self.raw_obs and self.raw_obs.get('select') is not None and not self.done and self.steps < 150:
            p_idx = self.raw_obs.get('current', {}).get('yourIndex', 0)
            if p_idx == 0:
                break

            # Opponent turn: let MasterAgent decide
            opp_act = self.opponent_agent(self.raw_obs)
            if not opp_act:
                opp_act = []
            try:
                self.raw_obs = battle_select(opp_act)
                self.current_obs = to_observation_class(self.raw_obs)
            except Exception:
                self.done = True
                break
            self.steps += 1

        if not self.raw_obs or self.raw_obs.get('select') is None:
            self.done = True

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute player action, advance game, and return (state, reward, done, info)."""
        if self.done or self.raw_obs is None:
            return np.zeros(STATE_DIM, dtype=np.float32), 0.0, True, {'winner': 2}

        options = self.raw_obs.get('select', {}).get('option', [])
        selection = [0]
        if options:
            if 0 <= action_idx < len(options):
                selection = [action_idx]
            else:
                selection = [0]

        try:
            self.raw_obs = battle_select(selection)
            self.current_obs = to_observation_class(self.raw_obs)
        except Exception:
            self.done = True
            return np.zeros(STATE_DIM, dtype=np.float32), 0.0, True, {'winner': 2}

        self.steps += 1
        self._advance_to_player_turn()

        # Compute shaped reward
        reward, info = self._compute_reward()

        state_vec = encode_dynamic_state(self.current_obs)
        return state_vec, reward, self.done, info

    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        """Reward shaping: dense prize differential + health delta + terminal outcome."""
        info = {'winner': -1}
        if self.current_obs is None or self.current_obs.current is None:
            if self.done:
                return 0.0, {'winner': 2}
            return 0.0, info

        state = self.current_obs.current
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]

        cur_my_prizes = sum(1 for p in (me.prize or []) if p is not None)
        cur_opp_prizes = sum(1 for p in (opp.prize or []) if p is not None)

        cur_my_hp = me.active[0].hp if (me.active and me.active[0]) else 0
        cur_opp_hp = opp.active[0].hp if (opp.active and opp.active[0]) else 0

        reward = 0.0

        # Prize difference progress
        my_prize_taken = max(0, self.prev_opp_prizes - cur_opp_prizes)
        opp_prize_taken = max(0, self.prev_my_prizes - cur_my_prizes)
        reward += 0.35 * my_prize_taken - 0.25 * opp_prize_taken

        # Damage progress
        damage_dealt = max(0, self.prev_opp_hp - cur_opp_hp)
        damage_taken = max(0, self.prev_my_hp - cur_my_hp)
        reward += 0.001 * (damage_dealt - damage_taken)

        self.prev_my_prizes = cur_my_prizes
        self.prev_opp_prizes = cur_opp_prizes
        self.prev_my_hp = cur_my_hp
        self.prev_opp_hp = cur_opp_hp

        # Check terminal outcome
        result = state.result
        if result in (0, 1):
            self.done = True
            info['winner'] = result
            reward += 1.0 if result == 0 else -1.0
        elif cur_opp_prizes == 0:
            self.done = True
            info['winner'] = 0
            reward += 1.0
        elif cur_my_prizes == 0:
            self.done = True
            info['winner'] = 1
            reward -= 1.0
        elif self.steps >= 120:
            self.done = True
            info['winner'] = 2

        return round(float(reward), 4), info

    def close(self):
        battle_finish()
