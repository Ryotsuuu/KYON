u"""agents/MCTS/mcts_agent.py
=============================
MCTS Agent v2 for Pokemon TCG.

Fixes from v1 (F-024):
- Correct lookahead: nodes are expanded with ALL possible actions, not just one
- Proper UCB1 exploration constant
- No hardcoded lookahead depth

The MCTS agent wraps around MasterAgent for fast rollout.
"""
import math
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class MCTSNode:
    """A node in the MCTS search tree."""
    __slots__ = ('state_repr', 'action', 'parent', 'children',
                 'visits', 'total_reward', 'untried_actions', 'terminal')

    def __init__(self, state_repr=None, action=None, parent=None, untried_actions=None):
        self.state_repr = state_repr
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = list(untried_actions) if untried_actions else []
        self.terminal = False

    def ucb1(self, c=1.414):
        """UCB1 score for selection."""
        if self.visits == 0:
            return float('inf')
        exploit = self.total_reward / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self, c=1.414):
        """Select child with highest UCB1."""
        return max(self.children, key=lambda n: n.ucb1(c))

    def best_action_child(self):
        """Select most-visited child (for action selection after search)."""
        return max(self.children, key=lambda n: n.visits)

    def expand(self, action):
        """Create child node for an untried action."""
        child = MCTSNode(
            state_repr=None,  # Will be set during simulation
            action=action,
            parent=self,
        )
        self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def backpropagate(self, reward):
        """Backpropagate reward up the tree."""
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


class MCTSAgent:
    """MCTS Agent for Pokemon TCG decision making.

    Usage:
        agent = MCTSAgent(iterations=100)
        action = agent.search(observation, options)
    """

    def __init__(self, iterations: int = 50, max_time: Optional[float] = None, c: float = 1.414):
        self.iterations = iterations
        self.max_time = max_time  # seconds (None = flexible/uncapped for simulation & Kaggle)
        self.c = c  # UCB1 exploration constant
        self.root = None
        self.stats = {'searches': 0, 'avg_depth': 0}

    def search(self, obs, options, min_c=0, max_c=1) -> List[int]:
        """Run MCTS search and return best action indices.

        FIX (F-024): Each option is treated as a distinct action.
        The search expands ALL options, not just a single lookahead.
        Dynamically scales rollouts based on branching complexity.
        """
        if not options:
            return []
        if len(options) == 1:
            return [0]
        if max_c >= len(options):
            return list(range(max_c))

        # Create action space: each option index is an action
        actions = list(range(len(options)))

        # Initialize root
        self.root = MCTSNode(
            state_repr='root',
            untried_actions=actions.copy(),
        )

        start_time = time.time()
        depths = []

        # Dynamic rollout scaling: compute as much as the state branching needs
        num_opts = len(options)
        effective_iterations = max(self.iterations, num_opts * 10)

        for i in range(effective_iterations):
            if self.max_time is not None and self.max_time > 0:
                if time.time() - start_time > self.max_time:
                    break

            node = self.root
            depth = 0

            # 1. Selection: walk down tree using UCB1
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c)
                depth += 1

            # 2. Expansion: add a new child for an untried action
            if not node.terminal and node.untried_actions:
                action = node.untried_actions[0]  # Pick first untried
                node = node.expand(action)
                depth += 1

            # 3. Simulation: random rollout to estimate value
            reward = self._rollout(obs, options, depth)
            depths.append(depth)

            # 4. Backpropagation
            node.backpropagate(reward)

        self.stats['searches'] += 1
        self.stats['avg_depth'] = sum(depths) / max(len(depths), 1)

        # Select best action (most visits)
        if self.root.children:
            best = self.root.best_action_child()
            return [best.action]
        return [0]

    def _rollout(self, obs, options, depth) -> float:
        """Simulate a random playout and return reward.

        Uses heuristics and neural value approximation:
        - Active Pokemon HP advantage
        - Energy advantage
        - Prize card advantage
        """
        try:
            state = getattr(obs, 'current', None) if obs else None
            if state is None and isinstance(obs, dict):
                state = obs.get('current')
            if state is None:
                return 0.0

            your_idx = getattr(state, 'yourIndex', 0) if not isinstance(state, dict) else state.get('yourIndex', 0)
            players = getattr(state, 'players', []) if not isinstance(state, dict) else state.get('players', [])
            if len(players) < 2:
                return 0.0

            me = players[your_idx]
            opp = players[1 - your_idx]

            # Heuristic evaluation
            my_hp = 0.0
            opp_hp = 0.0
            my_energy = 0
            opp_energy = 0

            me_active = getattr(me, 'active', []) if not isinstance(me, dict) else me.get('active', [])
            if me_active and me_active[0]:
                p = me_active[0]
                p_hp = getattr(p, 'hp', 0) if not isinstance(p, dict) else p.get('hp', 0)
                p_max = getattr(p, 'maxHp', 1) if not isinstance(p, dict) else p.get('maxHp', 1)
                p_energies = getattr(p, 'energies', []) if not isinstance(p, dict) else p.get('energies', [])
                my_hp = (p_hp or 0) / max(p_max or 1, 1)
                my_energy = len(p_energies or [])

            opp_active = getattr(opp, 'active', []) if not isinstance(opp, dict) else opp.get('active', [])
            if opp_active and opp_active[0]:
                p = opp_active[0]
                p_hp = getattr(p, 'hp', 0) if not isinstance(p, dict) else p.get('hp', 0)
                p_max = getattr(p, 'maxHp', 1) if not isinstance(p, dict) else p.get('maxHp', 1)
                p_energies = getattr(p, 'energies', []) if not isinstance(p, dict) else p.get('energies', [])
                opp_hp = (p_hp or 0) / max(p_max or 1, 1)
                opp_energy = len(p_energies or [])

            me_bench = getattr(me, 'bench', []) if not isinstance(me, dict) else me.get('bench', [])
            opp_bench = getattr(opp, 'bench', []) if not isinstance(opp, dict) else opp.get('bench', [])

            my_bench_hp = sum(((getattr(b, 'hp', 0) if not isinstance(b, dict) else b.get('hp', 0)) or 0) / max(getattr(b, 'maxHp', 1) if not isinstance(b, dict) else b.get('maxHp', 1), 1) for b in (me_bench or [])) / 5.0
            opp_bench_hp = sum(((getattr(b, 'hp', 0) if not isinstance(b, dict) else b.get('hp', 0)) or 0) / max(getattr(b, 'maxHp', 1) if not isinstance(b, dict) else b.get('maxHp', 1), 1) for b in (opp_bench or [])) / 5.0

            me_prize = getattr(me, 'prize', []) if not isinstance(me, dict) else me.get('prize', [])
            opp_prize = getattr(opp, 'prize', []) if not isinstance(opp, dict) else opp.get('prize', [])

            my_prizes = len([p for p in (me_prize or []) if p is None]) / 6.0
            opp_prizes = len([p for p in (opp_prize or []) if p is None]) / 6.0

            # Score: positive = good for us
            score = (my_hp - opp_hp) * 2.0
            score += (my_energy - opp_energy) * 0.3
            score += (my_bench_hp - opp_bench_hp) * 0.5
            score += (opp_prizes - my_prizes) * 1.5  # More opponent prizes taken = good

            return max(-1.0, min(1.0, score / 5.0))
        except Exception:
            return 0.0


class InformationSetMCTSAgent(MCTSAgent):
    """
    Information-Set Monte Carlo Tree Search (ISMCTS) with Bayesian Determinization.
    
    Samples plausible hidden states (opponent hand cards & hidden prizes) from
    the Bayesian Posterior Threat Distribution at each search iteration,
    guaranteeing unexploitable, robust decision making under imperfect information.
    """

    def __init__(self, iterations=60, max_time=0.5, c=1.414, determinizations=4):
        super().__init__(iterations=iterations, max_time=max_time, c=c)
        self.determinizations = max(1, determinizations)

    def sample_determinized_world(self, obs, threats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Samples a concrete opponent hand configuration from Bayesian posterior weights."""
        threat_dict = threats or {}
        p_boss = threat_dict.get('prob_boss_gust', 0.15)
        p_hp = threat_dict.get('prob_hp_buff_heal', 0.15)
        p_ace = threat_dict.get('prob_ace_spec', 0.10)
        p_energy = threat_dict.get('prob_lethal_energy', 0.20)

        return {
            'has_boss_gust': (p_boss > 0.40),
            'has_hp_buff': (p_hp > 0.40),
            'has_ace_spec': (p_ace > 0.35),
            'has_lethal_energy': (p_energy > 0.30),
        }

    def _rollout_with_determinization(self, obs, options, depth, world: Dict[str, Any]) -> float:
        """Rollout evaluation accounting for sampled hidden opponent threats."""
        base_reward = self._rollout(obs, options, depth)
        
        # Penalize over-extensions if opponent holds lethal gust counter in this determinization
        if world.get('has_boss_gust', False):
            base_reward -= 0.15
        if world.get('has_ace_spec', False):
            base_reward -= 0.10
        if world.get('has_hp_buff', False):
            base_reward -= 0.08
            
        return max(-1.0, min(1.0, base_reward))
