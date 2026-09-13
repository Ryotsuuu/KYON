"""
agents/Learning_System/condition_tracker.py
==========================================
Dynamic Gameplay Condition Tracker: Micro & Macro Event Telemetry for PTCG AI.

Capabilities:
- Micro-Events: Step-level card play, damage, energy attachment, status infliction, retreat timing
- Macro-Events: Turn-to-KO ratios, prize clock progression, comeback windows, tempo advantage
- Analytics Mining: Computes card efficiency, action quality, and strategic state transitions
"""
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class DynamicGameplayConditionTracker:
    """Tracks and records fine-grained micro and macro game events during battle."""

    def __init__(self, agent_name: str = "Agent_P1", opponent_name: str = "Agent_P2"):
        self.agent_name = agent_name
        self.opponent_name = opponent_name
        self.micro_events: List[Dict[str, Any]] = []
        self.macro_summary: Dict[str, Any] = {}
        self.start_time = time.perf_counter()

    def record_micro_step(
        self,
        turn: int,
        player_idx: int,
        context: int,
        action_type: int,
        selected_indices: List[int],
        posture: int = 0,
        my_active_hp: int = 0,
        opp_active_hp: int = 0,
        my_prizes_remaining: int = 6,
        opp_prizes_remaining: int = 6,
        event_tag: str = ""
    ) -> None:
        """Log a fine-grained step event during match simulation."""
        self.micro_events.append({
            'turn': turn,
            'player_idx': player_idx,
            'context': context,
            'action_type': action_type,
            'selected_indices': selected_indices,
            'posture': posture,
            'my_active_hp': my_active_hp,
            'opp_active_hp': opp_active_hp,
            'my_prizes_remaining': my_prizes_remaining,
            'opp_prizes_remaining': opp_prizes_remaining,
            'prize_differential': opp_prizes_remaining - my_prizes_remaining,
            'event_tag': event_tag,
            'timestamp': round(time.perf_counter() - self.start_time, 4),
        })

    def record_step(self, turn: int, dynamics: Dict[str, Any]) -> None:
        """Log a turn-level dynamics snapshot directly into condition tracking."""
        event = {
            'turn': turn,
            'timestamp': round(time.perf_counter() - self.start_time, 4),
            **dynamics
        }
        self.micro_events.append(event)

    def finalize_match(
        self,
        winner: int,
        total_turns: int,
        deck1: List[int],
        deck2: List[int]
    ) -> Dict[str, Any]:
        """Aggregate micro-events into a comprehensive macro-condition summary."""
        duration = round(time.perf_counter() - self.start_time, 3)

        # Micro statistics extraction
        attacks_count = sum(1 for e in self.micro_events if e.get('action_type') == 13)  # ATTACK
        energy_attachments = sum(1 for e in self.micro_events if e.get('action_type') == 8)  # ATTACH
        pivots_count = sum(1 for e in self.micro_events if e.get('action_type') == 12)  # RETREAT
        evolutions_count = sum(1 for e in self.micro_events if e.get('action_type') == 9)  # EVOLVE

        # Prize curve progression
        prize_curve = [
            (e['turn'], e['prize_differential'])
            for e in self.micro_events
            if 'prize_differential' in e
        ]

        # Comeback detection: was agent ever at a deficit of >= 2 prizes and won?
        had_deficit = any(e.get('prize_differential', 0) <= -2 for e in self.micro_events)
        is_comeback = (winner == 0 and had_deficit)

        self.macro_summary = {
            'agent_name': self.agent_name,
            'opponent_name': self.opponent_name,
            'winner': winner,
            'total_turns': total_turns,
            'duration_sec': duration,
            'deck1_sample': deck1[:10],
            'deck2_sample': deck2[:10],
            'metrics': {
                'attacks_count': attacks_count,
                'energy_attachments': energy_attachments,
                'pivots_count': pivots_count,
                'evolutions_count': evolutions_count,
                'is_comeback_win': is_comeback,
                'micro_event_count': len(self.micro_events),
            },
            'prize_curve': prize_curve[-10:] if prize_curve else [],
        }

        return {
            'macro_summary': self.macro_summary,
            'micro_events': self.micro_events,
        }


class CardLearningTracker:
    """Tracks and records fine-grained card usage, value upgrades, and timing analytics."""

    def __init__(self, storage_file: Optional[Any] = None):
        from pathlib import Path
        self.storage_file = storage_file or (Path(__file__).resolve().parent.parent.parent / "data" / "card_learning_analytics.json")
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        import json
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'total_simulations_tracked': 0,
            'total_turns_simulated': 0,
            'card_usage_stats': {},
            'neural_loss_history': [],
            'updated_at': None
        }

    def _save(self) -> None:
        import json
        try:
            self.data['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def record_simulation_run(
        self,
        games_completed: int,
        turns_total: int,
        deck1: List[int],
        deck2: List[int],
        winner: int,
        sim_type: str = "simulate",
        loss_val: Optional[float] = None
    ) -> Dict[str, Any]:
        """Record batch simulation results, updating card usage frequencies and win attributions."""
        self.data['total_simulations_tracked'] += games_completed
        self.data['total_turns_simulated'] += turns_total

        all_cards = set(deck1 + deck2)
        for cid in all_cards:
            str_cid = str(cid)
            if str_cid not in self.data['card_usage_stats']:
                self.data['card_usage_stats'][str_cid] = {
                    'total_played': 0,
                    'wins': 0,
                    'losses': 0,
                    'draws': 0,
                    'win_rate': 0.50,
                }

            stat = self.data['card_usage_stats'][str_cid]
            stat['total_played'] += games_completed

            in_d1 = (cid in deck1)
            in_d2 = (cid in deck2)
            if winner == 0 and in_d1:
                stat['wins'] += games_completed
            elif winner == 1 and in_d2:
                stat['wins'] += games_completed
            elif winner == 2:
                stat['draws'] += games_completed
            else:
                stat['losses'] += games_completed

            tot = stat['wins'] + stat['losses'] + stat['draws']
            stat['win_rate'] = round(stat['wins'] / max(1, tot), 4)

        if loss_val is not None:
            self.data['neural_loss_history'].append({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'loss': round(float(loss_val), 4),
                'games': games_completed,
                'sim_type': sim_type
            })
            if len(self.data['neural_loss_history']) > 100:
                self.data['neural_loss_history'] = self.data['neural_loss_history'][-100:]

        self._save()
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregated summary of top-performing cards and learning milestones."""
        stats = self.data.get('card_usage_stats', {})
        ranked_cards = []
        for str_cid, s in stats.items():
            if s['total_played'] >= 4:
                ranked_cards.append({
                    'card_id': int(str_cid),
                    'total_played': s['total_played'],
                    'wins': s['wins'],
                    'win_rate': s['win_rate']
                })

        ranked_cards.sort(key=lambda x: (x['win_rate'], x['total_played']), reverse=True)

        recent_losses = [h['loss'] for h in self.data.get('neural_loss_history', [])[-5:]]
        avg_recent_loss = round(sum(recent_losses) / len(recent_losses), 4) if recent_losses else 0.9500

        return {
            'total_simulations': self.data.get('total_simulations_tracked', 0),
            'total_turns': self.data.get('total_turns_simulated', 0),
            'unique_cards_tracked': len(stats),
            'top_performing_cards': ranked_cards[:10],
            'recent_loss_avg': avg_recent_loss,
            'learning_steps_recorded': len(self.data.get('neural_loss_history', []))
        }


_card_learning_tracker_instance: Optional[CardLearningTracker] = None


def get_card_learning_tracker() -> CardLearningTracker:
    global _card_learning_tracker_instance
    if _card_learning_tracker_instance is None:
        _card_learning_tracker_instance = CardLearningTracker()
    return _card_learning_tracker_instance


class DeductivePrizeLedger:
    """Zero-Hallucination Exact Prize Card Deductive Ledger for competitive matches."""

    def __init__(self, initial_deck: Optional[List[int]] = None):
        self.initial_deck = list(initial_deck) if initial_deck else []
        self.known_trapped_prizes: List[int] = []
        self.is_deduced = False

    def reconcile_search_observation(
        self,
        hand_cards: List[int],
        in_play_cards: List[int],
        discard_cards: List[int],
        remaining_deck_cards: List[int]
    ) -> List[int]:
        """Deduce exact prize cards via set subtraction when deck is visible during a search."""
        if not self.initial_deck or self.is_deduced:
            return self.known_trapped_prizes

        from collections import Counter
        deck_counter = Counter(self.initial_deck)
        
        # Subtract all known visible cards
        visible_cards = hand_cards + in_play_cards + discard_cards + remaining_deck_cards
        for c in visible_cards:
            if deck_counter[c] > 0:
                deck_counter[c] -= 1

        # The remaining cards in counter are exactly the 6 prize cards!
        trapped = []
        for cid, count in deck_counter.items():
            trapped.extend([cid] * count)

        self.known_trapped_prizes = trapped[:6]
        self.is_deduced = (len(self.known_trapped_prizes) == 6)
        return self.known_trapped_prizes

    def is_card_trapped_in_prizes(self, card_id: int) -> bool:
        """Check if all copies of a specific card are confirmed trapped in prize cards."""
        return card_id in self.known_trapped_prizes


_deductive_prize_ledger_instance: Optional[DeductivePrizeLedger] = None


def get_deductive_prize_ledger() -> DeductivePrizeLedger:
    global _deductive_prize_ledger_instance
    if _deductive_prize_ledger_instance is None:
        _deductive_prize_ledger_instance = DeductivePrizeLedger()
    return _deductive_prize_ledger_instance


