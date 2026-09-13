"""
simulation/simulator.py
=======================
High-Fidelity Virtual State Transition Engine & Multi-Ply Simulation Generator.
Extracted from AlphaZeroAgent to provide shared, zero-hallucination forward game mechanics
across MCTS, tactical planning, and counter-strike rollouts.
"""
from __future__ import annotations

import copy
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

try:
    from cg.api import OptionType
except ImportError:
    class OptionType:
        NUMBER = 0
        YES = 1
        NO = 2
        CARD = 3
        TOOL_CARD = 4
        ENERGY_CARD = 5
        ENERGY = 6
        PLAY = 7
        ATTACH = 8
        EVOLVE = 9
        ABILITY = 10
        DISCARD = 11
        RETREAT = 12
        ATTACK = 13
        END = 14

logger = logging.getLogger(__name__)


class VirtualOption:
    """Lightweight polymorphic action representation for virtual rollout simulations."""
    def __init__(self, opt_type: int, card_id: Optional[int] = None, attack_id: Optional[int] = None, name: str = ""):
        self.type = opt_type
        self.cardId = card_id
        self.attackId = attack_id
        self.name = name

    def __repr__(self):
        return f"<VirtualOption type={self.type} name='{self.name}' cardId={self.cardId} atk={self.attackId}>"


class VirtualGameSimulator:
    """
    High-Fidelity Virtual State Transition Engine.
    Simulates forward consequence of actions across attacks, knockouts,
    evolutions, attachments, retreats, and opponent counter-attacks.
    """

    def __init__(self, cards: Optional[Dict[int, Any]] = None, attacks: Optional[Dict[int, Any]] = None):
        self.cards = cards
        self.attacks = attacks

    @staticmethod
    def _to_clean_state_dict(obs: Any) -> Dict[str, Any]:
        """Convert any observation (object or dict) into a uniform simulator dict."""
        cur = getattr(obs, 'current', obs) if not isinstance(obs, dict) else obs.get('current', obs)
        if isinstance(cur, dict):
            return copy.deepcopy(cur)
        
        players = []
        for p in (getattr(cur, 'players', []) or []):
            act_list = []
            for m in (getattr(p, 'active', []) or []):
                act_list.append({
                    'hp': getattr(m, 'hp', 100),
                    'maxHp': getattr(m, 'maxHp', 100),
                    'energies': list(getattr(m, 'energies', []) or []),
                    'ex': getattr(m, 'ex', False),
                    'megaEx': getattr(m, 'megaEx', False),
                    'cardId': getattr(m, 'cardId', 0),
                })
            bench_list = []
            for m in (getattr(p, 'bench', []) or []):
                bench_list.append({
                    'hp': getattr(m, 'hp', 80),
                    'maxHp': getattr(m, 'maxHp', 80),
                    'energies': list(getattr(m, 'energies', []) or []),
                    'ex': getattr(m, 'ex', False),
                    'megaEx': getattr(m, 'megaEx', False),
                    'cardId': getattr(m, 'cardId', 0),
                })
            players.append({
                'active': act_list,
                'bench': bench_list,
                'prize': list(getattr(p, 'prize', []) or []),
                'handCount': getattr(p, 'handCount', 5),
            })
        return {
            'players': players,
            'yourIndex': getattr(cur, 'yourIndex', 0),
            'turn': getattr(cur, 'turn', 1),
        }

    @staticmethod
    def _resolve_ability_modifiers(my_card, my_bench_cards, opp_card, opp_bench_cards, base_dmg: int) -> int:
        """Calculate ability modifications (damage boosts, damage reductions, immunity shields)."""
        bonus_dmg = 0
        damage_reduction = 0

        # 1. Attacker & Attacker Bench abilities (damage boosts / offensive buffs)
        all_my_cards = [my_card] + [b for b in my_bench_cards if b]
        for c in all_my_cards:
            if not c or not getattr(c, 'skills', None):
                continue
            for s in c.skills:
                txt = (getattr(s, 'text', '') or '').lower()
                m_more = re.search(r'(\d+)\s+more\s+damage', txt)
                if m_more:
                    bonus_dmg += int(m_more.group(1))

        # 2. Defender & Defender Bench abilities (defensive reductions / immunity shields)
        all_opp_cards = [opp_card] + [b for b in opp_bench_cards if b]
        for c in all_opp_cards:
            if not c or not getattr(c, 'skills', None):
                continue
            for s in c.skills:
                txt = (getattr(s, 'text', '') or '').lower()
                m_less = re.search(r'(\d+)\s+(?:less\s+damage|damage\s+done)', txt)
                if m_less:
                    damage_reduction += int(m_less.group(1))
                # Complete immunity shields
                if 'prevent all damage' in txt:
                    if 'basic' in txt and getattr(my_card, 'basic', False):
                        return 0
                    elif 'ex' in txt and getattr(my_card, 'ex', False):
                        return 0

        adjusted_dmg = max(0, base_dmg + bonus_dmg - damage_reduction)
        return adjusted_dmg

    @staticmethod
    def simulate_action(cur_dict: Dict[str, Any], option: Any, your_idx: int) -> Tuple[Dict[str, Any], float, bool]:
        """
        Takes current state dict and chosen option, returns:
        (next_state_dict, immediate_reward, is_terminal) using actual card data.
        """
        from simulation.Decision_Engine import _get_cards, _get_attacks
        cards_db = _get_cards()
        attacks_db = _get_attacks()

        s = copy.deepcopy(cur_dict) if isinstance(cur_dict, dict) else {}
        players = s.get('players', [{}, {}])
        if len(players) < 2:
            return s, 0.0, False

        me = players[your_idx]
        opp = players[1 - your_idx]

        reward = 0.0
        terminal = False

        opt_type = getattr(option, 'type', None) if not isinstance(option, dict) else option.get('type')
        
        # 1. ATTACK (OptionType.ATTACK or 13)
        if opt_type in (OptionType.ATTACK, 13):
            me_act = me.get('active', [])
            opp_act = opp.get('active', [])
            
            my_mon = me_act[0] if me_act and me_act[0] else {}
            my_card_id = my_mon.get('cardId', 0)
            my_card = cards_db.get(my_card_id)

            opp_mon = opp_act[0] if opp_act and opp_act[0] else {}
            opp_card_id = opp_mon.get('cardId', 0)
            opp_card = cards_db.get(opp_card_id)

            # Determine real attack damage
            atk_id = getattr(option, 'attackId', None) if not isinstance(option, dict) else option.get('attackId')
            if atk_id is None and my_card and getattr(my_card, 'attacks', None):
                atk_id = my_card.attacks[0]
            
            atk_obj = attacks_db.get(atk_id) if atk_id is not None else None
            base_dmg = getattr(atk_obj, 'damage', 90) if atk_obj else 90

            # Weakness & Resistance calculation
            my_etype = getattr(my_card, 'energyType', None) if my_card else None
            opp_weakness = getattr(opp_card, 'weakness', None) if opp_card else None
            opp_resist = getattr(opp_card, 'resistance', None) if opp_card else None

            calc_dmg = base_dmg
            if opp_weakness is not None and my_etype is not None and opp_weakness == my_etype:
                calc_dmg = base_dmg * 2
            elif opp_resist is not None and my_etype is not None and opp_resist == my_etype:
                calc_dmg = max(0, base_dmg - 30)

            # Ability modifications
            my_bench_cards = [cards_db.get(b.get('cardId', 0)) for b in me.get('bench', []) if b]
            opp_bench_cards = [cards_db.get(b.get('cardId', 0)) for b in opp.get('bench', []) if b]
            calc_dmg = VirtualGameSimulator._resolve_ability_modifiers(my_card, my_bench_cards, opp_card, opp_bench_cards, calc_dmg)

            # Apply damage to opponent active
            if opp_act and opp_act[0]:
                cur_hp = opp_mon.get('hp', getattr(opp_card, 'hp', 100) if opp_card else 100)
                new_hp = max(0, cur_hp - calc_dmg)
                opp_mon['hp'] = new_hp

                if new_hp == 0:
                    # Knockout achieved! Calculate exact prize cards taken
                    is_mega = getattr(opp_card, 'megaEx', False) or opp_mon.get('megaEx', False)
                    is_ex = getattr(opp_card, 'ex', False) or opp_mon.get('ex', False)
                    prizes_taken = 3 if is_mega else (2 if is_ex else 1)
                    
                    cur_prizes = me.get('prize', [])
                    rem_prizes = max(0, len(cur_prizes) - prizes_taken)
                    me['prize'] = [None] * rem_prizes
                    reward += 0.35 * prizes_taken

                    if rem_prizes == 0:
                        terminal = True
                        reward += 1.0  # Decisive Prize Victory!
                    else:
                        opp_bench = opp.get('bench', [])
                        if opp_bench:
                            opp['active'] = [opp_bench.pop(0)]
                        else:
                            terminal = True
                            reward += 1.0  # Decisive Bench-Out Victory!
                else:
                    # Partial damage reward
                    reward += 0.15 * min(1.0, calc_dmg / max(1, cur_hp))

            # Opponent 2-ply counter-retaliation anticipation
            if not terminal and opp.get('active'):
                retaliating_mon = opp['active'][0]
                ret_cid = retaliating_mon.get('cardId', 0) if isinstance(retaliating_mon, dict) else 0
                ret_card = cards_db.get(ret_cid)
                if ret_card and getattr(ret_card, 'attacks', None):
                    highest_opp_dmg = 0
                    # Energy check: opponent currently attached energies + at most 1 turn attachment
                    opp_cur_energies = len(retaliating_mon.get('energies', []) or [])
                    max_opp_energy = opp_cur_energies + 1
                    for o_aid in ret_card.attacks:
                        o_atk = attacks_db.get(o_aid)
                        if o_atk:
                            req_e = len(getattr(o_atk, 'energies', []) or [])
                            # Only calculate retaliation if opponent can legally afford the attack
                            if req_e <= max_opp_energy:
                                highest_opp_dmg = max(highest_opp_dmg, getattr(o_atk, 'damage', 0))
                    
                    if highest_opp_dmg > 0:
                        my_weakness = getattr(my_card, 'weakness', None) if my_card else None
                        ret_etype = getattr(ret_card, 'energyType', None)
                        if my_weakness is not None and ret_etype == my_weakness:
                            highest_opp_dmg *= 2
                        
                        my_hp = my_mon.get('hp', getattr(my_card, 'hp', 100) if my_card else 100)
                        rem_my_hp = max(0, my_hp - highest_opp_dmg)
                        my_mon['hp'] = rem_my_hp
                        if rem_my_hp == 0:
                            # Opponent revenge KO penalty
                            is_my_ex = getattr(my_card, 'ex', False) or getattr(my_card, 'megaEx', False)
                            reward -= 0.40 * (2 if is_my_ex else 1)

        # 2. EVOLVE (OptionType.EVOLVE or 9)
        elif opt_type in (OptionType.EVOLVE, 9):
            me_act = me.get('active', [])
            card_id = getattr(option, 'cardId', None) if not isinstance(option, dict) else option.get('cardId')
            evolved_card = cards_db.get(card_id) if card_id else None

            if me_act and me_act[0]:
                old_max = me_act[0].get('maxHp', 80)
                old_hp = me_act[0].get('hp', 80)
                new_max = getattr(evolved_card, 'hp', old_max + 90) if evolved_card else old_max + 90
                hp_boost = max(0, new_max - old_max)
                
                me_act[0]['hp'] = old_hp + hp_boost
                me_act[0]['maxHp'] = new_max
                me_act[0]['stage'] = 2 if (evolved_card and getattr(evolved_card, 'stage2', False)) else 1
                if evolved_card:
                    me_act[0]['cardId'] = evolved_card.cardId
                    me_act[0]['ex'] = getattr(evolved_card, 'ex', False)
                    me_act[0]['megaEx'] = getattr(evolved_card, 'megaEx', False)
            reward += 0.20 + (0.10 if (evolved_card and getattr(evolved_card, 'ex', False)) else 0.0)

        # 3. ATTACH (OptionType.ATTACH or 8)
        elif opt_type in (OptionType.ATTACH, 8):
            area = getattr(option, 'inPlayArea', None) if not isinstance(option, dict) else option.get('inPlayArea')
            in_play_idx = getattr(option, 'inPlayIndex', 0) if not isinstance(option, dict) else option.get('inPlayIndex', 0)
            target_mon = None
            is_active_target = (area in (4, 110) or area == 'Active' or area is None)

            me_act = me.get('active', [])
            me_bench = me.get('bench', [])

            if is_active_target:
                if me_act and me_act[0]:
                    target_mon = me_act[0]
            else:
                if me_bench and in_play_idx is not None and 0 <= in_play_idx < len(me_bench):
                    target_mon = me_bench[in_play_idx]

            if target_mon:
                energies = list(target_mon.get('energies', []) or [])
                t_cid = target_mon.get('cardId', 0) if isinstance(target_mon, dict) else 0
                t_card = cards_db.get(t_cid)
                max_req = 1
                if t_card and getattr(t_card, 'attacks', None):
                    for aid in t_card.attacks:
                        atk = attacks_db.get(aid)
                        if atk:
                            max_req = max(max_req, len(getattr(atk, 'energies', []) or []))

                cur_len = len(energies)
                was_saturated = (cur_len >= max_req)
                energies.append(1)
                target_mon['energies'] = energies

                if is_active_target:
                    if was_saturated:
                        reward -= 0.30  # Active over-saturation penalty
                    elif cur_len + 1 >= max_req:
                        reward += 0.35  # Active attack unlocked
                    else:
                        reward += 0.20  # Progressing active attack
                else:
                    # Bench target
                    act_is_ready = False
                    if me_act and me_act[0]:
                        act_cid = me_act[0].get('cardId', 0)
                        act_card = cards_db.get(act_cid)
                        act_e = len(me_act[0].get('energies', []) or [])
                        if act_card and getattr(act_card, 'attacks', None):
                            for aid in act_card.attacks:
                                a_obj = attacks_db.get(aid)
                                if a_obj and act_e >= len(getattr(a_obj, 'energies', []) or []):
                                    act_is_ready = True
                                    break
                    if was_saturated:
                        reward -= 0.30
                    elif act_is_ready:
                        if cur_len + 1 >= max_req:
                            reward += 0.40  # Full power on backup carry
                        else:
                            reward += 0.25  # Carry acceleration
                    else:
                        reward -= 0.15  # Neglecting active attacker
            else:
                reward += 0.05

        # 4. PLAY / BENCH (OptionType.PLAY or 7)
        elif opt_type in (OptionType.PLAY, 7):
            bench = me.get('bench', [])
            card_id = getattr(option, 'cardId', None) if not isinstance(option, dict) else option.get('cardId')
            b_card = cards_db.get(card_id) if card_id else None
            
            if len(bench) < 5:
                bench_hp = getattr(b_card, 'hp', 70) if b_card else 70
                bench.append({
                    'hp': bench_hp,
                    'maxHp': bench_hp,
                    'energies': [],
                    'cardId': card_id or 0,
                    'ex': getattr(b_card, 'ex', False) if b_card else False,
                    'megaEx': getattr(b_card, 'megaEx', False) if b_card else False,
                })
                me['bench'] = bench
            reward += 0.12  # Bench protection against donk

        # 5. RETREAT (OptionType.RETREAT or 12)
        elif opt_type in (OptionType.RETREAT, 12):
            bench = me.get('bench', [])
            act = me.get('active', [])
            if bench and act:
                old_act = act[0]
                my_c = cards_db.get(old_act.get('cardId', 0)) if isinstance(old_act, dict) else None
                ret_cost = getattr(my_c, 'retreatCost', 1) if my_c else 1
                
                # Check for free retreat abilities on active or bench
                has_free_retreat = False
                all_my_cards = [my_c] + [cards_db.get(b.get('cardId', 0)) for b in bench if b]
                for c in all_my_cards:
                    if c and getattr(c, 'skills', None):
                        for s in c.skills:
                            txt = (getattr(s, 'text', '') or '').lower()
                            if 'retreat' in txt and ('0' in txt or 'no retreat' in txt or 'free' in txt):
                                has_free_retreat = True
                                break
                if has_free_retreat:
                    ret_cost = 0

                # Discard retreat cost
                cur_e = old_act.get('energies', []) if isinstance(old_act, dict) else []
                if isinstance(old_act, dict):
                    old_act['energies'] = cur_e[ret_cost:] if len(cur_e) >= ret_cost else []

                # Promote fresh bench pokemon
                new_act = bench.pop(0)
                bench.append(old_act)
                me['active'] = [new_act]
                me['bench'] = bench

                # Reward tactical retreat if old active was low HP
                cur_hp = old_act.get('hp', 100) if isinstance(old_act, dict) else 100
                max_hp = old_act.get('maxHp', 100) if isinstance(old_act, dict) else 100
                if cur_hp < 0.4 * max_hp:
                    reward += 0.25  # Saved a Pokémon from fainting!
                else:
                    reward += 0.05

        return s, reward, terminal

    @staticmethod
    def generate_legal_virtual_options(cur_dict: Dict[str, Any], your_idx: int) -> List[VirtualOption]:
        """
        Synthesizes plausible forward legal actions from a simulated state dict.
        Enables multi-ply deep recursive tree expansion down through 10-16 plies.
        """
        from simulation.Decision_Engine import _get_cards, _get_attacks
        cards_db = _get_cards()
        attacks_db = _get_attacks()

        players = cur_dict.get('players', [{}, {}])
        if len(players) <= your_idx:
            return [VirtualOption(OptionType.END, name="Virtual End Turn")]

        me = players[your_idx]
        me_act = me.get('active', [])
        bench = me.get('bench', [])
        hand_cnt = me.get('handCount', 5)

        virtual_options: List[VirtualOption] = []

        # 1. Attack Options
        if me_act and me_act[0]:
            my_mon = me_act[0]
            my_card = cards_db.get(my_mon.get('cardId', 0))
            if my_card and getattr(my_card, 'attacks', None):
                for atk_id in my_card.attacks:
                    atk_obj = attacks_db.get(atk_id)
                    atk_name = getattr(atk_obj, 'name', f"Attack #{atk_id}") if atk_obj else f"Attack #{atk_id}"
                    virtual_options.append(VirtualOption(OptionType.ATTACK, card_id=my_card.cardId, attack_id=atk_id, name=atk_name))
            else:
                virtual_options.append(VirtualOption(OptionType.ATTACK, name="Virtual Basic Attack"))

        # 2. Energy Attachment Option
        if hand_cnt > 0 and (me_act or bench):
            virtual_options.append(VirtualOption(OptionType.ATTACH, name="Virtual Attach Energy"))

        # 3. Evolution Option
        if hand_cnt > 0 and me_act:
            virtual_options.append(VirtualOption(OptionType.EVOLVE, name="Virtual Evolution"))

        # 4. Bench Basic Pokemon Option
        if hand_cnt > 0 and len(bench) < 5:
            virtual_options.append(VirtualOption(OptionType.PLAY, name="Virtual Bench Pokemon"))

        # 5. Tactical Retreat Option
        if me_act and bench:
            virtual_options.append(VirtualOption(OptionType.RETREAT, name="Virtual Tactical Retreat"))

        # 6. End Turn / Pass
        virtual_options.append(VirtualOption(OptionType.END, name="Virtual Pass / End Turn"))

        return virtual_options
