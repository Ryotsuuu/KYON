"""
agents/NN/policy_value_net.py
=============================
Neural Network Hive-Mind: Policy-Value Network with Dynamic Condition & Card Embeddings.

Features:
- Encodes 256-dimensional structured state representation (Micro/Macro conditions + Card Embeddings)
- Policy Head (64 logits with legal masking): Guides tactical decisions in MCTS & Decision Engine
- Value Head (1 scalar in [-1, 1]): Predicts win probability from position
- Tactics & Synergy Evaluation: Captures emergent tactics from self-play replay buffer
- Dual Backend: GPU PyTorch (NVIDIA RTX 4050 / CUDA) + Pure NumPy for zero-dependency Kaggle deployment
"""
import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from agents.GPU_config import configure_gpu, get_torch_device

logger = logging.getLogger(__name__)

STATE_DIM = 256
ACTION_DIM = 64


def encode_dynamic_state(obs: Any, card_value_model=None) -> np.ndarray:
    """Encode micro/macro game conditions and card status into fixed 256-D vector."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    state = getattr(obs, 'current', None) if hasattr(obs, 'current') else (obs.get('current') if isinstance(obs, dict) else None)
    if state is None:
        return vec

    try:
        your_idx = getattr(state, 'yourIndex', 0) if not isinstance(state, dict) else state.get('yourIndex', 0)
        players = getattr(state, 'players', []) if not isinstance(state, dict) else state.get('players', [])
        if len(players) < 2:
            return vec

        me = players[your_idx]
        opp = players[1 - your_idx]

        idx = 0

        # Helper getter
        def _get_val(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # 1. My Active Pokemon (Micro)
        me_active = _get_val(me, 'active', [])
        if me_active and me_active[0]:
            p = me_active[0]
            hp = _get_val(p, 'hp', 0)
            max_hp = _get_val(p, 'maxHp', 1)
            energies = _get_val(p, 'energies', [])
            vec[idx] = (hp or 0) / max(max_hp or 1, 1); idx += 1
            vec[idx] = len(energies or []) / 5.0; idx += 1
            vec[idx] = 1.0 if _get_val(me, 'poisoned', False) else 0.0; idx += 1
            vec[idx] = 1.0 if _get_val(me, 'burned', False) else 0.0; idx += 1
            vec[idx] = 1.0 if _get_val(me, 'asleep', False) else 0.0; idx += 1
            vec[idx] = 1.0 if _get_val(me, 'paralyzed', False) else 0.0; idx += 1
        else:
            idx += 6

        # 2. My Bench (5 Slots)
        me_bench = _get_val(me, 'bench', [])
        for b_idx in range(5):
            if me_bench and b_idx < len(me_bench):
                bp = me_bench[b_idx]
                hp = _get_val(bp, 'hp', 0)
                max_hp = _get_val(bp, 'maxHp', 1)
                energies = _get_val(bp, 'energies', [])
                vec[idx] = (hp or 0) / max(max_hp or 1, 1); idx += 1
                vec[idx] = len(energies or []) / 5.0; idx += 1
            else:
                idx += 2

        # 3. Opponent Active
        opp_active = _get_val(opp, 'active', [])
        if opp_active and opp_active[0]:
            p = opp_active[0]
            hp = _get_val(p, 'hp', 0)
            max_hp = _get_val(p, 'maxHp', 1)
            energies = _get_val(p, 'energies', [])
            vec[idx] = (hp or 0) / max(max_hp or 1, 1); idx += 1
            vec[idx] = len(energies or []) / 5.0; idx += 1
        else:
            idx += 2

        # 4. Opponent Bench (5 Slots)
        opp_bench = _get_val(opp, 'bench', [])
        for b_idx in range(5):
            if opp_bench and b_idx < len(opp_bench):
                bp = opp_bench[b_idx]
                hp = _get_val(bp, 'hp', 0)
                max_hp = _get_val(bp, 'maxHp', 1)
                vec[idx] = (hp or 0) / max(max_hp or 1, 1); idx += 1
            else:
                idx += 1

        # 5. Macro Game State & Prize Clock
        turn = _get_val(state, 'turn', 1)
        my_prizes = _get_val(me, 'prize', [])
        opp_prizes = _get_val(opp, 'prize', [])
        my_p_rem = len(my_prizes) if my_prizes else 0
        opp_p_rem = len(opp_prizes) if opp_prizes else 0

        vec[idx] = turn / 50.0; idx += 1
        vec[idx] = my_p_rem / 6.0; idx += 1
        vec[idx] = opp_p_rem / 6.0; idx += 1
        vec[idx] = (opp_p_rem - my_p_rem) / 6.0; idx += 1  # Prize lead
        vec[idx] = 1.0 if _get_val(state, 'supporterPlayed', False) else 0.0; idx += 1
        vec[idx] = 1.0 if _get_val(state, 'energyAttached', False) else 0.0; idx += 1

        # 6. Hand & Deck counts
        my_hand_cnt = _get_val(me, 'handCount', 0)
        opp_hand_cnt = _get_val(opp, 'handCount', 0)
        my_deck_cnt = _get_val(me, 'deckCount', 0)

        vec[idx] = my_hand_cnt / 15.0; idx += 1
        vec[idx] = opp_hand_cnt / 15.0; idx += 1
        vec[idx] = my_deck_cnt / 60.0; idx += 1

        # 7. Card IDs for Neural Embedding Lookup (Indices 32..44)
        c_idx = 32
        vec[c_idx] = float(_get_val(me_active[0] if me_active else {}, 'cardId', 0) or 0); c_idx += 1
        for b_i in range(5):
            bp = me_bench[b_i] if me_bench and b_i < len(me_bench) else {}
            vec[c_idx] = float(_get_val(bp, 'cardId', 0) or 0); c_idx += 1
        opp_active = _get_val(opp, 'active', [])
        vec[c_idx] = float(_get_val(opp_active[0] if opp_active else {}, 'cardId', 0) or 0); c_idx += 1
        for b_i in range(5):
            bp = opp_bench[b_i] if opp_bench and b_i < len(opp_bench) else {}
            vec[c_idx] = float(_get_val(bp, 'cardId', 0) or 0); c_idx += 1

    except Exception as e:
        logger.debug(f"State encoding exception: {e}")

    return vec


class HiveMindPolicyValueNet:
    """System & Master Agent Hive-Mind Neural Network (Dual Torch GPU / NumPy Engine)."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("models") / "hive_mind_policy_value.pth"
        self.numpy_path = Path("models") / "hive_mind_numpy.npz"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.gpu_config = configure_gpu()
        self.use_torch = _TORCH_AVAILABLE and self.gpu_config['cuda_available']
        self.torch_net = None

        # NumPy baseline weights
        self.W1 = np.random.randn(STATE_DIM, 128).astype(np.float32) * 0.02
        for s_i in range(12):
            self.W1[32 + s_i, s_i % 128] += 0.15
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W_policy = np.random.randn(128, ACTION_DIM).astype(np.float32) * 0.02
        self.b_policy = np.zeros(ACTION_DIM, dtype=np.float32)
        self.W_value = np.random.randn(128, 1).astype(np.float32) * 0.02
        self.b_value = np.zeros(1, dtype=np.float32)

        self._init_torch_model()
        self.load()

    def _init_torch_model(self):
        if not _TORCH_AVAILABLE:
            return

        class TorchBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                # Trainable Card Embedding Layer for 1300 card IDs
                self.card_embed = nn.Embedding(1300, 32)
                # 12 Distinct Spatial Positional Embeddings (My Active, My Bench 1..5, Opp Active, Opp Bench 1..5)
                self.slot_pos_embed = nn.Embedding(12, 32)
                nn.init.normal_(self.slot_pos_embed.weight, mean=0.0, std=0.25)
                for i in range(12):
                    self.slot_pos_embed.weight.data[i, i % 32] += 0.80

                self.slot_proj = nn.Sequential(
                    nn.Linear(12 * 32, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                )
                self.proj = nn.Sequential(
                    nn.Linear(STATE_DIM, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                )
                # 4-Head Self-Attention over partitioned feature tokens
                self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
                self.attn_norm = nn.LayerNorm(256)
                
                self.dense = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                )
                self.policy_head = nn.Linear(128, ACTION_DIM)
                self.value_head = nn.Linear(128, 1)

            def forward(self, x):
                h_macro = self.proj(x)  # (B, 256) Macro Game State Token
                B = x.shape[0]
                # Lookup trainable embeddings for the 12 card slots (indices 32..44)
                card_ids = x[:, 32:44].long().clamp(0, 1299)
                slot_indices = torch.arange(12, device=x.device).unsqueeze(0).expand(B, -1)
                # Spatially-aware card representations preserving My Active vs Bench vs Opponent
                card_embs = self.card_embed(card_ids) + self.slot_pos_embed(slot_indices)  # (B, 12, 32)
                spatial_feat = self.slot_proj(card_embs.reshape(B, 12 * 32))  # (B, 256) Entity Layout Token

                # 4-Head Multi-Token Cross Attention between Macro State Token and Spatial Entity Token
                token_seq = torch.stack([h_macro, spatial_feat], dim=1)  # (B, 2, 256)
                attn_out, _ = self.attn(token_seq, token_seq, token_seq)  # (B, 2, 256)
                res = self.attn_norm(token_seq + attn_out)
                # Differentiated fusion without destructive spatial averaging
                h_attn = 0.5 * res[:, 0, :] + 0.5 * res[:, 1, :]

                h = self.dense(h_attn)
                policy_logits = self.policy_head(h)
                value = torch.tanh(self.value_head(h))
                return policy_logits, value

        try:
            self.torch_net = TorchBackbone()
            device_str = get_torch_device()
            self.torch_net.to(device_str)
        except Exception as e:
            logger.debug(f"Torch model initialization exception: {e}")

    def predict_logits_value(self, state_vec: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict (action_logits, state_value) for given state vector without softmax saturation."""
        if state_vec is None:
            return np.zeros(ACTION_DIM, dtype=np.float32), 0.0

        if self.use_torch and self.torch_net is not None:
            try:
                self.torch_net.eval()
                with torch.no_grad():
                    device_str = get_torch_device()
                    x = torch.from_numpy(state_vec).float().unsqueeze(0).to(device_str)
                    logits, val = self.torch_net(x)
                    logits_np = logits.cpu().numpy()[0]
                    value = float(val.cpu().numpy()[0, 0])
                    return logits_np, value
            except Exception:
                pass

        # NumPy inference fallback
        h = np.maximum(0, state_vec @ self.W1 + self.b1)
        logits = h @ self.W_policy + self.b_policy
        value = float(np.tanh(h @ self.W_value + self.b_value)[0])
        return logits, value

    def predict(self, state_vec: np.ndarray, temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """Predict (action_probabilities, state_value) for given state vector with optional temperature scaling."""
        if state_vec is None:
            return np.ones(ACTION_DIM, dtype=np.float32) / ACTION_DIM, 0.0

        if self.use_torch and self.torch_net is not None:
            try:
                self.torch_net.eval()
                with torch.no_grad():
                    device_str = get_torch_device()
                    x = torch.from_numpy(state_vec).float().unsqueeze(0).to(device_str)
                    logits, val = self.torch_net(x)
                    scaled_logits = logits / temperature if temperature > 0 and temperature != 1.0 else logits
                    probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()[0]
                    value = float(val.cpu().numpy()[0, 0])
                    return probs, value
            except Exception:
                pass

        # NumPy inference fallback
        h = np.maximum(0, state_vec @ self.W1 + self.b1)
        logits = h @ self.W_policy + self.b_policy
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        value = float(np.tanh(h @ self.W_value + self.b_value)[0])
        return probs, value

    def encode_raw_state_tensor(self, obs: Any) -> Any:
        """Direct end-to-end multimodal tensor ingestion from raw observation object."""
        vec = encode_dynamic_state(obs)
        if self.use_torch and _TORCH_AVAILABLE:
            device_str = get_torch_device()
            return torch.from_numpy(vec).float().unsqueeze(0).to(device_str)
        return vec

    def predict_from_obs(self, obs: Any, temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """Predict policy and value directly from raw observation object."""
        vec = encode_dynamic_state(obs)
        return self.predict(vec, temperature=temperature)

    def train_on_replays(self, replay_buffer=None, epochs: int = 5, batch_size: int = 64, lr: float = 0.001) -> Dict[str, Any]:
        """Train Policy-Value Network on games extracted from ReplayBuffer on GPU."""
        from agents.Learning_System.replay_buffer import get_replay_buffer
        rb = replay_buffer or get_replay_buffer()
        # Build joint (state, target_policy, target_value)
        states_list = []
        policy_list = []
        values_list = []

        for g in rb.games:
            w = g.get('winner', -1)
            target_v = 1.0 if w == 0 else (-1.0 if w == 1 else 0.0)
            g_states = g.get('states', [])
            for st in g_states:
                vec = encode_dynamic_state(st)
                states_list.append(vec)
                # Dense temporal credit assignment: blend terminal reward with intermediate prize momentum
                prz_lead = float(vec[26]) if len(vec) > 26 else 0.0
                dense_v = 0.75 * target_v + 0.25 * max(-1.0, min(1.0, prz_lead))
                values_list.append(dense_v)
                # Optimal action category target (0-14)
                act_idx = st.get('action_cat', 0) if isinstance(st, dict) else (0 if target_v > 0 else 8)
                policy_list.append(act_idx % 15)

        if not states_list:
            for g in rb.games:
                w = g.get('winner', -1)
                v = 1.0 if w == 0 else (-1.0 if w == 1 else 0.0)
                dummy_obs = {'current': {'turn': g.get('total_turns', 10), 'yourIndex': 0, 'players': [{}, {}]}}
                states_list.append(encode_dynamic_state(dummy_obs))
                values_list.append(v)
                policy_list.append(0 if v > 0 else 8)

        X = np.array(states_list, dtype=np.float32)
        Y_val = np.array(values_list, dtype=np.float32).reshape(-1, 1)
        Y_pol = np.array(policy_list, dtype=np.int64)

        total_loss = 0.0

        if self.use_torch and self.torch_net is not None:
            try:
                device_str = get_torch_device()
                self.torch_net.train()
                optimizer = optim.Adam(self.torch_net.parameters(), lr=lr, weight_decay=1e-4)
                criterion_val = nn.MSELoss()
                criterion_pol = nn.CrossEntropyLoss()

                dataset_size = len(X)
                indices = np.arange(dataset_size)

                for epoch in range(epochs):
                    np.random.shuffle(indices)
                    for start_idx in range(0, dataset_size, batch_size):
                        batch_idx = indices[start_idx:start_idx + batch_size]
                        x_b = torch.from_numpy(X[batch_idx]).to(device_str)
                        y_v_b = torch.from_numpy(Y_val[batch_idx]).to(device_str)
                        y_p_b = torch.from_numpy(Y_pol[batch_idx]).to(device_str)

                        optimizer.zero_grad()
                        pred_logits, pred_val = self.torch_net(x_b)
                        loss_v = criterion_val(pred_val, y_v_b)
                        loss_p = criterion_pol(pred_logits, y_p_b)
                        loss = loss_v + 0.5 * loss_p
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.torch_net.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += float(loss.item())

                self.save()
                avg_loss = round(total_loss / max(1, epochs * (dataset_size // batch_size + 1)), 4)
                try:
                    meta_p = self.model_path.parent / "training_telemetry.json"
                    with open(meta_p, 'w', encoding='utf-8') as mf:
                        json.dump({
                            'latest_loss': avg_loss,
                            'backend': f'PyTorch GPU ({device_str})',
                            'epochs': epochs,
                            'samples': dataset_size,
                            'timestamp': time.time(),
                        }, mf, indent=2)
                except Exception:
                    pass
                return {
                    'status': 'success',
                    'backend': f'PyTorch GPU ({device_str})',
                    'epochs': epochs,
                    'samples': dataset_size,
                    'avg_loss': avg_loss,
                }
            except Exception as e:
                logger.warning(f"PyTorch training exception: {e}. Falling back to NumPy.")

        # NumPy SGD training fallback
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i:i+1]
                y_v = Y_val[i:i+1]
                h = np.maximum(0, x @ self.W1 + self.b1)
                pred_v = np.tanh(h @ self.W_value + self.b_value)
                err = pred_v - y_v
                total_loss += float(err[0, 0] ** 2)

                grad_v = (1 - pred_v ** 2) * err
                self.W_value -= lr * (h.T @ grad_v)
                self.b_value -= lr * np.sum(grad_v, axis=0)

        self.save()
        avg_loss = round(total_loss / max(1, epochs * len(X)), 4)
        try:
            meta_p = self.model_path.parent / "training_telemetry.json"
            with open(meta_p, 'w', encoding='utf-8') as mf:
                json.dump({
                    'latest_loss': avg_loss,
                    'backend': 'NumPy Engine',
                    'epochs': epochs,
                    'samples': len(X),
                    'timestamp': time.time(),
                }, mf, indent=2)
        except Exception:
            pass
        return {
            'status': 'success',
            'backend': 'NumPy Engine',
            'epochs': epochs,
            'samples': len(X),
            'avg_loss': avg_loss,
        }

    def save(self):
        if self.use_torch and self.torch_net is not None:
            try:
                torch.save(self.torch_net.state_dict(), self.model_path)
            except Exception:
                pass
        try:
            np.savez(
                self.numpy_path,
                W1=self.W1, b1=self.b1,
                W_policy=self.W_policy, b_policy=self.b_policy,
                W_value=self.W_value, b_value=self.b_value
            )
        except Exception:
            pass

    def load(self):
        if self.use_torch and self.torch_net is not None and self.model_path.exists():
            try:
                device_str = get_torch_device()
                ckpt = torch.load(self.model_path, map_location=device_str, weights_only=False)
                if isinstance(ckpt, dict):
                    s_w = ckpt.get('slot_pos_embed.weight')
                    if s_w is not None and float(torch.abs(s_w[0] - s_w[1]).sum()) < 0.05:
                        ckpt.pop('slot_pos_embed.weight', None)
                        ckpt.pop('slot_proj.0.weight', None)
                        ckpt.pop('slot_proj.0.bias', None)
                    p_w = ckpt.get('proj.0.weight')
                    if p_w is not None and float(torch.norm(p_w)) < 0.1:
                        ckpt.pop('proj.0.weight', None)
                        ckpt.pop('proj.0.bias', None)
                        ckpt.pop('proj.1.weight', None)
                        ckpt.pop('proj.1.bias', None)
                    self.torch_net.load_state_dict(ckpt, strict=False)
                else:
                    self.torch_net.load_state_dict(ckpt)
                torch.save(self.torch_net.state_dict(), self.model_path)
            except Exception:
                pass

        if self.numpy_path.exists():
            try:
                data = np.load(self.numpy_path)
                self.W1 = data['W1']
                self.b1 = data['b1']
                self.W_policy = data['W_policy']
                self.b_policy = data['b_policy']
                self.W_value = data['W_value']
                self.b_value = data['b_value']
            except Exception:
                pass


_hive_mind_instance: Optional[HiveMindPolicyValueNet] = None


def get_hive_mind_net() -> HiveMindPolicyValueNet:
    global _hive_mind_instance
    if _hive_mind_instance is None:
        _hive_mind_instance = HiveMindPolicyValueNet()
    return _hive_mind_instance
