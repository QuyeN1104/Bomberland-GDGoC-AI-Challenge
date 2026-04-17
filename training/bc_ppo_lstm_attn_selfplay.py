"""BC + PPO training with (optional) spatial attention and opponent-pool self-play.

Variants:
- lstm:      MapEncoderNoPool -> map_vec + AuxMLP(aux) -> LSTM -> actor/critic
- attn:      MapBackbone -> tokens + 2D pos + Attn -> CLS -> + AuxMLP(aux) -> actor/critic
- attn_lstm: MapBackbone -> tokens + 2D pos + Attn -> CLS -> + AuxMLP(aux) -> LSTM -> actor/critic

Key differences vs training/bc_ppo_lstm.py:
- No dropout in map encoder.
- Avoid average pooling: no AdaptiveAvgPool2d in map encoder or SE blocks.
- Aux vector is embedded by a small MLP before concatenation.
- Optional opponent-pool self-play with simple ELO updates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union, Literal

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from engine import BomberEnv

try:
    from .bomber_shared import (
        AGENT_LOOKUP,
        NUM_ACTIONS,
        _make_agent,
        collect_demonstrations,
        encode_obs,
        normalize_opponent_names,
    )
    from .reward_02 import EpisodeRewardState, compute_reward_icec
    from .utils import plot_loss, plot_moving_average, plot_rewards
except ImportError:
    from bomber_shared import (
        AGENT_LOOKUP,
        NUM_ACTIONS,
        _make_agent,
        collect_demonstrations,
        encode_obs,
        normalize_opponent_names,
    )
    from reward_02 import EpisodeRewardState, compute_reward_icec
    from utils import plot_loss, plot_moving_average, plot_rewards


BC_BASE_CLASS_WEIGHTS = torch.tensor([0.3, 1.0, 1.0, 1.0, 1.0, 1.8], dtype=torch.float32)


def _csv_append(path: str, fieldnames: list[str], row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        safe_row = {k: row.get(k, "") for k in fieldnames}
        w.writerow(safe_row)


def _build_bc_class_weights(actions: np.ndarray, device: str) -> torch.Tensor:
    """Compute robust class weights from demo frequencies and keep a mild bomb boost."""
    counts = np.bincount(actions.astype(np.int64), minlength=NUM_ACTIONS).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / (NUM_ACTIONS * counts)
    inv = inv / np.mean(inv)
    data_weights = torch.from_numpy(inv).to(device)
    weights = data_weights * BC_BASE_CLASS_WEIGHTS.to(device)
    return torch.clamp(weights, min=0.25, max=4.0)


class AuxMLP(nn.Module):
    def __init__(self, aux_dim: int, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or max(embed_dim * 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(aux_dim),
            nn.Linear(aux_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.embed_dim = int(embed_dim)

    def forward(self, aux_x: torch.Tensor) -> torch.Tensor:
        return self.net(aux_x)


class _ResBlockNoPool(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        return self.act(out)


class MapBackboneNoPool(nn.Module):
    """CNN backbone that returns a feature map (no avg pooling, no SE, no dropout)."""

    def __init__(self, map_shape, base: int = 32, blocks_per_stage: int = 1):
        super().__init__()
        c, _, _ = map_shape

        def _make_stage(in_ch: int, out_ch: int, blocks: int, stride: int):
            layers = [_ResBlockNoPool(in_ch, out_ch, stride=stride)]
            for _ in range(blocks - 1):
                layers.append(_ResBlockNoPool(out_ch, out_ch, stride=1))
            return nn.Sequential(*layers)

        self.stem = nn.Sequential(
            nn.Conv2d(c, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(base, base, blocks=blocks_per_stage, stride=1)
        self.stage2 = _make_stage(base, base * 2, blocks=blocks_per_stage, stride=2)
        self.out_channels = base * 2

    def forward(self, map_x: torch.Tensor) -> torch.Tensor:
        x = self.stem(map_x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x


class MapEncoderNoPool(nn.Module):
    """Map backbone + flatten + projection (no avg pooling, no dropout)."""

    def __init__(self, map_shape, feat_dim: int = 128):
        super().__init__()
        self.backbone = MapBackboneNoPool(map_shape)
        self.proj = nn.LazyLinear(feat_dim)
        self.feat_dim = int(feat_dim)

    def forward(self, map_x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(map_x)  # (B, C2, H2, W2)
        flat = fmap.flatten(1)
        return F.relu(self.proj(flat), inplace=True)


class SpatialPositionalEmbedding(nn.Module):
    """Learned 2D row+col embedding for (H,W) token grids."""

    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 32):
        super().__init__()
        self.row = nn.Embedding(max_h, d_model)
        self.col = nn.Embedding(max_w, d_model)
        self.max_h = int(max_h)
        self.max_w = int(max_w)

    def forward(self, h: int, w: int, device) -> torch.Tensor:
        if h > self.max_h or w > self.max_w:
            raise ValueError(f"positional embedding too small: got {(h, w)} max={(self.max_h, self.max_w)}")
        r = torch.arange(h, device=device)
        c = torch.arange(w, device=device)
        rr = self.row(r)[:, None, :]  # (H,1,D)
        cc = self.col(c)[None, :, :]  # (1,W,D)
        return (rr + cc).reshape(h * w, -1)  # (L,D)


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = max(d_model * mlp_ratio, 64)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class SpatialAttentionEncoder(nn.Module):
    """Backbone -> tokens -> pos emb -> attn -> CLS vector (no avg pooling)."""

    def __init__(
        self,
        map_shape,
        d_model: int = 128,
        n_heads: int = 4,
        pos_max_h: int = 32,
        pos_max_w: int = 32,
    ):
        super().__init__()
        self.backbone = MapBackboneNoPool(map_shape)
        self.to_tokens = nn.Conv2d(self.backbone.out_channels, d_model, kernel_size=1, stride=1, padding=0, bias=True)
        self.pos = SpatialPositionalEmbedding(d_model=d_model, max_h=pos_max_h, max_w=pos_max_w)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = AttentionBlock(d_model=d_model, n_heads=n_heads)
        self.d_model = int(d_model)
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, map_x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(map_x)  # (B,C2,H2,W2)
        tok = self.to_tokens(fmap)   # (B,D,H2,W2)
        b, d, h, w = tok.shape
        tokens = tok.permute(0, 2, 3, 1).reshape(b, h * w, d)  # (B,L,D)
        pos = self.pos(h, w, device=map_x.device).unsqueeze(0)  # (1,L,D)
        tokens = tokens + pos
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # (B,1+L,D)
        x = self.attn(x)
        return x[:, 0, :]  # CLS


ModelVariant = Literal["lstm", "attn", "attn_lstm"]


class ActorCriticAttnLSTM(nn.Module):
    def __init__(
        self,
        map_shape,
        aux_dim: int,
        num_actions: int,
        variant: ModelVariant = "lstm",
        map_feat_dim: int = 128,
        aux_embed_dim: int = 32,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        attn_d_model: int = 128,
        attn_heads: int = 4,
        pos_max_h: int = 32,
        pos_max_w: int = 32,
    ):
        super().__init__()
        self.variant: ModelVariant = variant
        self.aux_dim = int(aux_dim)
        self.aux_embed = AuxMLP(aux_dim=aux_dim, embed_dim=aux_embed_dim)

        self.num_actions = int(num_actions)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)

        if variant == "lstm":
            self.map_enc = MapEncoderNoPool(map_shape, feat_dim=map_feat_dim)
            trunk_dim = self.map_enc.feat_dim + self.aux_embed.embed_dim
        else:
            self.map_enc = SpatialAttentionEncoder(
                map_shape, d_model=attn_d_model, n_heads=attn_heads, pos_max_h=pos_max_h, pos_max_w=pos_max_w
            )
            trunk_dim = self.map_enc.d_model + self.aux_embed.embed_dim

        self.use_lstm = variant in ("lstm", "attn_lstm")
        if self.use_lstm:
            self.lstm = nn.LSTM(trunk_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True)
            head_in = lstm_hidden
        else:
            self.lstm = None
            head_in = trunk_dim

        self.actor = nn.Linear(head_in, num_actions)
        self.critic = nn.Linear(head_in, 1)

    def init_hidden(self, batch_size: int, device):
        if not self.use_lstm:
            raise RuntimeError("init_hidden called but variant has no LSTM")
        z = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return z, z.clone()

    def _encode_step(self, map_x: torch.Tensor, aux_x: torch.Tensor) -> torch.Tensor:
        map_feat = self.map_enc(map_x)
        aux_feat = self.aux_embed(aux_x)
        return torch.cat([map_feat, aux_feat], dim=-1)

    def forward_bc(self, map_x: torch.Tensor, aux_x: torch.Tensor):
        """I.i.d. BC: fresh LSTM state per batch (if used)."""
        b = map_x.shape[0]
        x = self._encode_step(map_x, aux_x)
        if not self.use_lstm:
            h = x
        else:
            x = x.unsqueeze(1)
            h0, c0 = self.init_hidden(b, map_x.device)
            out, _ = self.lstm(x, (h0, c0))
            h = out.squeeze(1)
        return self.actor(h), self.critic(h).squeeze(-1)

    def forward_step(
        self,
        map_x: torch.Tensor,
        aux_x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Single timestep for N parallel envs. If LSTM is unused, hidden can be None."""
        x = self._encode_step(map_x, aux_x)
        if not self.use_lstm:
            h = x
            return self.actor(h), self.critic(h).squeeze(-1), None
        if hidden is None:
            hidden = self.init_hidden(map_x.shape[0], map_x.device)
        out, new_hidden = self.lstm(x.unsqueeze(1), hidden)
        h = out.squeeze(1)
        return self.actor(h), self.critic(h).squeeze(-1), new_hidden

    def forward_sequence(
        self,
        obs_map: torch.Tensor,
        obs_aux: torch.Tensor,
        dones: torch.Tensor | None,
    ):
        """Rollout (T,N,...) -> logits(T,N,A) and values(T,N)."""
        t_max, n_env = obs_map.shape[0], obs_map.shape[1]
        device = obs_map.device

        if not self.use_lstm:
            flat_map = obs_map.reshape(t_max * n_env, *obs_map.shape[2:])
            flat_aux = obs_aux.reshape(t_max * n_env, obs_aux.shape[-1])
            x = self._encode_step(flat_map, flat_aux)
            logits = self.actor(x).reshape(t_max, n_env, self.num_actions)
            values = self.critic(x).reshape(t_max, n_env)
            return logits, values

        if dones is None:
            raise ValueError("dones is required for LSTM variants")
        h_state, c_state = self.init_hidden(n_env, device)
        logits_list, values_list = [], []
        for t in range(t_max):
            if t > 0:
                d = dones[t - 1].float().view(1, n_env, 1)
                h_state = h_state * (1.0 - d)
                c_state = c_state * (1.0 - d)
            logits, val, (h_state, c_state) = self.forward_step(obs_map[t], obs_aux[t], (h_state, c_state))
            logits_list.append(logits)
            values_list.append(val)
        return torch.stack(logits_list, dim=0), torch.stack(values_list, dim=0)


def pretrain_bc(
    model: ActorCriticAttnLSTM,
    bc_data: dict,
    device: str,
    bc_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_ratio: float = 0.1,
):
    n = len(bc_data["action"])
    idx = np.random.permutation(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    weights = _build_bc_class_weights(bc_data["action"], device)
    print(f"  BC class weights: {weights.detach().cpu().numpy().round(3).tolist()}")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )
    best_val = float("inf")
    best_sd = None
    history: list[float] = []

    for epoch in range(bc_epochs):
        model.train()
        np.random.shuffle(train_idx)
        train_loss = train_n = 0
        for start in range(0, len(train_idx), batch_size):
            bi = train_idx[start:start + batch_size]
            m = torch.from_numpy(bc_data["map"][bi]).to(device)
            a = torch.from_numpy(bc_data["aux"][bi]).to(device)
            y = torch.from_numpy(bc_data["action"][bi]).to(device)
            logits, _ = model.forward_bc(m, a)
            loss = F.cross_entropy(logits, y, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bi)
            train_n += len(bi)
        train_loss /= max(train_n, 1)
        history.append(train_loss)

        model.eval()
        val_loss = val_n = 0
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                bi = val_idx[start:start + batch_size]
                m = torch.from_numpy(bc_data["map"][bi]).to(device)
                a = torch.from_numpy(bc_data["aux"][bi]).to(device)
                y = torch.from_numpy(bc_data["action"][bi]).to(device)
                logits, _ = model.forward_bc(m, a)
                loss = F.cross_entropy(logits, y, weight=weights)
                val_loss += loss.item() * len(bi)
                val_n += len(bi)
        val_loss /= max(val_n, 1)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  BC epoch {epoch + 1}/{bc_epochs} train={train_loss:.4f} val={val_loss:.4f}")

    if best_sd is not None:
        model.load_state_dict(best_sd)
        model.to(device)
    model.train()
    print(f"  BC done — best val loss: {best_val:.4f}")
    return history


def save_checkpoint(path: str, model: nn.Module, optimizer, meta: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = dict(meta)
    meta.setdefault("num_actions", NUM_ACTIONS)
    payload = {
        "model_state_dict": model.state_dict(),
        "meta": meta,
        "agent_type": "bc_ppo_lstm_attn_selfplay",
        "num_actions": int(meta["num_actions"]),
    }
    if meta.get("input_spec") is not None:
        payload["input_shape"] = meta["input_spec"]
        payload["input_spec"] = meta["input_spec"]
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"Saved checkpoint {path}")


def load_checkpoint(path: str, model: nn.Module, device: str, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint {path}")
    return ckpt.get("meta", {})


@dataclass
class EloState:
    rating: dict[str, float]
    k: float = 24.0

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update_pair(self, a: str, b: str, score_a: float):
        ra = float(self.rating.get(a, 1000.0))
        rb = float(self.rating.get(b, 1000.0))
        ea = self.expected(ra, rb)
        eb = 1.0 - ea
        sa = float(score_a)
        sb = 1.0 - sa
        self.rating[a] = ra + self.k * (sa - ea)
        self.rating[b] = rb + self.k * (sb - eb)


def ranks_from_alive(obs: dict) -> list[int]:
    """Rank 0 for alive, 1 for dead (ties allowed)."""
    alive = obs["players"][:, 2].astype(np.int8)
    return [0 if int(a) == 1 else 1 for a in alive.tolist()]


def elo_update_from_ranks(elo: EloState, ids: list[str], ranks: list[int]) -> None:
    """Pairwise ELO update from ranks (lower rank is better)."""
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[i] < ranks[j]:
                s_i = 1.0
            elif ranks[i] > ranks[j]:
                s_i = 0.0
            else:
                s_i = 0.5
            elo.update_pair(ids[i], ids[j], s_i)


class PolicySnapshotAgent:
    """Load a saved checkpoint and act greedily; keeps LSTM memory if applicable."""

    def __init__(self, agent_id: int, checkpoint_path: str, device: str | None = None):
        self.agent_id = int(agent_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        meta = ckpt.get("meta", {})
        input_spec = meta.get("input_spec") or ckpt.get("input_shape")
        if input_spec is None:
            raise ValueError(f"Checkpoint {checkpoint_path!r} missing input_spec")

        map_shape = tuple(input_spec[0])
        aux_dim = int(input_spec[1])
        num_actions = int(meta.get("num_actions", ckpt.get("num_actions", NUM_ACTIONS)))
        variant: ModelVariant = meta.get("model_variant", "lstm")
        self.variant = variant

        self.model = ActorCriticAttnLSTM(
            map_shape=map_shape,
            aux_dim=aux_dim,
            num_actions=num_actions,
            variant=variant,
            map_feat_dim=int(meta.get("map_feat_dim", 128)),
            aux_embed_dim=int(meta.get("aux_embed_dim", 32)),
            lstm_hidden=int(meta.get("lstm_hidden", 256)),
            lstm_layers=int(meta.get("lstm_layers", 1)),
            attn_d_model=int(meta.get("attn_d_model", 128)),
            attn_heads=int(meta.get("attn_heads", 4)),
            pos_max_h=int(meta.get("pos_max_h", 32)),
            pos_max_w=int(meta.get("pos_max_w", 32)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_memory(self):
        self._hidden = None

    def act(self, obs: dict, agent_ids: list[int]) -> int:
        map_state, aux_state = encode_obs(obs, [self.agent_id, *[i for i in agent_ids if i != self.agent_id]])
        m = torch.from_numpy(map_state).float().unsqueeze(0).to(self.device)
        a = torch.from_numpy(aux_state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model.use_lstm:
                if self._hidden is None:
                    self._hidden = self.model.init_hidden(1, self.device)
                logits, _, self._hidden = self.model.forward_step(m, a, self._hidden)
            else:
                logits, _, _ = self.model.forward_step(m, a, None)
        return int(logits.argmax(dim=-1).item())


def evaluate_and_update_elo(
    device: str,
    current_model: ActorCriticAttnLSTM,
    current_id: str,
    pool: list[tuple[str, str]],
    elo: EloState,
    n_games: int,
    seed: int,
    max_steps: int,
):
    """Run evaluation games and update ELO for current and sampled pool opponents.

    pool entries: (policy_id, checkpoint_path)
    """
    if not pool or n_games <= 0:
        return

    rng = np.random.default_rng(seed)
    current_model.eval()

    for gi in range(n_games):
        # Choose 3 opponents from pool with replacement (small pools OK).
        opp = [pool[int(rng.integers(0, len(pool)))] for _ in range(3)]
        opp_ids = [p[0] for p in opp]
        opp_agents = [PolicySnapshotAgent(agent_id=i + 1, checkpoint_path=p[1], device=device) for i, p in enumerate(opp)]

        env = BomberEnv(max_steps=max_steps, seed=int(rng.integers(0, 2**31 - 1)))
        obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for a in opp_agents:
            a.reset_memory()

        agent_ids = [0, 1, 2, 3]
        h = None
        if current_model.use_lstm:
            h = current_model.init_hidden(1, device)

        for _ in range(max_steps):
            m0, a0 = encode_obs(obs, agent_ids)
            m_t = torch.from_numpy(m0).float().unsqueeze(0).to(device)
            a_t = torch.from_numpy(a0).float().unsqueeze(0).to(device)
            with torch.no_grad():
                if current_model.use_lstm:
                    logits0, _, h = current_model.forward_step(m_t, a_t, h)
                else:
                    logits0, _, _ = current_model.forward_step(m_t, a_t, None)
            act_list = [None] * 4
            act_list[0] = int(logits0.argmax(dim=-1).item())
            for idx, oa in enumerate(opp_agents, start=1):
                act_list[idx] = oa.act(obs, agent_ids)
            obs, term, trunc = env.step(act_list)
            if term or trunc:
                break

        ids = [current_id, *opp_ids]
        ranks = ranks_from_alive(obs)
        elo_update_from_ranks(elo, ids, ranks)

    current_model.train()


def train_bc_ppo_attn_selfplay(
    user_id: int = 0,
    expert_type: str = "tactical",
    enemy_type: Union[str, Sequence[str]] = "simple",
    demo_episodes: int = 100,
    bc_epochs: int = 15,
    ppo_updates: int = 500,
    ppo_steps: int = 128,
    parallel_envs: int = 4,
    max_steps: int = 500,
    seed: int = 86,
    model_variant: ModelVariant = "lstm",
    map_feat_dim: int = 128,
    aux_embed_dim: int = 32,
    attn_d_model: int = 128,
    attn_heads: int = 4,
    pos_max_h: int = 32,
    pos_max_w: int = 32,
    lstm_hidden: int = 128,
    lstm_layers: int = 1,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.03,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    save_model: bool = True,
    load_checkpoint_path: str | None = None,
    skip_bc: bool = False,
    device: str | None = None,
    reward_log_episodes: int = 3,
    shuffle_enemy_types: bool = True,
    p_selfplay: float = 0.25,
    snapshot_interval: int = 50,
    pool_max_size: int = 20,
    eval_interval: int = 100,
    eval_games: int = 8,
    elo_k: float = 24.0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path(__file__).resolve().parent.parent  # repo root

    demo_opp_ids = [i for i in range(4) if i != 0]
    enemy_type_tag = "_".join(normalize_opponent_names(enemy_type, demo_opp_ids))

    bc_data, _, input_spec = collect_demonstrations(
        expert_type=expert_type,
        opponent_type=enemy_type,
        num_episodes=demo_episodes,
        max_steps=max_steps,
        seed=seed,
        augment=True,
        store_dqfd_buffer=False,
        reward_fn=None,
    )
    if len(bc_data["action"]) == 0:
        print("No BC data — increase demo_episodes or weaken opponents.")
        return

    map_shape = tuple(input_spec[0])
    aux_dim = int(input_spec[1])
    model = ActorCriticAttnLSTM(
        map_shape=map_shape,
        aux_dim=aux_dim,
        num_actions=NUM_ACTIONS,
        variant=model_variant,
        map_feat_dim=map_feat_dim,
        aux_embed_dim=aux_embed_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        attn_d_model=attn_d_model,
        attn_heads=attn_heads,
        pos_max_h=pos_max_h,
        pos_max_w=pos_max_w,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    tag = f"bcppo_{model_variant}_{expert_type}_{enemy_type_tag}_p{parallel_envs}_s{seed}"
    out_dir = str(base_dir / "ckpts" / tag)
    pool_dir = str(Path(out_dir) / "pool")
    os.makedirs(pool_dir, exist_ok=True)

    ppo_csv_path = str(Path(out_dir) / "metrics_ppo.csv")
    eval_csv_path = str(Path(out_dir) / "metrics_eval_elo.csv")
    ppo_fields = [
        "update",
        "model_variant",
        "pi_loss",
        "v_loss",
        "entropy",
        "total_obj",
        "rollout_return_mean",
        "reward_mean",
        "reward_std",
        "pool_size",
        "elo_current",
    ]
    eval_fields = [
        "update",
        "model_variant",
        "eval_games",
        "pool_size",
        "elo_current",
        "elo_top5",
    ]

    bc_loss_history: list[float] = []
    if load_checkpoint_path:
        load_checkpoint(load_checkpoint_path, model, device, optimizer)
        model.train()
    if not skip_bc:
        print(f"Phase 1: Behavioral cloning (variant={model_variant})")
        bc_loss_history = pretrain_bc(model, bc_data, device, bc_epochs=bc_epochs)
        if save_model:
            save_checkpoint(
                str(Path(out_dir) / "after_bc.pth"),
                model,
                None,
                {
                    "input_spec": input_spec,
                    "model_variant": model_variant,
                    "map_feat_dim": map_feat_dim,
                    "aux_embed_dim": aux_embed_dim,
                    "attn_d_model": attn_d_model,
                    "attn_heads": attn_heads,
                    "pos_max_h": pos_max_h,
                    "pos_max_w": pos_max_w,
                    "lstm_hidden": lstm_hidden,
                    "lstm_layers": lstm_layers,
                },
            )

    enemy_ids = [i for i in range(4) if i != user_id]
    enemy_names = list(normalize_opponent_names(enemy_type, enemy_ids))
    if shuffle_enemy_types:
        np.random.default_rng(seed + 90208).shuffle(enemy_names)
    scripted_enemy_agents = [_make_agent(name, agent_id=i) for name, i in zip(enemy_names, enemy_ids)]
    agent_ids = [user_id, *enemy_ids]

    n_env = max(1, int(parallel_envs))
    envs = [BomberEnv(max_steps=max_steps, seed=seed + i) for i in range(n_env)]
    rng = np.random.default_rng(seed + 90210)

    # Opponent pool and ELO
    pool: list[tuple[str, str]] = []
    elo = EloState(rating={}, k=float(elo_k))
    current_policy_id = "current"

    ep_reward_sum = [0.0] * n_env
    ep_comp_sum: list[defaultdict[str, float]] = [defaultdict(float) for _ in range(n_env)]
    ep_log_this = [False] * n_env
    reward_log_done = 0

    env_opponents: list[list[object]] = [[] for _ in range(n_env)]

    def _choose_enemy(eid: int):
        if pool and (rng.random() < float(p_selfplay)):
            # Sample opponents with mild ELO bias: softmax over ratings.
            ratings = np.array([elo.rating.get(pid, 1000.0) for pid, _ in pool], dtype=np.float32)
            p = np.exp((ratings - ratings.max()) / 200.0)
            p = p / p.sum()
            idx = int(rng.choice(len(pool), p=p))
            pid, ckpt_path = pool[idx]
            return PolicySnapshotAgent(agent_id=eid, checkpoint_path=ckpt_path, device=device)
        # Fall back to scripted for this id
        for a in scripted_enemy_agents:
            if int(a.agent_id) == int(eid):
                return a
        raise RuntimeError(f"missing scripted agent for id {eid}")

    def reset_env_state(ei: int):
        obs = envs[ei].reset(seed=int(rng.integers(0, 2**31 - 1)))
        m, a = encode_obs(obs, agent_ids)
        ep = EpisodeRewardState()
        prev = None
        ep_reward_sum[ei] = 0.0
        ep_comp_sum[ei].clear()
        ep_log_this[ei] = reward_log_episodes > 0 and reward_log_done < reward_log_episodes
        # Choose opponents for this episode and reset their memory if needed.
        env_opponents[ei] = [_choose_enemy(eid) for eid in enemy_ids]
        for opp in env_opponents[ei]:
            if hasattr(opp, "reset_memory"):
                opp.reset_memory()
        return obs, m, a, ep, prev

    obs_l, map_l, aux_l, ep_l, prev_l = zip(*[reset_env_state(i) for i in range(n_env)])
    obs_l = list(obs_l)
    map_l = list(map_l)
    aux_l = list(aux_l)
    ep_l = list(ep_l)
    prev_l = list(prev_l)

    c, hh, ww = map_l[0].shape
    model.train()

    print(
        f"Phase 2: PPO variant={model_variant} device={device} parallel_envs={n_env} "
        f"steps/rollout={ppo_steps}"
    )

    ppo_loss_history: list[float] = []
    reward_history: list[float] = []
    rollout_return_history: list[float] = []

    pbar = tqdm(range(ppo_updates), desc=f"PPO({model_variant})")
    for upd in pbar:
        stor_m = np.zeros((ppo_steps, n_env, c, hh, ww), dtype=np.float32)
        stor_a = np.zeros((ppo_steps, n_env, aux_dim), dtype=np.float32)
        stor_act = np.zeros((ppo_steps, n_env), dtype=np.int64)
        stor_logp = np.zeros((ppo_steps, n_env), dtype=np.float32)
        stor_rew = np.zeros((ppo_steps, n_env), dtype=np.float32)
        stor_done = np.zeros((ppo_steps, n_env), dtype=np.float32)
        stor_val = np.zeros((ppo_steps, n_env), dtype=np.float32)

        h_s = c_s = None
        if model.use_lstm:
            h_s, c_s = model.init_hidden(n_env, device)

        for t in range(ppo_steps):
            m_t = torch.from_numpy(np.stack(map_l)).to(device)
            a_t = torch.from_numpy(np.stack(aux_l)).to(device)

            if model.use_lstm and t > 0:
                dprev = torch.from_numpy(stor_done[t - 1]).float().to(device).view(1, n_env, 1)
                h_s = h_s * (1.0 - dprev)
                c_s = c_s * (1.0 - dprev)

            logits, val, hc = model.forward_step(m_t, a_t, (h_s, c_s) if model.use_lstm else None)
            if model.use_lstm:
                h_s, c_s = hc  # type: ignore[misc]

            dist = Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)

            stor_m[t] = np.stack(map_l)
            stor_a[t] = np.stack(aux_l)
            stor_act[t] = actions.detach().cpu().numpy()
            stor_logp[t] = logp.detach().cpu().numpy()
            stor_val[t] = val.detach().cpu().numpy()

            for n in range(n_env):
                act_list = [None] * 4
                act_list[user_id] = int(stor_act[t, n])
                for opp in env_opponents[n]:
                    if isinstance(opp, PolicySnapshotAgent):
                        act_list[int(opp.agent_id)] = opp.act(obs_l[n], agent_ids)
                    else:
                        act_list[opp.agent_id] = opp.act(obs_l[n])
                next_obs, term, trunc = envs[n].step(act_list)
                done = term or trunc

                if ep_log_this[n]:
                    if reward_log_done < reward_log_episodes:
                        comp_buf: dict[str, float] = {}
                        r, ep_l[n] = compute_reward_icec(prev_l[n], next_obs, user_id, ep_l[n], out_components=comp_buf)
                        for k, v in comp_buf.items():
                            if k not in ("dense_decay", "dense_applied"):
                                ep_comp_sum[n][k] += float(v)
                    else:
                        r, ep_l[n] = compute_reward_icec(prev_l[n], next_obs, user_id, ep_l[n])
                    ep_reward_sum[n] += r
                else:
                    r, ep_l[n] = compute_reward_icec(prev_l[n], next_obs, user_id, ep_l[n])

                stor_rew[t, n] = r
                stor_done[t, n] = float(done)
                if done:
                    if ep_log_this[n] and reward_log_done < reward_log_episodes:
                        print(
                            f"[reward components] finished_episode={reward_log_done + 1}/"
                            f"{reward_log_episodes} env={n} return={ep_reward_sum[n]:.4f}"
                        )
                        keys = sorted(ep_comp_sum[n].keys(), key=lambda x: (x == "total", x))
                        for k in keys:
                            print(f"    {k}: {ep_comp_sum[n][k]:.6f}")
                        reward_log_done += 1
                    obs_l[n], map_l[n], aux_l[n], ep_l[n], prev_l[n] = reset_env_state(n)
                else:
                    obs_l[n] = next_obs
                    map_l[n], aux_l[n] = encode_obs(next_obs, agent_ids)
                    prev_l[n] = next_obs

        reward_history.extend(stor_rew.reshape(-1).tolist())
        rollout_return_history.append(float(np.mean(stor_rew.sum(axis=0))))

        with torch.no_grad():
            m_last = torch.from_numpy(np.stack(map_l)).to(device)
            a_last = torch.from_numpy(np.stack(aux_l)).to(device)
            if model.use_lstm:
                _, v_last, _ = model.forward_step(m_last, a_last, (h_s, c_s))  # type: ignore[arg-type]
            else:
                _, v_last, _ = model.forward_step(m_last, a_last, None)

        rew_t = torch.from_numpy(stor_rew).to(device)
        done_t = torch.from_numpy(stor_done).to(device)
        val_t = torch.from_numpy(stor_val).to(device)
        old_logp = torch.from_numpy(stor_logp).to(device)

        next_v = v_last * (1.0 - done_t[-1])
        advantages = torch.zeros_like(rew_t)
        last_gae = torch.zeros(n_env, device=device)
        for t in reversed(range(ppo_steps)):
            nv = next_v if t == ppo_steps - 1 else val_t[t + 1]
            nonterm = 1.0 - done_t[t]
            delta = rew_t[t] + gamma * nv * nonterm - val_t[t]
            last_gae = delta + gamma * gae_lambda * nonterm * last_gae
            advantages[t] = last_gae
        returns = advantages + val_t

        adv_norm = advantages.clone()
        adv_norm = (adv_norm - adv_norm.mean()) / (adv_norm.std() + 1e-8)

        obs_map_t = torch.from_numpy(stor_m).to(device)
        obs_aux_t = torch.from_numpy(stor_a).to(device)
        act_t = torch.from_numpy(stor_act).long().to(device)

        envs_per_mb = max(1, min(n_env, minibatch_size // max(1, ppo_steps)))

        pi_loss = v_loss = ent_scalar = torch.zeros((), device=device)
        for _ in range(ppo_epochs):
            env_order = np.random.permutation(n_env)
            n_mb = 0
            pi_acc = v_acc = ent_acc = 0.0
            for s in range(0, n_env, envs_per_mb):
                idx = env_order[s: s + envs_per_mb]
                idx_t = torch.tensor(idx, device=device, dtype=torch.long)

                mb_map = obs_map_t.index_select(1, idx_t)
                mb_aux = obs_aux_t.index_select(1, idx_t)
                mb_done = done_t.index_select(1, idx_t)
                mb_act = act_t.index_select(1, idx_t)
                mb_old_logp = old_logp.index_select(1, idx_t)
                mb_adv = adv_norm.index_select(1, idx_t)
                mb_ret = returns.index_select(1, idx_t)

                Tm, K = mb_map.shape[0], mb_map.shape[1]
                logits_full, values_full = model.forward_sequence(mb_map, mb_aux, mb_done if model.use_lstm else None)
                dist = Categorical(logits=logits_full.reshape(-1, NUM_ACTIONS))
                new_logp = dist.log_prob(mb_act.reshape(-1)).reshape(Tm, K)
                entropy = dist.entropy().reshape(Tm, K).mean()
                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(values_full, mb_ret)
                loss = pi_loss + vf_coef * v_loss - ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                n_mb += 1
                pi_acc += pi_loss.item()
                v_acc += v_loss.item()
                ent_acc += entropy.item()

            pi_loss = torch.tensor(pi_acc / max(n_mb, 1), device=device)
            v_loss = torch.tensor(v_acc / max(n_mb, 1), device=device)
            ent_scalar = torch.tensor(ent_acc / max(n_mb, 1), device=device)

        total_obj = pi_loss + vf_coef * v_loss - ent_coef * ent_scalar
        ppo_loss_history.append(float(total_obj.item()))
        pbar.set_postfix(pi=f"{pi_loss.item():.3f}", v=f"{v_loss.item():.3f}", ent=f"{ent_scalar.item():.3f}")

        # Metrics CSV (every update)
        try:
            rew_flat = stor_rew.reshape(-1)
            _csv_append(
                ppo_csv_path,
                ppo_fields,
                {
                    "update": int(upd + 1),
                    "model_variant": str(model_variant),
                    "pi_loss": float(pi_loss.item()),
                    "v_loss": float(v_loss.item()),
                    "entropy": float(ent_scalar.item()),
                    "total_obj": float(total_obj.item()),
                    "rollout_return_mean": float(rollout_return_history[-1]) if rollout_return_history else "",
                    "reward_mean": float(np.mean(rew_flat)) if rew_flat.size else "",
                    "reward_std": float(np.std(rew_flat)) if rew_flat.size else "",
                    "pool_size": int(len(pool)),
                    "elo_current": float(elo.rating.get(current_policy_id, 1000.0)),
                },
            )
        except Exception as e:
            print(f"[warn] failed writing PPO metrics csv: {e}")

        # Periodic snapshot to pool
        if save_model and snapshot_interval > 0 and (upd + 1) % snapshot_interval == 0:
            snap_id = f"upd{upd+1}"
            snap_path = str(Path(pool_dir) / f"{snap_id}.pth")
            save_checkpoint(
                snap_path,
                model,
                None,
                {
                    "input_spec": input_spec,
                    "model_variant": model_variant,
                    "map_feat_dim": map_feat_dim,
                    "aux_embed_dim": aux_embed_dim,
                    "attn_d_model": attn_d_model,
                    "attn_heads": attn_heads,
                    "pos_max_h": pos_max_h,
                    "pos_max_w": pos_max_w,
                    "lstm_hidden": lstm_hidden,
                    "lstm_layers": lstm_layers,
                },
            )
            pool.append((snap_id, snap_path))
            elo.rating.setdefault(snap_id, 1000.0)
            if len(pool) > int(pool_max_size):
                # Drop oldest snapshot.
                old_id, _old_path = pool.pop(0)
                elo.rating.pop(old_id, None)

        # Periodic ELO evaluation
        if eval_interval > 0 and (upd + 1) % eval_interval == 0:
            elo.rating.setdefault(current_policy_id, 1000.0)
            evaluate_and_update_elo(
                device=device,
                current_model=model,
                current_id=current_policy_id,
                pool=pool,
                elo=elo,
                n_games=int(eval_games),
                seed=seed + 70000 + upd,
                max_steps=max_steps,
            )
            if elo.rating:
                top = sorted(elo.rating.items(), key=lambda kv: kv[1], reverse=True)[:5]
                top_str = ", ".join(f"{k}:{v:.0f}" for k, v in top)
                print(f"[ELO] top: {top_str}")
                # Eval/ELO metrics CSV (every evaluation)
                try:
                    _csv_append(
                        eval_csv_path,
                        eval_fields,
                        {
                            "update": int(upd + 1),
                            "model_variant": str(model_variant),
                            "eval_games": int(eval_games),
                            "pool_size": int(len(pool)),
                            "elo_current": float(elo.rating.get(current_policy_id, 1000.0)),
                            "elo_top5": top_str,
                        },
                    )
                except Exception as e:
                    print(f"[warn] failed writing eval/ELO metrics csv: {e}")

    os.makedirs(out_dir, exist_ok=True)
    if bc_loss_history:
        plot_loss(bc_loss_history, save_path=str(Path(out_dir) / f"{tag}_bc_loss.png"))
    plot_loss(ppo_loss_history, save_path=str(Path(out_dir) / f"{tag}_ppo_loss.png"))
    plot_rewards(reward_history, save_path=str(Path(out_dir) / f"{tag}_rewards.png"))
    plot_moving_average(rollout_return_history, window_size=10, save_path=str(Path(out_dir) / f"{tag}_moving_avg.png"))

    if save_model:
        save_checkpoint(
            str(Path(out_dir) / "final.pth"),
            model,
            optimizer,
            {
                "input_spec": input_spec,
                "model_variant": model_variant,
                "map_feat_dim": map_feat_dim,
                "aux_embed_dim": aux_embed_dim,
                "attn_d_model": attn_d_model,
                "attn_heads": attn_heads,
                "pos_max_h": pos_max_h,
                "pos_max_w": pos_max_w,
                "lstm_hidden": lstm_hidden,
                "lstm_layers": lstm_layers,
                "parallel_envs": n_env,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC + PPO (optional spatial attention) + opponent-pool self-play")
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--user_id", type=int, default=0)
    parser.add_argument("--expert_type", type=str, default="tactical", choices=list(AGENT_LOOKUP.keys()))
    parser.add_argument(
        "--enemy_type",
        nargs="+",
        default=["simple"],
        metavar="TYPE",
        choices=list(AGENT_LOOKUP.keys()),
        help="One type (broadcast to all 3 bots) or three types: one per enemy id.",
    )
    parser.add_argument("--demo_episodes", type=int, default=500)
    parser.add_argument("--bc_epochs", type=int, default=12)
    parser.add_argument("--ppo_updates", type=int, default=1000)
    parser.add_argument("--ppo_steps", type=int, default=256)
    parser.add_argument("--parallel_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=256)

    parser.add_argument("--model_variant", type=str, default="lstm", choices=["lstm", "attn", "attn_lstm"])
    parser.add_argument("--map_feat_dim", type=int, default=128)
    parser.add_argument("--aux_embed_dim", type=int, default=32)
    parser.add_argument("--attn_d_model", type=int, default=128)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--pos_max_h", type=int, default=32)
    parser.add_argument("--pos_max_w", type=int, default=32)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.04)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)

    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--skip_bc", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reward-log-episodes", type=int, default=3)
    parser.add_argument("--no-shuffle-enemy-types", action="store_true")

    parser.add_argument("--p_selfplay", type=float, default=0.25)
    parser.add_argument("--snapshot_interval", type=int, default=50)
    parser.add_argument("--pool_max_size", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_games", type=int, default=8)
    parser.add_argument("--elo_k", type=float, default=24.0)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_bc_ppo_attn_selfplay(
        user_id=args.user_id,
        expert_type=args.expert_type,
        enemy_type=tuple(args.enemy_type) if len(args.enemy_type) != 1 else args.enemy_type[0],
        demo_episodes=args.demo_episodes,
        bc_epochs=args.bc_epochs,
        ppo_updates=args.ppo_updates,
        ppo_steps=args.ppo_steps,
        parallel_envs=args.parallel_envs,
        max_steps=args.max_steps,
        seed=args.seed,
        model_variant=args.model_variant,
        map_feat_dim=args.map_feat_dim,
        aux_embed_dim=args.aux_embed_dim,
        attn_d_model=args.attn_d_model,
        attn_heads=args.attn_heads,
        pos_max_h=args.pos_max_h,
        pos_max_w=args.pos_max_w,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        save_model=args.save_model,
        load_checkpoint_path=args.load_checkpoint,
        skip_bc=args.skip_bc,
        device=args.device,
        reward_log_episodes=args.reward_log_episodes,
        shuffle_enemy_types=not args.no_shuffle_enemy_types,
        p_selfplay=args.p_selfplay,
        snapshot_interval=args.snapshot_interval,
        pool_max_size=args.pool_max_size,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        elo_k=args.elo_k,
    )
