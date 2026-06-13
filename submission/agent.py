"""DQN Agent v2 — Double Dueling DQN + PER + Noisy Nets + N-step returns.

Upgrades over v1:
  - Double DQN (decouple action selection / evaluation)
  - Dueling architecture (separate V and A streams)
  - Noisy Networks (learned exploration, no epsilon schedule)
  - Prioritized Experience Replay with SumTree
  - N-step returns (n=3) for faster reward propagation
  - Huber loss + gradient clipping for stability
  - Soft target updates (Polyak averaging)
  - Enhanced observation: 13 spatial channels + 7 scalars
"""
from pathlib import Path
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Local imports (available at both training and inference time)
from model import DuelingDQN
from obs_encoder import encode_obs


# ──────────────────── Submission Agent (mandatory) ──────────────────────
class Agent:
    """Eval Agent with integrated Action Masking. NoisyNet is explicitly disabled."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cpu")
        self.q_net = None

        ckpt_path = Path(__file__).parent / "model.pth"
        if ckpt_path.exists():
            self._load(str(ckpt_path))
        else:
            for f in Path(__file__).parent.glob("*.pth"):
                self._load(str(f)); break

    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        spec = ckpt.get("input_spec", ckpt.get("input_shape", ckpt["input_dim"]))
        ms = tuple(spec[0])
        ad = int(spec[1])
        na = ckpt["num_actions"]
        has_noisy_keys = "value.0.weight_mu" in ckpt["model_state_dict"]
        noisy = ckpt.get("noisy", has_noisy_keys)
        if has_noisy_keys or "value.0.weight" in ckpt["model_state_dict"]:
            self.q_net = DuelingDQN(ms, ad, na, noisy=noisy)
        
        self.q_net.load_state_dict(ckpt["model_state_dict"])
        self.q_net.to(self.device)
        
        # VÔ HIỆU HOÁ NOISE CHO QUÁ TRÌNH THI ĐẤU/ĐÁNH GIÁ (TUYỆT ĐỐI QUAN TRỌNG)
        self.q_net.eval()

    def _get_valid_moves(self, map_state):
        pos_ch = map_state[5]
        pos = np.argwhere(pos_ch > 0.5)
        if len(pos) == 0:
            return {1, 2, 3, 4}

        ax, ay = int(pos[0][0]), int(pos[0][1])
        H, W = map_state.shape[1], map_state.shape[2]

        valid = set()
        for action, (dx, dy) in {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.items():
            nx, ny = ax + dx, ay + dy
            if not (0 < nx < H - 1 and 0 < ny < W - 1):
                continue
            if map_state[1, nx, ny] > 0.5 or map_state[2, nx, ny] > 0.5:
                continue
            if map_state[7, nx, ny] > 0.01:
                continue
            valid.add(action)
        return valid

    def act(self, obs):
        try:
            ms, axs = encode_obs(obs, [self.agent_id])
            mt = torch.from_numpy(ms).unsqueeze(0).to(self.device)
            at = torch.from_numpy(axs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_net(mt, at).squeeze(0)
                
                valid_moves = self._get_valid_moves(ms)
                mask = torch.ones(6, dtype=torch.bool, device=self.device)
                
                for act_idx in [1, 2, 3, 4]:
                    if act_idx not in valid_moves:
                        mask[act_idx] = False
                        
                has_bombs = float(axs[0]) > 0
                if not has_bombs:
                    mask[5] = False
                    
                q_values[~mask] = -float('inf')
                self.last_q_values = q_values.cpu().numpy().tolist()
                return q_values.argmax().item()
                
        except Exception:
            self.last_q_values = None
            return 0


