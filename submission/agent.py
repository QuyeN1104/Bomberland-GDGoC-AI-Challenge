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
    """DQN v2 agent for competition submission."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cpu")
        self.q_net = None

        ckpt_path = Path(__file__).parent / "model.pth"
        if ckpt_path.exists():
            self._load(str(ckpt_path))
        else:
            # Fallback: look for any .pth in same dir
            for f in Path(__file__).parent.glob("*.pth"):
                self._load(str(f)); break

    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        spec = ckpt.get("input_spec", ckpt.get("input_shape", ckpt["input_dim"]))
        ms = tuple(spec[0])
        ad = int(spec[1])
        na = ckpt["num_actions"]
        # Auto-detect noisy from state_dict keys (weight_mu → NoisyLinear)
        has_noisy_keys = "value.0.weight_mu" in ckpt["model_state_dict"]
        noisy = ckpt.get("noisy", has_noisy_keys)
        # Detect old vs new architecture
        if has_noisy_keys or "value.0.weight" in ckpt["model_state_dict"]:
            self.q_net = DuelingDQN(ms, ad, na, noisy=noisy)
        
        self.q_net.load_state_dict(ckpt["model_state_dict"])
        self.q_net.to(self.device)
        self.q_net.eval()

    def act(self, obs):
        try:
            ms, axs = encode_obs(obs, [self.agent_id])
            mt = torch.from_numpy(ms).unsqueeze(0).to(self.device)
            at = torch.from_numpy(axs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.q_net(mt, at).argmax(1).item()
        except Exception:
            return 0
