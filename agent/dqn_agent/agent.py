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


# ──────────────────────────── Training Agent ────────────────────────────
class TrainingAgent:
    """Agent wrapper for training with Double Dueling DQN.

    Exploration strategy (v3 — bomb-biased warmup):
      - Epsilon-greedy warmup: eps 1.0 → 0.05 over eps_decay_episodes
      - During random exploration, bomb action (5) gets BOMB_EXPLORE_PROB (30%)
        instead of uniform 1/6 ≈ 17%, ensuring agent discovers bombing early
      - NoisyNets still active on top of epsilon for continuous exploration
    """
    team_id = "DQNv3"

    # ── Exploration hyperparameters ──
    # Epsilon-greedy OFF by default — NoisyNets handle exploration adaptively.
    # Enable via CLI: --eps_start 1.0 --eps_end 0.05
    EPS_START = 0.0           # No epsilon-greedy (NoisyNets only)
    EPS_END = 0.0             # No epsilon-greedy (NoisyNets only)
    BOMB_ACTION = 5           # Action index for placing bomb
    BOMB_EXPLORE_PROB = 0.30  # 30% chance to pick bomb during random exploration

    def __init__(self, agent_id, input_spec, num_actions, lr=5e-4,
                 device="cpu", pretrained_model=None):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.device = device
        self.gamma = 0.99
        self.lr = lr
        self.global_step = 0
        self.tau = 0.005  # Polyak soft-update coefficient
        self.episode_count = 0  # Track episodes for epsilon decay
        self.num_episodes = 1   # Total episodes (set by train_dqn)

        if pretrained_model:
            self._load(pretrained_model)
        else:
            self.map_shape = tuple(input_spec[0])
            self.aux_dim = int(input_spec[1])
            self.q_net = DuelingDQN(self.map_shape, self.aux_dim, num_actions).to(device)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr,
                                        eps=1.5e-4, weight_decay=1e-5)

        self.target_net = DuelingDQN(self.map_shape, self.aux_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

    @property
    def epsilon(self):
        """Linear epsilon decay: EPS_START → EPS_END over num_episodes."""
        frac = min(self.episode_count / max(self.num_episodes, 1), 1.0)
        return self.EPS_START + (self.EPS_END - self.EPS_START) * frac

    def _random_action_bomb_biased(self):
        """Bomb-biased random action: 30% bomb, 70% uniform over other actions."""
        if random.random() < self.BOMB_EXPLORE_PROB:
            return self.BOMB_ACTION  # Force bomb action
        # Uniform over non-bomb actions: 0,1,2,3,4
        return random.randint(0, self.num_actions - 2)

    # ── action selection ──
    def act(self, map_state, aux_state, epsilon=None):
        """Epsilon-greedy with bomb-biased exploration + NoisyNets.

        During training: uses self.epsilon (auto-decaying) unless overridden.
        During eval: pass epsilon=0.0 to disable random exploration.
        """
        eps = self.epsilon if epsilon is None else epsilon
        if eps > 0 and random.random() < eps:
            return self._random_action_bomb_biased()
        mt = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        at = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(mt, at).argmax().item()

    # ── training step (Double DQN + Huber + PER weights) ──
    def train_step(self, ms, axs, nms, naxs, act, rew, done,
                   weights=None, n_step_gamma=None):
        dev = self.device
        ms_t  = torch.from_numpy(ms).to(dev)
        ax_t  = torch.from_numpy(axs).to(dev)
        nms_t = torch.from_numpy(nms).to(dev)
        nax_t = torch.from_numpy(naxs).to(dev)
        a_t   = torch.from_numpy(act).unsqueeze(1).to(dev)
        r_t   = torch.from_numpy(rew).unsqueeze(1).to(dev)
        d_t   = torch.from_numpy(done).unsqueeze(1).to(dev)
        g = n_step_gamma if n_step_gamma else self.gamma

        q = self.q_net(ms_t, ax_t).gather(1, a_t)

        with torch.no_grad():
            # Double DQN: q_net selects, target_net evaluates
            best_a = self.q_net(nms_t, nax_t).argmax(1, keepdim=True)
            next_q = self.target_net(nms_t, nax_t).gather(1, best_a)
            target = r_t + g * next_q * (1 - d_t)

        td_err = (q - target).detach().cpu().numpy().flatten()

        if weights is not None:
            w = torch.from_numpy(weights).unsqueeze(1).to(dev)
            loss = (w * F.smooth_l1_loss(q, target, reduction='none')).mean()
        else:
            loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft target update
        for tp, sp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        # Reset noise for next forward pass
        self.q_net.reset_noise()
        self.target_net.reset_noise()

        self.global_step += 1
        return loss.item(), td_err

    # ── persistence ──
    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        spec = ckpt.get("input_spec", ckpt.get("input_shape", ckpt["input_dim"]))
        self.map_shape = tuple(spec[0])
        self.aux_dim = int(spec[1])
        self.num_actions = ckpt["num_actions"]
        noisy = ckpt.get("noisy", True)
        self.q_net = DuelingDQN(self.map_shape, self.aux_dim,
                                self.num_actions, noisy=noisy).to(self.device)
        self.q_net.load_state_dict(ckpt["model_state_dict"])
        self.lr = ckpt.get("lr", 5e-4)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr,
                                     eps=1.5e-4, weight_decay=1e-5)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.episode_count = ckpt.get("episode_count", 0)


# ──────────────────────────── Training Loop ─────────────────────────────
def train_dqn(user_id=0, enemy_type="simple", num_episodes=100,
              max_steps=500, seed=86, save_model=True, pretrained_model=None,
              lr=1e-4, eps_start=None, eps_end=None,
              episode_count_override=None):
    from tqdm import tqdm
    import sys as _sys
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from reward import compute_reward
    from utils import (plot_loss, plot_rewards, plot_win_rates,
                       plot_moving_average, seed_everything, save_model_fn)
    from agent import (SimpleRuleAgent, SmarterRuleAgent,
                       TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent)
    from engine import BomberEnv
    from replay_buffer import PrioritizedReplayBuffer

    env = BomberEnv(max_steps=max_steps, seed=seed)
    enemies = {
        "simple": SimpleRuleAgent, "smarter": SmarterRuleAgent,
        "tactical": TacticalRuleAgent, "genius": GeniusRuleAgent,
        "box_farmer": BoxFarmerAgent,
    }
    enemy_agent = enemies[enemy_type](1)

    # Hyperparameters
    batch_size = 128
    n_step = 3
    buf_cap = 100_000

    dummy = env.reset(seed=seed)
    ids = [user_id, enemy_agent.agent_id]
    s0 = encode_obs(dummy, ids)
    input_spec = (s0[0].shape, s0[1].shape[0])
    n_actions = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = TrainingAgent(user_id, input_spec, n_actions, lr=lr,
                          device=device, pretrained_model=pretrained_model)

    # Override exploration params if specified
    if eps_start is not None:
        agent.EPS_START = eps_start
    if eps_end is not None:
        agent.EPS_END = eps_end
    agent.num_episodes = num_episodes  # Decay over total training episodes
    if episode_count_override is not None:
        agent.episode_count = episode_count_override

    print(f"Exploration: eps={agent.epsilon:.3f} "
          f"(start={agent.EPS_START}, end={agent.EPS_END}, "
          f"decay over {num_episodes} episodes, "
          f"episode_count={agent.episode_count})")

    buf = PrioritizedReplayBuffer(buf_cap, input_spec[0], input_spec[1],
                                  n_step=n_step, gamma=agent.gamma)
    n_step_gamma = agent.gamma ** n_step

    loss_hist, rew_hist, win_hist = [], [], []

    best_moving_avg = -float('inf')
    save_folder = f"ckpts/dqnv2_{enemy_type}_{num_episodes}ep_{seed}s"
    Path(save_folder).mkdir(parents=True, exist_ok=True) # Đảm bảo thư mục tồn tại

    with tqdm(total=num_episodes, desc="Training DQN v2") as pbar:
        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            prev_obs = None
            total_r = 0.0
            ms, axs = encode_obs(obs, ids)

            for _ in range(max_steps):
                ua = agent.act(ms, axs)  # auto epsilon + bomb-biased + noisy
                ea = enemy_agent.act(obs)
                actions = [None, None]
                actions[user_id] = ua
                actions[enemy_agent.agent_id] = ea

                nobs, term, trunc = env.step(actions)
                done = term or trunc
                r = compute_reward(prev_obs, nobs, agent_id=user_id)
                total_r += r

                nms, naxs = encode_obs(nobs, ids)
                buf.push(ms, axs, ua, r, nms, naxs, float(done))

                if len(buf) >= batch_size:
                    sms, saxs, snms, snaxs, sa, sr, sd, tidx, w = buf.sample(batch_size)
                    loss, td = agent.train_step(sms, saxs, snms, snaxs, sa, sr, sd,
                                                weights=w, n_step_gamma=n_step_gamma)
                    buf.update_priorities(tidx, td)
                    loss_hist.append(loss)

                prev_obs = obs
                obs = nobs
                ms, axs = nms, naxs
                if done:
                    win_hist.append(1 if nobs["players"][user_id][2] else 0)
                    break

            agent.episode_count += 1  # Track for epsilon decay
            rew_hist.append(total_r)
            buf.anneal_beta(ep / num_episodes)
            pbar.update(1)
            pbar.set_postfix(R=f"{total_r:.1f}", eps=f"{agent.epsilon:.3f}", step=agent.global_step)

            if save_model:
                # 1. Lưu backup mỗi 10 episodes (để lỡ crash thì load lại từ đây)
                if (ep + 1) % 10 == 0:
                    latest_path = f"{save_folder}/latest_checkpoint.pth"
                    save_model_fn(agent.q_net, agent.optimizer, agent.global_step,
                                  0.0, agent.lr, input_spec, n_actions, latest_path,
                                  episode_count=agent.episode_count)
                
                current_ma = np.mean(rew_hist[-10:]) if len(rew_hist) >= 10 else total_r
                
                if current_ma > best_moving_avg and ep > 5:
                    best_moving_avg = current_ma
                    best_path = f"{save_folder}/best_model.pth"
                    save_model_fn(agent.q_net, agent.optimizer, agent.global_step,
                                  0.0, agent.lr, input_spec, n_actions, best_path,
                                  episode_count=agent.episode_count)

    tag = f"dqnv2_{enemy_type}_{num_episodes}ep_{seed}s"
    folder = f"ckpts/{tag}"
  
    plot_loss(loss_hist, save_path=f"{folder}/{tag}_loss.png")
    plot_rewards(rew_hist, save_path=f"{folder}/{tag}_rewards.png")
    plot_win_rates(win_hist, save_path=f"{folder}/{tag}_winrates.png")
    plot_moving_average(rew_hist, 10, save_path=f"{folder}/{tag}_ma.png")


def training():
    from utils import seed_everything
    p = argparse.ArgumentParser()
    p.add_argument("--enemy_type", default="simple",
                   choices=["simple","smarter","tactical","genius","box_farmer"])
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (1e-4 fine-tune, 5e-4 train mới)")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--load_model", type=str, default=None)
    # Exploration overrides
    p.add_argument("--eps_start", type=float, default=None,
                   help="Override epsilon start (default: 0.0, set 1.0 to enable)")
    p.add_argument("--eps_end", type=float, default=None,
                   help="Override epsilon end (default: 0.0, set 0.05 for light exploration)")
    p.add_argument("--episode_count", type=int, default=None,
                   help="Override starting episode count (ép epsilon về giá trị mong muốn)")
    args = p.parse_args()
    seed_everything(args.seed)
    train_dqn(enemy_type=args.enemy_type, num_episodes=args.num_episodes,
              max_steps=args.max_steps, seed=args.seed,
              save_model=args.save_model, pretrained_model=args.load_model,
              lr=args.lr, eps_start=args.eps_start, eps_end=args.eps_end,
              episode_count_override=args.episode_count)


# ──────────────────── Submission Agent (mandatory) ──────────────────────
class Agent:
    """DQN v2 agent for competition submission."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cpu")
        self.q_net = None

        ckpt_path = Path(__file__).parent / "latest_checkpoint (4).pth"
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


if __name__ == "__main__":
    training()