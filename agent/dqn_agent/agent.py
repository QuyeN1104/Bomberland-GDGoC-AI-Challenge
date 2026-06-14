"""DQN Agent v5 — Double Dueling DQN + PER + Noisy Nets + Action Masking.

Upgrades over previous versions:
  - Explicit NoisyNet Toggling: Forces .train() during rollout for proper exploration, 
    and .eval() during submission/inference to freeze noise.
  - Action Masking: Forces Q-Network to ignore invalid moves (walls/boxes) and empty bombs.
  - Survival Instinct: Overrides exploration to force escaping when standing on an active bomb.
  - Combo Tactics: 10% chance to drop consecutive bombs while escaping.
  - Local Epsilon Tracking: Fine-tuning correctly decays epsilon for the current run only.
  - FIXED: Allows movement through bombs (Channel 7) and bounds checking includes edge tiles.
  - FIXED: Blocks placing bombs if there is already a bomb under feet.
"""
from pathlib import Path
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import DuelingDQN
from obs_encoder import encode_obs


# ──────────────────────────── Training Agent ────────────────────────────
class TrainingAgent:
    team_id = "DQNv5"

    # ── Exploration hyperparameters ──
    EPS_START = 0.0           
    EPS_END = 0.0             
    BOMB_ACTION = 5           
    BOMB_EXPLORE_PROB = 0.35  
    COMBO_BOMB_PROB = 0.10    
    MOVE_MOMENTUM = 0.35      

    def __init__(self, agent_id, input_spec, num_actions, lr=5e-4,
                 device="cpu", pretrained_model=None):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.device = device
        self.gamma = 0.99
        self.lr = lr
        self.global_step = 0
        self.tau = 0.001  
        self.episode_count = 0  
        self.local_episode = 0  
        self.num_episodes = 1   
        self._last_random_action = None  

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
        frac = min(self.local_episode / max(self.num_episodes, 1), 1.0)
        return self.EPS_START + (self.EPS_END - self.EPS_START) * frac

    _ACTION_DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def _get_valid_moves(self, map_state):
        pos_ch = map_state[5]
        pos = np.argwhere(pos_ch > 0.5)
        if len(pos) == 0:
            return {1, 2, 3, 4}

        ax, ay = int(pos[0][0]), int(pos[0][1])
        H, W = map_state.shape[1], map_state.shape[2]

        valid = set()
        for action, (dx, dy) in self._ACTION_DELTAS.items():
            nx, ny = ax + dx, ay + dy
            # FIXED: Cho phép đi sát mép bản đồ (0 <= nx < H)
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            # FIXED: Chỉ chặn Tường (1) và Hòm (2). Cho phép đi xuyên bom.
            if map_state[1, nx, ny] > 0.5 or map_state[2, nx, ny] > 0.5:
                continue
            valid.add(action)
        return valid

    def _random_action_bomb_biased(self, map_state=None, aux_state=None):
        valid_moves = self._get_valid_moves(map_state) if map_state is not None else {1, 2, 3, 4}
        has_bombs = aux_state is None or float(aux_state[0]) > 0

        is_in_danger = False
        bomb_under_feet = False
        
        if map_state is not None:
            pos_ch = map_state[5]
            pos = np.argwhere(pos_ch > 0.5)
            if len(pos) > 0:
                ax, ay = int(pos[0][0]), int(pos[0][1])
                if map_state[7, ax, ay] > 0.01:
                    is_in_danger = True
                    bomb_under_feet = True # Xác định có bom dưới chân

        if is_in_danger:
            if has_bombs and not bomb_under_feet and random.random() < self.COMBO_BOMB_PROB:
                self._last_random_action = self.BOMB_ACTION
                return self.BOMB_ACTION
            
            if valid_moves:
                action = random.choice(list(valid_moves))
                self._last_random_action = action
                return action
            return 0  

        if has_bombs and not bomb_under_feet:
            bomb_prob = self.BOMB_EXPLORE_PROB
            if map_state is not None and len(pos) > 0:
                if map_state[11, ax, ay] > 0.01:  
                    bomb_prob = 0.60
            
            if random.random() < bomb_prob:
                self._last_random_action = self.BOMB_ACTION
                return self.BOMB_ACTION

        if not valid_moves:
            self._last_random_action = 0
            return 0

        last = self._last_random_action
        if last is not None and last in valid_moves and random.random() < self.MOVE_MOMENTUM:
            return last

        action = random.choice(list(valid_moves))
        self._last_random_action = action
        return action

    def act(self, map_state, aux_state, is_training=True):
        if is_training and self.epsilon > 0 and random.random() < self.epsilon:
            return self._random_action_bomb_biased(map_state, aux_state)
            
        mt = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        at = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)
        
        if is_training:
            self.q_net.train()
        else:
            self.q_net.eval()
            
        with torch.no_grad():
            q_values = self.q_net(mt, at).squeeze(0)
            
            valid_moves = self._get_valid_moves(map_state)
            mask = torch.ones(6, dtype=torch.bool, device=self.device)
            
            for act_idx in [1, 2, 3, 4]:
                if act_idx not in valid_moves:
                    mask[act_idx] = False
                    
            # FIXED: Chặn spam bom
            has_bombs = float(aux_state[0]) > 0
            bomb_under_feet = False
            pos = np.argwhere(map_state[5] > 0.5)
            if len(pos) > 0:
                ax, ay = int(pos[0][0]), int(pos[0][1])
                if map_state[7, ax, ay] > 0.01:
                    bomb_under_feet = True
                    
            if not has_bombs or bomb_under_feet:
                mask[self.BOMB_ACTION] = False
                
            q_values[~mask] = -float('inf')
            best_action = q_values.argmax().item()
            
            if best_action in [1, 2, 3, 4]:
                self._last_random_action = best_action
                
            return best_action

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

        self.q_net.train()
        q = self.q_net(ms_t, ax_t).gather(1, a_t)

        with torch.no_grad():
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

        for tp, sp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        self.q_net.reset_noise()
        self.target_net.reset_noise()

        self.global_step += 1
        return loss.item(), td_err

    def _load(self, path, requested_lr=None):
        ckpt = torch.load(path, map_location=self.device)
        spec = ckpt.get("input_spec", ckpt.get("input_shape", ckpt["input_dim"]))
        self.map_shape = tuple(spec[0])
        self.aux_dim = int(spec[1])
        self.num_actions = ckpt["num_actions"]
        noisy = ckpt.get("noisy", True)
        self.q_net = DuelingDQN(self.map_shape, self.aux_dim,
                                self.num_actions, noisy=noisy).to(self.device)
        self.q_net.load_state_dict(ckpt["model_state_dict"])
        
        current_lr = requested_lr if requested_lr is not None else ckpt.get("lr", 5e-4)
        self.lr = current_lr
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr,
                                     eps=1.5e-4, weight_decay=1e-5)
                                     
        if "optimizer_state_dict" in ckpt and requested_lr is None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            
        self.global_step = ckpt.get("global_step", 0)
        self.episode_count = ckpt.get("episode_count", 0)


# ──────────────────────────── Training Loop ─────────────────────────────
def train_dqn(user_id=0, enemy_type="simple", num_episodes=100,
              max_steps=500, seed=86, save_model=True, pretrained_model=None,
              lr=1e-4, eps_start=None, eps_end=None, save_dir="ckpts"):
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
    enemy_classes = {
        "simple": SimpleRuleAgent, "smarter": SmarterRuleAgent,
        "tactical": TacticalRuleAgent, "genius": GeniusRuleAgent,
        "box_farmer": BoxFarmerAgent,
    }
    EnemyClass = enemy_classes[enemy_type]

    batch_size = 128
    n_step = 3
    buf_cap = 100_000
    train_every = 4

    dummy = env.reset(seed=seed)
    ids = list(range(4))
    s0 = encode_obs(dummy, ids)
    input_spec = (s0[0].shape, s0[1].shape[0])
    n_actions = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = TrainingAgent(user_id, input_spec, n_actions, lr=lr,
                          device=device, pretrained_model=pretrained_model)

    if eps_start is not None:
        agent.EPS_START = eps_start
    if eps_end is not None:
        agent.EPS_END = eps_end
        
    agent.num_episodes = num_episodes
    agent.local_episode = 0

    print(f"Exploration: eps={agent.epsilon:.3f} "
          f"(start={agent.EPS_START}, end={agent.EPS_END}, "
          f"decay over {num_episodes} FRESH episodes)")

    buf = PrioritizedReplayBuffer(buf_cap, input_spec[0], input_spec[1],
                                  n_step=n_step, gamma=agent.gamma)
    n_step_gamma = agent.gamma ** n_step

    loss_hist, rew_hist, win_hist = [], [], []

    best_moving_avg = -float('inf')
    tag = f"dqnv5_{enemy_type}_{num_episodes}ep_{seed}s"
    save_folder = f"{save_dir}/{tag}"
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {save_folder}")

    with tqdm(total=num_episodes, desc="Training DQN") as pbar:
        for ep in range(num_episodes):
            ep_user_id = random.randint(0, 3)
            ep_enemy_ids = [i for i in range(4) if i != ep_user_id]
            enemy_agents = [EnemyClass(eid) for eid in ep_enemy_ids]

            obs = env.reset(seed=seed + ep)
            prev_obs = obs
            buf.reset_episode()
            total_r = 0.0
            ids = [ep_user_id] + ep_enemy_ids
            ms, axs = encode_obs(obs, ids)

            env_step = 0
            for _ in range(max_steps):
                ua = agent.act(ms, axs, is_training=True)
                actions = [None, None, None, None]
                actions[ep_user_id] = ua
                for ea_agent in enemy_agents:
                    actions[ea_agent.agent_id] = ea_agent.act(obs)

                nobs, term, trunc = env.step(actions)
                done = term or trunc
                r = compute_reward(prev_obs, nobs, agent_id=ep_user_id)
                total_r += r

                nms, naxs = encode_obs(nobs, ids)
                buf.push(ms, axs, ua, r, nms, naxs, float(done))
                env_step += 1

                if len(buf) >= batch_size and env_step % train_every == 0:
                    sms, saxs, snms, snaxs, sa, sr, sd, tidx, w = buf.sample(batch_size)
                    loss, td = agent.train_step(sms, saxs, snms, snaxs, sa, sr, sd,
                                                weights=w, n_step_gamma=n_step_gamma)
                    buf.update_priorities(tidx, td)
                    loss_hist.append(loss)

                prev_obs = obs
                obs = nobs
                ms, axs = nms, naxs
                if done:
                    win_hist.append(1 if nobs["players"][ep_user_id][2] else 0)
                    break

            agent.episode_count += 1
            agent.local_episode += 1
            
            rew_hist.append(total_r)
            buf.anneal_beta(ep / num_episodes)
            pbar.update(1)
            pbar.set_postfix(R=f"{total_r:.1f}", eps=f"{agent.epsilon:.3f}", step=agent.global_step)

            if save_model:
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

    plot_loss(loss_hist, save_path=f"{save_folder}/{tag}_loss.png")
    plot_rewards(rew_hist, save_path=f"{save_folder}/{tag}_rewards.png")
    plot_win_rates(win_hist, save_path=f"{save_folder}/{tag}_winrates.png")
    plot_moving_average(rew_hist, 10, save_path=f"{save_folder}/{tag}_ma.png")


def training():
    from utils import seed_everything
    p = argparse.ArgumentParser()
    p.add_argument("--enemy_type", default="simple",
                   choices=["simple","smarter","tactical","genius","box_farmer"])
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--save_dir", type=str, default="ckpts")
    p.add_argument("--load_model", type=str, default=None)
    p.add_argument("--eps_start", type=float, default=None)
    p.add_argument("--eps_end", type=float, default=None)
    args = p.parse_args()
    seed_everything(args.seed)
    train_dqn(enemy_type=args.enemy_type, num_episodes=args.num_episodes,
              max_steps=args.max_steps, seed=args.seed,
              save_model=args.save_model, pretrained_model=args.load_model,
              lr=args.lr, eps_start=args.eps_start, eps_end=args.eps_end,
              save_dir=args.save_dir)


# ──────────────────── Submission Agent (mandatory) ──────────────────────
class Agent:
    """Eval Agent with integrated Action Masking. NoisyNet is explicitly disabled."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cpu")
        self.q_net = None

        ckpt_path = Path(__file__).parent / "latest_checkpoint_test.pth"
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
            # FIXED: Cho phép đi sát mép bản đồ (0 <= nx < H)
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            # FIXED: Chỉ chặn Tường (1) và Hòm (2). Cho phép đi xuyên bom.
            if map_state[1, nx, ny] > 0.5 or map_state[2, nx, ny] > 0.5:
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
                        
                # FIXED: Chặn spam bom
                has_bombs = float(axs[0]) > 0
                bomb_under_feet = False
                pos = np.argwhere(ms[5] > 0.5)
                if len(pos) > 0:
                    ax, ay = int(pos[0][0]), int(pos[0][1])
                    if ms[7, ax, ay] > 0.01:
                        bomb_under_feet = True
                        
                if not has_bombs or bomb_under_feet:
                    mask[5] = False
                    
                q_values[~mask] = -float('inf')
                self.last_q_values = q_values.cpu().numpy().tolist()
                return q_values.argmax().item()
                
        except Exception:
            self.last_q_values = None
            return 0


if __name__ == "__main__":
    training()