import argparse
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
from tqdm import tqdm

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from engine import BomberEnv
from DQN import DQNAgent, ReplayBuffer, encode_obs, save_model_fn
from reward import compute_reward
from utils import plot_loss, plot_rewards, plot_win_rates, plot_moving_average
from agent import (
    SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent,
    GeniusRuleAgent, BoxFarmerAgent,
)

EXPERT_AGENTS = {
    "genius":     GeniusRuleAgent,
    "tactical":   TacticalRuleAgent,
    "smarter":    SmarterRuleAgent,
    "simple":     SimpleRuleAgent,
    "box_farmer": BoxFarmerAgent,
}

ENEMY_AGENTS = EXPERT_AGENTS


def _make_rule_agent(name: str, agent_id: int):
    cls = EXPERT_AGENTS.get(name) or ENEMY_AGENTS.get(name)
    if cls is None:
        raise ValueError(f"Unknown agent type: {name}")
    return cls(agent_id)


def collect_demonstrations(
    expert_type: str,
    opponent_type: str,
    num_episodes: int,
    max_steps: int,
    seed: int,
):
    """Run an expert rule agent and record its transitions.

    The expert plays as player 0.  Each transition is stored with
    reward = 1.0 (the SQIL signal for demonstration data).

    Returns
    -------
    demo_buffer : ReplayBuffer
        Filled buffer whose reward column is all 1.0.
    input_spec : tuple
        (map_shape, aux_dim) inferred from the first observation.
    """
    env = BomberEnv(max_steps=max_steps, seed=seed)
    expert = _make_rule_agent(expert_type, agent_id=0)
    opponent = _make_rule_agent(opponent_type, agent_id=1)
    agent_ids = [0, 1]

    dummy_obs = env.reset(seed=seed)
    sample = encode_obs(dummy_obs, agent_ids)
    input_spec = (sample[0].shape, sample[1].shape[0])

    capacity = num_episodes * max_steps
    demo_buffer = ReplayBuffer(capacity=capacity, map_shape=input_spec[0], aux_dim=input_spec[1])

    total_transitions = 0
    for ep in tqdm(range(num_episodes), desc="Collecting demos"):
        obs = env.reset(seed=seed + ep)
        map_state, aux_state = encode_obs(obs, agent_ids)

        for _ in range(max_steps):
            expert_action = expert.act(obs)
            opp_action = opponent.act(obs)
            actions = [expert_action, opp_action]

            next_obs, terminated, truncated = env.step(actions)
            done = terminated or truncated

            next_map_state, next_aux_state = encode_obs(next_obs, agent_ids)
            demo_buffer.push(
                map_state, aux_state,
                expert_action, 1.0,
                next_map_state, next_aux_state,
                float(done),
            )
            total_transitions += 1

            obs = next_obs
            map_state = next_map_state
            aux_state = next_aux_state

            if done:
                break

    print(f"Collected {total_transitions} expert transitions "
          f"({num_episodes} episodes, expert={expert_type}, opponent={opponent_type})")
    return demo_buffer, input_spec


def train_sqil(
    user_id: int = 0,
    expert_type: str = "tactical",
    enemy_type: str = "tactical",
    demo_episodes: int = 50,
    num_episodes: int = 200,
    max_steps: int = 500,
    seed: int = 86,
    save_model: bool = True,
    pretrained_model=None,
):
    """Soft Q Imitation Learning.

    Phase 1 — collect expert demonstrations (reward overridden to 1).
    Phase 2 — train DQN where the agent's own transitions use reward 0,
              and each mini-batch is 50 % demo / 50 % self-play.
    """

    # ------------------------------------------------------------------
    # Phase 1: demonstration collection
    # ------------------------------------------------------------------
    demo_buffer, input_spec = collect_demonstrations(
        expert_type=expert_type,
        opponent_type=enemy_type,
        num_episodes=demo_episodes,
        max_steps=max_steps,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Phase 2: SQIL training loop
    # ------------------------------------------------------------------
    env = BomberEnv(max_steps=max_steps, seed=seed)
    enemy_agent = _make_rule_agent(enemy_type, agent_id=1)
    agent_ids = [user_id, enemy_agent.agent_id]
    num_actions = 6

    epsilon_start = 0.5
    epsilon_min   = 0.05
    epsilon_decay = 0.995
    epsilon       = epsilon_start
    batch_size    = 64
    half_batch    = batch_size // 2
    lr            = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    user_agent = DQNAgent(
        user_id, input_spec, num_actions,
        lr=lr, device=device, pretrained_model=pretrained_model,
    )

    env_buffer = ReplayBuffer(capacity=10_000, map_shape=input_spec[0], aux_dim=input_spec[1])

    loss_history = []
    reward_history = []
    win_history = []

    with tqdm(total=num_episodes, desc="Training SQIL") as pbar:
        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            map_state, aux_state = encode_obs(obs, agent_ids)
            prev_obs = None
            total_reward = 0.0

            for _ in range(max_steps):
                user_action  = user_agent.act(map_state, aux_state, epsilon=epsilon)
                enemy_action = enemy_agent.act(obs)
                actions = [None, None]
                actions[user_id]              = user_action
                actions[enemy_agent.agent_id] = enemy_action

                next_obs, terminated, truncated = env.step(actions)
                done = terminated or truncated

                next_map_state, next_aux_state = encode_obs(next_obs, agent_ids)

                # SQIL: agent's own experience gets reward = 0
                env_buffer.push(
                    map_state, aux_state,
                    user_action, 0.0,
                    next_map_state, next_aux_state,
                    float(done),
                )

                r = compute_reward(prev_obs, next_obs, agent_id=user_id)
                total_reward += r
                reward_history.append(r)
                if done:
                    win_history.append(1 if next_obs["players"][user_id][2] else 0)

                # Train once both buffers have enough data
                if len(env_buffer) >= half_batch and len(demo_buffer) >= half_batch:
                    d_map, d_aux, d_nmap, d_naux, d_act, d_rew, d_done = demo_buffer.sample(half_batch)
                    e_map, e_aux, e_nmap, e_naux, e_act, e_rew, e_done = env_buffer.sample(half_batch)

                    batch_map  = np.concatenate([d_map,  e_map],  axis=0)
                    batch_aux  = np.concatenate([d_aux,  e_aux],  axis=0)
                    batch_nmap = np.concatenate([d_nmap, e_nmap], axis=0)
                    batch_naux = np.concatenate([d_naux, e_naux], axis=0)
                    batch_act  = np.concatenate([d_act,  e_act],  axis=0)
                    batch_rew  = np.concatenate([d_rew,  e_rew],  axis=0)
                    batch_done = np.concatenate([d_done, e_done], axis=0)

                    loss = user_agent.train_step(
                        batch_map, batch_aux,
                        batch_nmap, batch_naux,
                        batch_act, batch_rew, batch_done,
                    )
                    loss_history.append(loss)

                prev_obs  = obs
                obs       = next_obs
                map_state = next_map_state
                aux_state = next_aux_state

                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if ep % 10 == 0:
                user_agent.update_target_network()
            pbar.update(1)
            pbar.set_postfix(reward=f"{total_reward:.2f}", epsilon=f"{epsilon:.3f}")

    # ------------------------------------------------------------------
    # Save & plot
    # ------------------------------------------------------------------
    tag = f"sqil_{expert_type}_{enemy_type}_{num_episodes}ep_{max_steps}steps_{seed}seed"
    model_folder = f"ckpts/{tag}"

    if save_model:
        model_path = f"{model_folder}/{user_agent.global_step}_global_step.pth"
        save_model_fn(
            user_agent.q_net, user_agent.optimizer,
            user_agent.global_step, user_agent.epsilon,
            user_agent.lr, input_spec, num_actions, model_path,
        )

    plot_loss(loss_history, save_path=f"{model_folder}/{tag}_loss.png")
    plot_rewards(reward_history, save_path=f"{model_folder}/{tag}_rewards.png")
    plot_win_rates(win_history, save_path=f"{model_folder}/{tag}_win_rates.png")
    plot_moving_average(reward_history, window_size=10, save_path=f"{model_folder}/{tag}_moving_avg.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soft Q Imitation Learning (SQIL)")
    parser.add_argument("--expert_type", type=str, default="tactical",
                        choices=list(EXPERT_AGENTS.keys()),
                        help="Expert rule agent to imitate")
    parser.add_argument("--enemy_type", type=str, default="tactical",
                        choices=list(ENEMY_AGENTS.keys()),
                        help="Opponent during RL training (and demo collection)")
    parser.add_argument("--demo_episodes", type=int, default=50,
                        help="Episodes of expert demonstrations to collect")
    parser.add_argument("--num_episodes", type=int, default=200,
                        help="RL training episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=86,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_model", action="store_true",
                        help="Save model checkpoint after training")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a pretrained model checkpoint")
    args = parser.parse_args()

    train_sqil(
        expert_type=args.expert_type,
        enemy_type=args.enemy_type,
        demo_episodes=args.demo_episodes,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        save_model=args.save_model,
        pretrained_model=args.load_model,
    )
