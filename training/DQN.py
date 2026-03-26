import argparse
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from engine import *
from agent import SimpleRuleAgent, SmarterRuleAgent

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def __len__(self):
        return len(self.buf)

    def push(self, state, action, reward, next_state, done):
        if len(self.buf) < self.capacity:
            self.buf.append((state, action, reward, next_state, done))
        else:
            self.buf[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buf)), batch_size)
        state, action, reward, next_state, done = zip(*[self.buf[i] for i in indices])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, player_id, input_dim, num_actions):
        self.player_id = player_id
        self.num_actions = num_actions
        
        self.gamma = 0.99
        self.lr = 1e-3
        
        # Networks: Q-Network (learning) and Target-Network (stable target)
        self.q_net = DQNModel(input_dim, num_actions)
        self.target_net = DQNModel(input_dim, num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict()) # Sync weights initially
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-08, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()

    def act(self, obs, epsilon=0.0, enemy_ids=[1]):
        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        state = encode_obs(obs, [self.player_id, *enemy_ids])
        state_tensor = torch.FloatTensor(state).unsqueeze(0) # add batch dim
        
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values).item()
            
        # action with the highest predicted Q-value
        return action

    def train_step(self, buffer, batch_size):
        if len(buffer) < batch_size:
            return # Not enough data to train yet
            
        # 1. Sample from replay buffer
        state, action, reward, next_state, done = buffer.sample(batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # 2. Calculate current Q-values: Q(s, a)
        # gather() extracts the Q-value for the specific action taken
        q_values = self.q_net(state).gather(1, action)
        
        # 3. Calculate Target Q-values using Bellman equation
        with torch.no_grad():
            # max(1)[0] gets the max Q-value for the next state
            # ~ max_a' {Q(s', a', weights)}
            max_next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            # If done=1, the future reward is 0.
            # Q*(s, a) = E[r + gamma * max_a' {Q*(s', a')}]
            # ~ Q(s, a) = r + gamma * max_a' {Q(s', a', weights)} if not done else Q(s, a) = r
            target_q = reward + self.gamma * max_next_q * (1 - done)
            
        # 4. Caclulate loss, backward,...
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        """Copies the learned weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    

def compute_reward(prev_obs, curr_obs, agent_id):
    """
    Args:
        prev_obs: {map, players, bombs} before an action
        curr_obs: {map, players, bombs} after an action
        agent_id
    """
    if prev_obs is None:
        return 0.0

    prev_players = prev_obs["players"]
    curr_players = curr_obs["players"]
    prev_alive = int(prev_players[agent_id][2])
    curr_alive = int(curr_players[agent_id][2])

    reward = 0.0
    # win/loss
    if prev_alive == 1 and curr_alive == 0:
        reward -= 1.0
    # small time penalty
    reward -= 0.001
    # box destruction progress (global proxy)
    prev_boxes = int(np.sum((prev_obs["map"]) == Map.BOX))
    curr_boxes = int(np.sum((curr_obs["map"]) == Map.BOX))
    boxes_destroyed = max(0, prev_boxes - curr_boxes)
    reward += 0.05 * boxes_destroyed
    # item collection proxy via player stats deltas
    prev_bombs_left = int(prev_players[agent_id][3])
    curr_bombs_left = int(curr_players[agent_id][3])
    prev_radius_bonus = int(prev_players[agent_id][4])
    curr_radius_bonus = int(curr_players[agent_id][4])
    reward += 0.1 * max(0, curr_bombs_left - prev_bombs_left)
    reward += 0.05 * max(0, curr_radius_bonus - prev_radius_bonus)
    return float(reward)

    

BOMB_MAX_TIMER = 7  # matches Bomb.__init__ default timer

def encode_obs(obs, agent_ids):
    """
    Returns a fixed-size float vector for an MLP-style DQN.

    agent_ids: int (user's player id) or list/tuple [user_id, opp_id].
    When a single int is given the enemy is inferred as the other
    player in a 2-player game (1 - user_id).
    """
    if obs is None:
        raise ValueError("obs should not be None")

    # Normalise agent_ids to (user_id, opp_id)
    user_id = int(agent_ids[0])
    opp_id  = int(agent_ids[1]) if len(agent_ids) > 1 else (1 - user_id)

    grid    = obs["map"]      # (H, W)
    players = obs["players"]  # (num_players, 5)
    bombs   = obs["bombs"]    # (N, 4), N may be 0
    H, W    = grid.shape

    # One-hot map: grass, wall, box, item_radius, item_capacity
    map_channels = []
    for v in [Map.GRASS, Map.WALL, Map.BOX, Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
        map_channels.append((grid == v).astype(np.float32))
    map_feat = np.stack(map_channels, axis=0)  # (5, H, W)

    # Player position masks
    my_x, my_y, my_alive, my_bombs_left, my_radius_bonus = players[user_id]
    ox,   oy,   opp_alive, _,            _               = players[opp_id]
    my_pos  = np.zeros((H, W), dtype=np.float32)
    opp_pos = np.zeros((H, W), dtype=np.float32)
    if int(my_alive)  == 1:
        my_pos[int(my_x), int(my_y)] = 1.0
    if int(opp_alive) == 1:
        opp_pos[int(ox), int(oy)]    = 1.0

    # Bomb channels — bombs is a numpy array, not a list of Bomb objects
    bomb_timer = np.zeros((H, W), dtype=np.float32)
    bomb_owned = np.zeros((H, W), dtype=np.float32)
    for b in bombs:
        bx, by, timer, owner_id = b
        bx, by = int(bx), int(by)
        t = float(timer) / BOMB_MAX_TIMER  # normalise by default max timer
        bomb_timer[bx, by] = max(bomb_timer[bx, by], t)
        bomb_owned[bx, by] = 1.0 if int(owner_id) == user_id else 0.0

    scalar = np.array([
        float(my_bombs_left)   / Player.MAX_BOMB_CAPACITY,
        float(my_radius_bonus) / Player.MAX_BOMB_RADIUS,
        float(opp_alive),
    ], dtype=np.float32)

    feat = np.concatenate([
        map_feat.reshape(-1),
        my_pos.reshape(-1),
        opp_pos.reshape(-1),
        bomb_timer.reshape(-1),
        bomb_owned.reshape(-1),
        scalar,
    ], axis=0)
    return feat



def train_dqn(user_id=0, enemy_type="simple", num_episodes=100, max_steps=500, seed=86):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    if enemy_type == "simple":
        enemy_agent = SimpleRuleAgent(1)
    elif enemy_type == "smarter":
        enemy_agent = SmarterRuleAgent(1)
    else:
        raise ValueError(f"Invalid enemy type: {enemy_type}")

    # hyperparam
    epsilon_start = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    epsilon = epsilon_start
    batch_size = 64 
    
    dummy_obs = env.reset(seed=seed)
    sample_state = encode_obs(dummy_obs, agent_ids=[user_id, enemy_agent.player_id])
    input_dim = len(sample_state)
    num_actions = 6
    user_agent = DQNAgent(user_id, input_dim, num_actions)
    buffer = ReplayBuffer(capacity=10_000)

    with tqdm(total=num_episodes, desc="Training DQN") as pbar:
        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            prev_obs = None
            total_reward = 0

            for t in range(max_steps):
                # Action
                user_action = user_agent.act(obs, epsilon=epsilon, enemy_ids=[enemy_agent.player_id])
                enemy_action = enemy_agent.act(obs)
                actions = [None, None]
                actions[user_id] = user_action
                actions[enemy_agent.player_id] = enemy_action

                # Step
                next_obs, terminated, truncated = env.step(actions)
                done = terminated or truncated
                
                # Calculate reward
                r = compute_reward(prev_obs, next_obs, agent_id=user_id)
                total_reward += r
                # if enemy agent is dead, reward is 1.0
                if enemy_agent.player_id == 1 and not next_obs["players"][1][2]:
                    r += 1.0

                # Store in buffer
                state = encode_obs(obs, [user_id, enemy_agent.player_id])
                next_state = encode_obs(next_obs, [user_id, enemy_agent.player_id])
                # print("state: ", state)
                # print("next_state: ", next_state)
                # print("user_action: ", user_action)
                # print("r: ", r)
                # print("done: ", done)
                # print("--------------------------------\n\n")
                buffer.push(state, user_action, r, next_state, done)

                # Train
                user_agent.train_step(buffer, batch_size)

                prev_obs = obs
                obs = next_obs
                if done:
                    break
            # Decay epsilon at the end of each episode
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            # Update Target Network every 10 episodes
            if ep % 10 == 0:
                user_agent.update_target_network()
            pbar.update(1)
            pbar.set_postfix(reward=total_reward, epsilon=epsilon)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enemy_type", type=str, default="simple")
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=86)
    args = parser.parse_args()
    train_dqn(enemy_type=args.enemy_type, num_episodes=args.num_episodes, max_steps=args.max_steps, seed=args.seed)
    # env = BomberEnv()