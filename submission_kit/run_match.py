import random
import argparse
import numpy as np

from engine.game import BomberEnv
from agent.random_agent import RandomAgent
from agent.simple_rule_agent import SimpleRuleAgent
from agent.smarter_rule_agent import SmarterRuleAgent
from agent.genius_rule_agent import GeniusRuleAgent

def run_match(model_paths, num_episodes=10, max_steps=100, seed=None):
    env = BomberEnv(max_steps=max_steps)
    n_players = len(model_paths)
    agents = [None] * n_players
    info = [{
        "name": None,
        "win": 0,
        "kill": 0,
        "death": 0
        } for _ in range(n_players)]
    
    for i, path in enumerate(model_paths):
        if path != "None":
            # suppose submission file is /agent/team_name/agent.py -> extract team_name as agent name
            info[i]["name"] = path.split("/")[-2]
            pass # TODO: train models -> load models from model_paths and create agents for testing
        else:
            x = random.randint(0, 3)
            if x == 0:
                info[i]["name"] = "RandomAgent"
                agents[i] = RandomAgent(i)
            elif x == 1:
                info[i]["name"] = "SimpleRuleAgent"
                agents[i] = SimpleRuleAgent(i)
            elif x == 2:
                info[i]["name"] = "SmarterRuleAgent"
                agents[i] = SmarterRuleAgent(i)
            elif x == 3:
                info[i]["name"] = "GeniusRuleAgent"
                agents[i] = GeniusRuleAgent(i)

    for episode in range(num_episodes):
        obs = env.reset(seed=seed)
        done = False
        step = 0

        while not done and step < max_steps:
            actions = [agent.act(obs) for agent in agents]
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1
        
        winner = 1 if sum(p[2] for p in obs["players"]) == 1 else None
        
        for i, player in enumerate(obs["players"]):
            if winner is not None and player[2] == 1: # alive
                info[i]["win"] += 1
            else:
                info[i]["death"] += 1
                # info[i]["kill"] not yet implemented
    for i in range(n_players):
        print(f"Player {info[i]['name']}: Win {info[i]['win']}, Death {info[i]['death']}, Kill {info[i]['kill']}")
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", default=["None", "None"]) # max 4 players, currently v1.0 supoprts only 2
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_match(
        model_paths=args.model_paths,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed
    )