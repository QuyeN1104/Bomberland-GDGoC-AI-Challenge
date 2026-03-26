import random
import argparse
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from agent.random_agent import RandomAgent
from agent.simple_rule_agent import SimpleRuleAgent
from agent.smarter_rule_agent import SmarterRuleAgent
from agent.genius_rule_agent import GeniusRuleAgent
from agent.box_farmer_agent import BoxFarmerAgent
from agent.tactical_rule_agent import TacticalRuleAgent

def run_match(model_paths, num_episodes=10, max_steps=500, seed=None):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    n_players = len(model_paths)
    agents = [None] * n_players
    info = [{"name": None, "wins": 0} for _ in range(n_players)]
    
    if seed is not None:
        random.seed(seed)
    
    for i, path in enumerate(model_paths):
        if path != "None":
            # suppose submission file is /agent/team_name/agent.py -> extract team_name as agent name
            info[i]["name"] = path.split("/")[-2]
            pass # TODO: train models -> load models from model_paths and create agents for testing
                
        else:
            # Random rule-based agent for headless testing
            x = random.randint(0, 6)
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
            elif x == 4:
                info[i]["name"] = "BoxFarmerAgent"
                agents[i] = BoxFarmerAgent(i)
            else:
                info[i]["name"] = "TacticalRuleAgent"
                agents[i] = TacticalRuleAgent(i)

    for episode in range(num_episodes):
        episode_seed = None if seed is None else seed + episode
        obs = env.reset(seed=episode_seed)
        done = False
        step = 0
        death_order = []
        prev_alive = [bool(p[2]) for p in obs["players"]]

        while not done and step < max_steps:
            actions = [agent.act(obs) for agent in agents]
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

            alive_now = [bool(p[2]) for p in obs["players"]]
            for i in range(n_players):
                if prev_alive[i] and not alive_now[i]:
                    death_order.append(info[i]["name"])
            prev_alive = alive_now
        
        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [i for i in range(n_players) if alive_final[i]]
        
        if len(survivors) == 1:
            winner = survivors[0]
            info[winner]["wins"] += 1
            print(f"Episode {episode + 1}: {info[winner]['name']} wins | Died: {death_order}")
        else:
            print(f"Episode {episode + 1}: Draw | Died: {death_order}")

    print("\n=== Summary ===")
    for i in range(n_players):
        print(f"{info[i]['name']}: {info[i]['wins']} wins")
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", default=["None", "None"])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    run_match(
        model_paths=args.model_paths,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed
    )