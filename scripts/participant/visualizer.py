import os
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame

parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
root_dir = parent_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from agent.dqn_agent.reward import compute_reward
except ImportError:
    compute_reward = None

from engine import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent
from competition.evaluation.runtime_guard import load_agent_instance

class Viewer:
	PLAYER_COLORS = [(220, 50, 50), (50, 50, 220), (30, 150, 30), (200, 140, 0)]

	def __init__(self, width=13, height=13, cell_size=42, fps=8, panel_width=280):
		self.width = width
		self.height = height
		self.cell_size = cell_size
		self.fps = fps
		self.panel_width = panel_width

		self.top_bar = 60
		self.grid_width = width * cell_size
		self.screen_width = self.grid_width + panel_width
		self.screen_height = height * cell_size + self.top_bar

		pygame.init()
		self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
		pygame.display.set_caption("Bomberland Enhanced Viewer")
		self.clock = pygame.time.Clock()
		self.font_info = pygame.font.SysFont(None, 24)
		self.font_small = pygame.font.SysFont(None, 20)
		self.explosion_overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
		self.explosion_overlay.fill((255, 140, 0, 130))

	def draw_grid(self, grid):
		for row in range(self.height):
			for col in range(self.width):
				rect = pygame.Rect(
					col * self.cell_size,
					row * self.cell_size + self.top_bar,
					self.cell_size,
					self.cell_size,
				)
				cell_type = int(grid[row, col])
				if cell_type == 1:
					pygame.draw.rect(self.screen, (80, 80, 80), rect)
					pygame.draw.rect(self.screen, (40, 40, 40), rect, 2)
				elif cell_type == 2:
					pygame.draw.rect(self.screen, (139, 69, 19), rect)
					pygame.draw.rect(self.screen, (101, 67, 33), rect, 2)
					pygame.draw.line(self.screen, (101, 67, 33), (rect.left, rect.top), (rect.right, rect.bottom), 2)
					pygame.draw.line(self.screen, (101, 67, 33), (rect.right, rect.top), (rect.left, rect.bottom), 2)
				elif cell_type == 3:
					pygame.draw.rect(self.screen, (225, 225, 225), rect)
					pygame.draw.circle(self.screen, (255, 0, 0), rect.center, self.cell_size // 4)
					text = self.font_small.render("R", True, (255, 255, 255))
					self.screen.blit(text, (rect.centerx - 5, rect.centery - 8))
				elif cell_type == 4:
					pygame.draw.rect(self.screen, (225, 225, 225), rect)
					pygame.draw.circle(self.screen, (0, 0, 255), rect.center, self.cell_size // 4)
					text = self.font_small.render("C", True, (255, 255, 255))
					self.screen.blit(text, (rect.centerx - 5, rect.centery - 8))
				else:
					pygame.draw.rect(self.screen, (144, 238, 144), rect)
					pygame.draw.rect(self.screen, (120, 200, 120), rect, 1)

	def draw_players(self, players):
		for i, p in enumerate(players):
			if p[2] != 1:
				continue
			center = (
				int(p[1]) * self.cell_size + self.cell_size // 2,
				int(p[0]) * self.cell_size + self.top_bar + self.cell_size // 2,
			)
			pygame.draw.circle(self.screen, self.PLAYER_COLORS[i % len(self.PLAYER_COLORS)], center, self.cell_size // 3)
			img = self.font_small.render(str(i), True, (255, 255, 255))
			self.screen.blit(img, (center[0] - 5, center[1] - 8))
			stats_text = f"B:{int(p[3])} R:{int(p[4])}"
			stats_img = self.font_small.render(stats_text, True, (0, 0, 0))
			self.screen.blit(stats_img, (center[0] - 16, center[1] + 12))

	def draw_bombs(self, bombs):
		for b in bombs:
			if b[2] <= 0:
				continue
			center = (
				int(b[1]) * self.cell_size + self.cell_size // 2,
				int(b[0]) * self.cell_size + self.top_bar + self.cell_size // 2,
			)
			pygame.draw.circle(self.screen, (20, 20, 20), center, self.cell_size // 4)
			timer_img = self.font_small.render(str(int(b[2])), True, (255, 255, 255))
			self.screen.blit(timer_img, (center[0] - 5, center[1] - 8))

	def draw_agent_sidebar(self, obs, agent_names):
		"""Right panel: agent name, alive/dead, bombs available, radius power-up bonus,
		plus step reward, cumulative reward, and action Q-values."""
		players = obs["players"]
		x0 = self.grid_width
		pygame.draw.rect(self.screen, (52, 58, 64), (x0, 0, self.panel_width, self.screen_height))
		pygame.draw.line(self.screen, (30, 30, 30), (x0, 0), (x0, self.screen_height), 2)

		title = self.font_info.render("Agents", True, (245, 245, 245))
		self.screen.blit(title, (x0 + 10, self.top_bar + 8))

		y = self.top_bar + 40
		line_h = 18

		rewards = obs.get("rewards", [0.0] * len(players))
		cum_rewards = obs.get("cum_rewards", [0.0] * len(players))
		q_values_list = obs.get("q_values", [None] * len(players))
		chosen_actions = obs.get("actions", [0] * len(players))

		action_labels = ["Stop", "Up", "Down", "Left", "Right", "Bomb"]

		for i, p in enumerate(players):
			name = agent_names[i] if i < len(agent_names) and agent_names[i] else f"Agent {i}"
			alive = int(p[2]) == 1
			bombs_left = int(p[3])
			radius_bonus = int(p[4])
			color = self.PLAYER_COLORS[i % len(self.PLAYER_COLORS)]

			is_our_agent = ("dqn" in name.lower() or name == "DQNv5" or name not in ["RandomAgent", "SimpleRuleAgent", "SmarterRuleAgent", "TacticalRuleAgent", "GeniusRuleAgent", "BoxFarmerAgent"])

			q_vals = q_values_list[i] if i < len(q_values_list) else None
			chosen_act = chosen_actions[i] if i < len(chosen_actions) else 0

			breakdowns = obs.get("breakdowns", [None] * len(players))
			bd = breakdowns[i] if i < len(breakdowns) else None
			bd_items = []
			if bd:
				short_keys = {
					"Death": "Death",
					"Kill Enemy": "Kill",
					"Win Match": "Win",
					"Standing Still": "Still",
					"Moving": "Move",
					"Time Penalty": "Time",
					"Approach Item": "ApprItem",
					"Collect Item": "CollItem",
					"Item Advantage": "ItemAdv",
					"Escape Danger": "Escape",
					"Enter Danger": "EnterDang",
					"Linger Danger": "Linger",
					"Approach Safe Tile": "ApprSafe",
					"Approach Enemy": "ApprEnemy",
					"Plant Bomb Base": "PlantBase",
					"Safe Bomb Plant": "SafePlant",
					"Plant Near Box": "NearBox",
					"Plant Near Enemy": "NearEnemy",
					"Chain Bomb Plant": "Chain",
					"Suicide Bomb Plant": "Suicide",
					"Destroy Box": "DestrBox",
					"Survival Bonus": "Survival"
				}
				bd_items = [f"{short_keys.get(k, k)}:{v:+.2f}" for k, v in bd.items() if abs(v) > 0.001]

			if is_our_agent:
				# Highlighted box background for DQN agent
				box_height = 80
				if q_vals is not None and alive:
					box_height += 54
				elif alive:
					box_height += 18
				if bd_items:
					box_height += ((len(bd_items) + 1) // 2) * line_h + 4
				
				pygame.draw.rect(self.screen, (40, 45, 52), (x0 + 5, y - 4, self.panel_width - 10, box_height), border_radius=6)
				pygame.draw.rect(self.screen, (0, 180, 216), (x0 + 5, y - 4, self.panel_width - 10, box_height), width=1, border_radius=6)

			# draw elements
			pygame.draw.circle(self.screen, color, (x0 + 16, y + 8), 6)
			name_color = (0, 180, 216) if is_our_agent else (240, 240, 240)
			name_img = self.font_small.render(str(name)[:28], True, name_color)
			self.screen.blit(name_img, (x0 + 30, y))
			y += line_h

			status = "Alive" if alive else "Dead"
			status_color = (120, 220, 140) if alive else (220, 100, 100)
			status_img = self.font_small.render(status, True, status_color)
			self.screen.blit(status_img, (x0 + 16, y))
			
			stats = f"Bombs: {bombs_left} | +Rad: {radius_bonus}"
			stats_img = self.font_small.render(stats, True, (180, 180, 180))
			self.screen.blit(stats_img, (x0 + 80, y))
			y += line_h

			# Step Reward & Cum Reward
			rew = rewards[i] if i < len(rewards) else 0.0
			cum_rew = cum_rewards[i] if i < len(cum_rewards) else 0.0
			rew_color = (120, 220, 140) if rew > 0 else (220, 100, 100) if rew < 0 else (180, 180, 180)
			rew_text = f"Step R: {rew:+.2f}"
			cum_text = f"Cum R: {cum_rew:+.2f}"
			
			rew_img = self.font_small.render(rew_text, True, rew_color)
			cum_img = self.font_small.render(cum_text, True, (200, 200, 200))
			self.screen.blit(rew_img, (x0 + 16, y))
			self.screen.blit(cum_img, (x0 + 130, y))
			y += line_h

			# Q-values or Action Display
			if q_vals is not None and alive:
				for row_idx in range(3):
					# Col 1
					a1 = row_idx * 2
					q1 = q_vals[a1]
					q1_str = f"{q1:.2f}" if q1 != -float('inf') else "-inf"
					text1 = f"{action_labels[a1]}:{q1_str}"
					color1 = (255, 215, 0) if a1 == chosen_act else (150, 150, 150)
					img1 = self.font_small.render(text1, True, color1)
					self.screen.blit(img1, (x0 + 16, y))

					# Col 2
					a2 = row_idx * 2 + 1
					q2 = q_vals[a2]
					q2_str = f"{q2:.2f}" if q2 != -float('inf') else "-inf"
					text2 = f"{action_labels[a2]}:{q2_str}"
					color2 = (255, 215, 0) if a2 == chosen_act else (150, 150, 150)
					img2 = self.font_small.render(text2, True, color2)
					self.screen.blit(img2, (x0 + 130, y))
					y += line_h
			else:
				if alive:
					act_name = action_labels[chosen_act] if chosen_act < len(action_labels) else str(chosen_act)
					act_img = self.font_small.render(f"Action: {act_name}", True, (255, 215, 0))
					self.screen.blit(act_img, (x0 + 16, y))
					y += line_h
				else:
					y += line_h

			if is_our_agent and bd_items:
				pygame.draw.line(self.screen, (70, 75, 80), (x0 + 10, y + 2), (x0 + self.panel_width - 10, y + 2), 1)
				y += 4
				for idx in range(0, len(bd_items), 2):
					chunk = bd_items[idx:idx+2]
					line_str = " | ".join(chunk)
					item_img = self.font_small.render(line_str, True, (130, 200, 250))
					self.screen.blit(item_img, (x0 + 16, y))
					y += line_h

			y += 14 if is_our_agent else 10

	def draw_header(self, episode_idx, total_episodes, step_idx, total_steps, paused):
		pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.screen_width, self.top_bar))
		status = "PAUSED" if paused else "PLAYING"
		text = (
			f"Ep {episode_idx + 1}/{total_episodes} | "
			f"Step {step_idx}/{max(total_steps - 1, 0)} | {status}"
		)
		help_text = "[A/D] Step [W/S] Ep [SPACE] Pause [ESC] Quit"
		self.screen.blit(self.font_info.render(text, True, (245, 245, 245)), (10, 5))
		self.screen.blit(self.font_small.render(help_text, True, (210, 210, 210)), (10, 35))

	def _in_bounds(self, row, col):
		return 0 <= row < self.height and 0 <= col < self.width

	def _blast_tiles(self, grid, bx, by, radius):
		tiles = {(bx, by)}
		for drow, dcol in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
			for r in range(1, radius + 1):
				tr, tc = bx + drow * r, by + dcol * r
				if not self._in_bounds(tr, tc):
					break
				cell = int(grid[tr, tc])
				if cell == 1:
					break
				tiles.add((tr, tc))
				if cell == 2:
					break
		return tiles

	def _explosion_tiles_from_transition(self, prev_obs, obs):
		if prev_obs is None:
			return set()

		prev_bombs = prev_obs["bombs"]
		curr_bombs = obs["bombs"]
		curr_positions = {(int(b[0]), int(b[1])) for b in curr_bombs}
		prev_players = prev_obs["players"]
		prev_grid = prev_obs["map"]

		tiles = set()
		for b in prev_bombs:
			bx, by, timer, owner_id = int(b[0]), int(b[1]), int(b[2]), int(b[3])
			exploded = timer <= 1 or (bx, by) not in curr_positions
			if not exploded:
				continue
			radius = 1
			if 0 <= owner_id < len(prev_players):
				radius = 1 + int(prev_players[owner_id][4])
			tiles.update(self._blast_tiles(prev_grid, bx, by, radius))
		return tiles

	def draw_explosions(self, explosion_tiles):
		for row, col in explosion_tiles:
			px = col * self.cell_size
			py = row * self.cell_size + self.top_bar
			self.screen.blit(self.explosion_overlay, (px, py))
			center = (px + self.cell_size // 2, py + self.cell_size // 2)
			pygame.draw.circle(self.screen, (255, 220, 120), center, self.cell_size // 6)

	def render(self, obs, prev_obs, episode_idx, total_episodes, step_idx, total_steps, paused, agent_names):
		self.screen.fill((245, 245, 245))
		self.draw_grid(obs["map"])
		explosion_tiles = self._explosion_tiles_from_transition(prev_obs, obs)
		self.draw_explosions(explosion_tiles)
		self.draw_players(obs["players"])
		self.draw_bombs(obs["bombs"])
		self.draw_agent_sidebar(obs, agent_names)
		self.draw_header(episode_idx, total_episodes, step_idx, total_steps, paused)
		pygame.display.flip()
		self.clock.tick(self.fps)

	def close(self):
		pygame.quit()


def str2bool(value):
	if isinstance(value, bool):
		return value
	value = str(value).strip().lower()
	if value in {"true", "1", "yes", "y", "t"}:
		return True
	if value in {"false", "0", "no", "n", "f"}:
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def make_agents(agent_paths, seed=None):
	n_players = len(agent_paths)
	agents = [None] * n_players
	names = [None] * n_players

	if seed is not None:
		random.seed(seed)

	for i, path in enumerate(agent_paths):
		if path == "None" or path.lower() == "random":
			# Random rule-based baseline
			x = random.randint(0, 5)
			if x == 0:
				names[i] = "RandomAgent"
				agents[i] = RandomAgent(i)
			elif x == 1:
				names[i] = "SimpleRuleAgent"
				agents[i] = SimpleRuleAgent(i)
			elif x == 2:
				names[i] = "SmarterRuleAgent"
				agents[i] = SmarterRuleAgent(i)
			elif x == 3:
				names[i] = "GeniusRuleAgent"
				agents[i] = GeniusRuleAgent(i)
			elif x == 4:
				names[i] = "BoxFarmerAgent"
				agents[i] = BoxFarmerAgent(i)
			else:
				names[i] = "TacticalRuleAgent"
				agents[i] = TacticalRuleAgent(i)
		elif path == "RandomAgent":
			names[i] = "RandomAgent"
			agents[i] = RandomAgent(i)
		elif path == "SimpleRuleAgent":
			names[i] = "SimpleRuleAgent"
			agents[i] = SimpleRuleAgent(i)
		elif path == "SmarterRuleAgent":
			names[i] = "SmarterRuleAgent"
			agents[i] = SmarterRuleAgent(i)
		elif path == "GeniusRuleAgent":
			names[i] = "GeniusRuleAgent"
			agents[i] = GeniusRuleAgent(i)
		elif path == "BoxFarmerAgent":
			names[i] = "BoxFarmerAgent"
			agents[i] = BoxFarmerAgent(i)
		elif path == "TacticalRuleAgent":
			names[i] = "TacticalRuleAgent"
			agents[i] = TacticalRuleAgent(i)
		else:
			# Custom agent path
			p = Path(path)
			if p.is_dir():
				p = p / "agent.py"
			if not p.exists():
				raise FileNotFoundError(f"Agent file not found: {p}")
			
			try:
				agents[i] = load_agent_instance(str(p), i)
				if hasattr(agents[i], "team_id"):
					names[i] = agents[i].team_id
				else:
					names[i] = p.parent.name if p.parent.name else p.name
			except Exception as e:
				raise RuntimeError(f"Failed to load agent from {p}: {e}")

	return agents, names


def clone_obs(obs):
	return {
		"map": np.array(obs["map"], copy=True),
		"players": np.array(obs["players"], copy=True),
		"bombs": np.array(obs["bombs"], copy=True),
	}


def simulate_episodes(agent_paths, num_episodes=10, max_steps=500, seed=None, model_variants=None):
	env = BomberEnv(max_steps=max_steps)
	agents, names = make_agents(agent_paths, seed=seed)
	
	episodes = []
	num_agents = len(agents)

	for episode in range(num_episodes):
		episode_seed = None if seed is None else seed + episode
		obs = env.reset(seed=episode_seed)
		done = False
		step = 0
		
		# Initialize cumulative rewards
		cum_rewards = [0.0] * num_agents
		
		first_obs = clone_obs(obs)
		first_obs["rewards"] = [0.0] * num_agents
		first_obs["cum_rewards"] = [0.0] * num_agents
		first_obs["actions"] = [0] * num_agents
		first_obs["q_values"] = [None] * num_agents
		first_obs["breakdowns"] = [None] * num_agents
		trajectory = [first_obs]

		while not done and step < max_steps:
			actions = []
			step_q_values = []
			for i in range(num_agents):
				try:
					action = agents[i].act(obs)
					q_vals = getattr(agents[i], "last_q_values", None)
				except Exception as e:
					print(f"Agent {names[i]} failed to act: {e}")
					action = 0
					q_vals = None
				actions.append(action)
				step_q_values.append(q_vals)
			
			# Store actions and q-values for the state we were in when making the decision
			trajectory[-1]["actions"] = actions
			trajectory[-1]["q_values"] = step_q_values
			
			prev_obs = obs
			obs, terminated, truncated = env.step(actions)
			
			# Compute reward for transition
			step_rewards = []
			step_breakdowns = []
			for i in range(num_agents):
				rew = 0.0
				bd = None
				if compute_reward is not None:
					try:
						rew, bd = compute_reward(prev_obs, obs, i, return_breakdown=True)
					except Exception:
						try:
							rew = compute_reward(prev_obs, obs, i)
						except Exception:
							pass
				step_rewards.append(rew)
				step_breakdowns.append(bd)
				cum_rewards[i] += rew
				
			obs_cloned = clone_obs(obs)
			obs_cloned["rewards"] = step_rewards
			obs_cloned["cum_rewards"] = list(cum_rewards)
			obs_cloned["actions"] = [0] * num_agents
			obs_cloned["q_values"] = [None] * num_agents
			obs_cloned["breakdowns"] = step_breakdowns
			
			trajectory.append(obs_cloned)
			done = terminated or truncated
			step += 1

		episodes.append(trajectory)

	return episodes, names


def run_simple_viewer(agent_paths, num_episodes=10, max_steps=100, seed=None, autoplay=True, model_variants=None):
	episodes, agent_names = simulate_episodes(
		agent_paths=agent_paths,
		num_episodes=num_episodes,
		max_steps=max_steps,
		seed=seed,
		model_variants=model_variants,
	)
	if not episodes:
		print("No episodes to display.")
		return

	first_obs = episodes[0][0]
	viewer = Viewer(width=first_obs["map"].shape[1], height=first_obs["map"].shape[0])

	print("Agents:", ", ".join(agent_names))
	print("Controls: A/D step, W/S episode, SPACE pause/play, ESC quit")

	episode_idx = 0
	step_idx = 0
	paused = not autoplay
	last_tick = time.time()

	running = True
	while running:
		now = time.time()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_SPACE:
					paused = not paused
				elif event.key == pygame.K_d:
					step_idx = min(step_idx + 1, len(episodes[episode_idx]) - 1)
					paused = True
				elif event.key == pygame.K_a:
					step_idx = max(step_idx - 1, 0)
					paused = True
				elif event.key == pygame.K_s:
					episode_idx = min(episode_idx + 1, len(episodes) - 1)
					step_idx = 0
				elif event.key == pygame.K_w:
					episode_idx = max(episode_idx - 1, 0)
					step_idx = 0

		if not paused and (now - last_tick) >= (1 / max(viewer.fps, 1)):
			if step_idx < len(episodes[episode_idx]) - 1:
				step_idx += 1
			else:
				paused = True
			last_tick = now

		current_obs = episodes[episode_idx][step_idx]
		prev_obs = episodes[episode_idx][step_idx - 1] if step_idx > 0 else None
		viewer.render(
			obs=current_obs,
			prev_obs=prev_obs,
			episode_idx=episode_idx,
			total_episodes=len(episodes),
			step_idx=step_idx,
			total_steps=len(episodes[episode_idx]),
			paused=paused,
			agent_names=agent_names,
		)

	viewer.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Local viewer for agents."
	)
	parser.add_argument("--agent_paths", nargs="+", default=["None", "None", "None", "None"])
	parser.add_argument("--num_episodes", type=int, default=10)
	parser.add_argument("--max_steps", type=int, default=500)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--autoplay", type=str2bool, default=True)
	args = parser.parse_args()

	run_simple_viewer(
		agent_paths=args.agent_paths,
		num_episodes=args.num_episodes,
		max_steps=args.max_steps,
		seed=args.seed,
		autoplay=args.autoplay,
	)
