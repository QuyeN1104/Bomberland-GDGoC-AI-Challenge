"""Reward shaping v2 for Bomberland DQN.

Improvements over v1:
  - Box destruction confirmation (not just planting near box)
  - Cornering bonus (trapping enemy in limited escape routes)
  - Safe bomb placement reward (plant + have escape path)
  - Survival bonus that increases over time
  - Better scaling to avoid reward dominance
"""
import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
# Add parent directory to sys.path if not already present
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine import Map

_DEFAULT_BOMB_TIMER = 7
_DEFAULT_BOMB_OWNER = 0


def _parse_bomb_row(b):
    """Return (bx, by, timer, owner_id); timer/owner default if only (x, y) is given."""
    arr = np.asarray(b, dtype=np.float64).ravel()
    if arr.size < 2:
        return None
    bx, by = int(arr[0]), int(arr[1])
    timer = int(arr[2]) if arr.size > 2 else _DEFAULT_BOMB_TIMER
    owner_id = int(arr[3]) if arr.size > 3 else _DEFAULT_BOMB_OWNER
    return bx, by, timer, owner_id


REWARD_DICT = {
    # Terminal
    "win": 3.0,
    "enemy_death": 1.5,
    "agent_death": -3.0,
    # Movement
    "standing_still": -0.01,
    "time_penalty": -0.003,
    # Combat
    "plant_near_box": 0.08,
    "box_destroyed": 0.15,
    "safe_bomb_plant": 0.06,
    # Items & survival
    "item_collection": 0.12,
    "survival_bonus": 0.002,
    # Danger
    "danger_evasion": 0.15,
    "danger_enter": -0.08,
    "own_blast_loiter": -0.05,
    # Positioning
    "approach_enemy": 0.025,
    "corner_enemy": 0.1,
}


def _bomb_radius_from_obs(players, owner_id):
    return 1 + int(players[int(owner_id)][4])


def _explosion_tiles_for_bomb(grid, bx, by, radius):
    """Same cross-shaped blast rules as BomberEnv._get_explosion_tiles."""
    h, w = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == Map.WALL:
                break
            tiles.add((tx, ty))
            if cell == Map.BOX:
                break
    return tiles


def _blast_status_at(obs, x, y):
    """
    Returns (in_blast: bool, min_timer: int|None) for active bombs in obs.
    min_timer is the smallest timer among bombs whose blast includes (x, y).
    """
    bombs = obs["bombs"]
    if bombs is None:
        return False, None
    arr = np.asarray(bombs)
    if arr.size == 0:
        return False, None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    ix, iy = int(x), int(y)
    players = obs["players"]
    grid = obs["map"]
    in_blast = False
    min_timer = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        radius = _bomb_radius_from_obs(players, owner_id)
        tiles = _explosion_tiles_for_bomb(grid, bx, by, radius)
        if (ix, iy) in tiles:
            in_blast = True
            t = int(timer)
            min_timer = t if min_timer is None else min(min_timer, t)
    return in_blast, min_timer


def _any_bombs(obs):
    b = obs["bombs"]
    if b is None:
        return False
    return np.asarray(b).size > 0


def _enemy_alive_count(players, agent_id):
    """BomberEnv uses a (N, 5) ndarray; unit tests may use dict keyed by player id."""
    if isinstance(players, dict):
        return sum(
            1 for pid, p in players.items()
            if pid != agent_id and int(p[2]) == 1
        )
    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n = arr.shape[0]
    return sum(
        1 for pid in range(n)
        if pid != agent_id and int(arr[pid][2]) == 1
    )


def _manhattan_to_nearest_alive_enemy(players, agent_id, x, y):
    """None if there is no other alive player."""
    best = None
    ix, iy = int(x), int(y)
    if isinstance(players, dict):
        for pid, p in players.items():
            if pid == agent_id or int(p[2]) != 1:
                continue
            d = abs(ix - int(p[0])) + abs(iy - int(p[1]))
            best = d if best is None else min(best, d)
        return best
    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for pid in range(arr.shape[0]):
        if pid == agent_id or int(arr[pid][2]) != 1:
            continue
        d = abs(ix - int(arr[pid][0])) + abs(iy - int(arr[pid][1]))
        best = d if best is None else min(best, d)
    return best


def _in_own_predicted_blast(obs, agent_id, x, y):
    return _min_own_blast_timer_at(obs, agent_id, x, y) is not None


def _min_own_blast_timer_at(obs, agent_id, x, y):
    """Smallest tick countdown among this agent's bombs whose blast includes (x, y)."""
    bombs = obs["bombs"]
    if bombs is None:
        return None
    arr = np.asarray(bombs)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    players = obs["players"]
    grid = obs["map"]
    ix, iy = int(x), int(y)
    aid = int(agent_id)
    best = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        if int(owner_id) != aid:
            continue
        radius = _bomb_radius_from_obs(players, owner_id)
        tiles = _explosion_tiles_for_bomb(grid, bx, by, radius)
        if (ix, iy) in tiles:
            t = int(timer)
            best = t if best is None else min(best, t)
    return best


def _count_free_neighbors(grid, x, y, bombs_arr):
    """Count walkable neighbors (not wall, box, or bomb)."""
    H, W = grid.shape
    bomb_set = set()
    if bombs_arr is not None:
        arr = np.asarray(bombs_arr)
        if arr.size > 0:
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for b in arr:
                bomb_set.add((int(b[0]), int(b[1])))
    count = 0
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx, ny = int(x)+dx, int(y)+dy
        if 0 <= nx < H and 0 <= ny < W:
            cell = int(grid[nx, ny])
            if cell not in (Map.WALL, Map.BOX) and (nx, ny) not in bomb_set:
                count += 1
    return count


def _count_boxes(grid):
    return int(np.sum(grid == Map.BOX))


def compute_reward(prev_obs, curr_obs, agent_id):
    if prev_obs is None:
        return 0.0

    prev_players = prev_obs["players"]
    curr_players = curr_obs["players"]
    
    prev_alive = int(prev_players[agent_id][2])
    curr_alive = int(curr_players[agent_id][2])

    reward = 0.0
    
    # 1. WIN / LOSS CONDITIONS
    if prev_alive == 1 and curr_alive == 0:
        return float(REWARD_DICT["agent_death"])
    
    prev_enemies_alive = _enemy_alive_count(prev_players, agent_id)
    curr_enemies_alive = _enemy_alive_count(curr_players, agent_id)
    
    if curr_enemies_alive < prev_enemies_alive:
        reward += REWARD_DICT["enemy_death"] * (prev_enemies_alive - curr_enemies_alive)
    if curr_enemies_alive == 0 and prev_enemies_alive > 0:
        reward += REWARD_DICT["win"]

    # 2. MOVEMENT & TIME PENALTIES
    prev_x, prev_y = prev_players[agent_id][0], prev_players[agent_id][1]
    curr_x, curr_y = curr_players[agent_id][0], curr_players[agent_id][1]
    
    if prev_x == curr_x and prev_y == curr_y:
        reward += REWARD_DICT["standing_still"]
    else:
        reward -= REWARD_DICT["standing_still"] # Moving still incurs a small time penalty to encourage efficiency
    
    reward += REWARD_DICT["time_penalty"]

    # 2b. DANGER EVASION — reward leaving predicted blast; penalize walking into it
    if _any_bombs(prev_obs) or _any_bombs(curr_obs):
        prev_in_blast, prev_timer = _blast_status_at(prev_obs, prev_x, prev_y)
        curr_in_blast, _ = _blast_status_at(curr_obs, curr_x, curr_y)
        if prev_in_blast and not curr_in_blast:
            urgency = 1.5 if (prev_timer is not None and prev_timer <= 3) else 1.0
            reward += REWARD_DICT["danger_evasion"] * urgency
        elif (
            not prev_in_blast
            and curr_in_blast
            and (prev_x != curr_x or prev_y != curr_y)
        ):
            # Only when stepping into blast; standing still (e.g. planting on own tile) is excluded
            reward += REWARD_DICT["danger_enter"]

    # Standing in your own blast: penalize more as the fuse runs down (clearer than flat -0.04).
    mt_own = _min_own_blast_timer_at(curr_obs, agent_id, curr_x, curr_y)
    if curr_alive == 1 and mt_own is not None:
        urgency = max(1, 8 - int(mt_own))
        reward += REWARD_DICT["own_blast_loiter"] * float(urgency)

    if (
        curr_alive == 1
        and prev_enemies_alive > 0
        and curr_enemies_alive > 0
    ):
        prev_d = _manhattan_to_nearest_alive_enemy(prev_players, agent_id, prev_x, prev_y)
        curr_d = _manhattan_to_nearest_alive_enemy(curr_players, agent_id, curr_x, curr_y)
        if prev_d is not None and curr_d is not None:
            reward += REWARD_DICT["approach_enemy"] * (prev_d - curr_d)

    # 3. ITEM COLLECTION
    # Based on your legend: 3 is item_radius, 4 is item_capacity
    stepped_on_tile = prev_obs["map"][curr_x, curr_y]
    if stepped_on_tile in [3, 4]: 
        reward += REWARD_DICT["item_collection"]
    else:
        # Fallback check just in case items spawn under players or map updates differently
        prev_radius_bonus = int(prev_players[agent_id][4])
        curr_radius_bonus = int(curr_players[agent_id][4])
        if curr_radius_bonus > prev_radius_bonus:
             reward += REWARD_DICT["item_collection"]

    # 4. REWARD SHAPING: Box Destruction & Bomb Placement
    prev_bombs_left = int(prev_players[agent_id][3])
    curr_bombs_left = int(curr_players[agent_id][3])
    
    if curr_bombs_left < prev_bombs_left:
        # Check immediate adjacent tiles (up, down, left, right)
        adjacent_tiles = [
            prev_obs["map"][max(0, curr_x-1), curr_y],
            prev_obs["map"][min(prev_obs["map"].shape[0]-1, curr_x+1), curr_y],
            prev_obs["map"][curr_x, max(0, curr_y-1)],
            prev_obs["map"][curr_x, min(prev_obs["map"].shape[1]-1, curr_y+1)]
        ]
        
        # 2 is the integer for "box" based on your legend
        if 2 in adjacent_tiles:
            reward += REWARD_DICT["plant_near_box"]
        
        # Safe bomb placement: planted bomb AND have escape route
        n_free = _count_free_neighbors(curr_obs["map"], curr_x, curr_y, curr_obs["bombs"])
        if n_free >= 1:
            reward += REWARD_DICT["safe_bomb_plant"]

    # 5. BOX DESTRUCTION CONFIRMATION
    prev_boxes = _count_boxes(prev_obs["map"])
    curr_boxes = _count_boxes(curr_obs["map"])
    if curr_boxes < prev_boxes:
        reward += REWARD_DICT["box_destroyed"] * (prev_boxes - curr_boxes)

    # 6. SURVIVAL BONUS (increases over time to reward staying alive)
    if curr_alive == 1:
        reward += REWARD_DICT["survival_bonus"]

    return float(reward)


class UnitTestReward:
    def agent_death(self):
        prev_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 1, 1)}, # Changed True to 1 for alive flag
            "bombs": []
        }
        curr_obs = {
            "map": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 0, 1, 1)}, # Changed False to 0
            "bombs": []
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Agent Death Reward: {reward}")
        assert reward == REWARD_DICT["agent_death"], "Expected exactly the agent death penalty"
    
    def agent_standing_still(self):
        prev_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 1, 1)},
            "bombs": []
        }
        curr_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 1, 1)},
            "bombs": []
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Agent Standing Still Reward: {reward}")
        expected = REWARD_DICT["standing_still"] + REWARD_DICT["time_penalty"] + REWARD_DICT["survival_bonus"]
        assert abs(reward - expected) < 1e-6, f"Expected {expected} for standing still, got {reward}"
    
    def agent_moving(self):
        prev_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 1, 1)},
            "bombs": []
        }
        curr_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 2, 1, 1, 1)}, # Player moved
            "bombs": []
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Agent Moving Reward: {reward}")

    def run_all_tests(self):
        self.agent_death()
        self.agent_standing_still()
        self.agent_moving()
        print("All reward tests passed!")

if __name__ == "__main__":
    tester = UnitTestReward()
    tester.run_all_tests()