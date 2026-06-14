"""Hybrid Reward v5.2 — Local Attention & Exponential Shaping for Bomberland DQN.

Continuous dynamic reward multipliers:
  - Exponential Hunt Curve: hunt_w = min(bombs_left / 3.0, 1.0) ** 2
  - Local Bounded BFS: Radar limits for Item (5) and Enemy (8) to reduce global noise.
  - Action Justification: Bomb planting exempt from standing still penalty.
  - FIXED: Linger penalty correctly penalizes "fire dancing" (moving between 2 danger tiles).
  - FIXED: BFS pathfinding treats impending explosions (danger_now) as walls.
"""
import numpy as np
from collections import deque

# ── Hard-coded map cell constants (MUST NOT import from engine) ──
WALL = 1
BOX = 2
ITEM_RADIUS = 3
ITEM_CAPACITY = 4

_DEFAULT_BOMB_TIMER = 7
_DEFAULT_BOMB_OWNER = 0

# =====================================================================
# 1. BẢNG CẤU HÌNH PHẦN THƯỞNG (HYBRID REWARD MATRIX)
# =====================================================================
REWARD_DICT = {
    # ── Terminal ──
    "win": 5.0,                  
    "enemy_death": 3.0,          
    "agent_death": -5.0,         

    # ── Di chuyển & Thuế ──
    "standing_still": -0.05,     
    "time_penalty": -0.02,       

    # ── Chiến đấu ──
    "bomb_plant_base": 0.20,     
    "plant_near_box": 0.40,      
    "plant_near_enemy": 0.60,    
    "box_destroyed": 0.80,       
    "safe_bomb_plant": 1.00,     
    "suicide_bomb_plant": -2.0,  
    "chain_bomb_plant": 0.50,    

    # ── Né bom ──
    "danger_enter": -1.00,       
    "post_bomb_linger": -0.50,   

    # ── Kinh tế & Vật phẩm ──
    "item_collection": 0.80,     
    "approach_item": 0.02,       
    "item_compete_bonus": 0.15,  
    "survival_bonus": 0.02,      

    # ── Định vị không gian ──
    "approach_enemy": 0.02,      
}

# =====================================================================
# 2. CÁC HÀM BỔ TRỢ HÌNH HỌC & RADAR (BOUNDED BFS)
# =====================================================================
def _parse_bomb_row(b):
    arr = np.asarray(b, dtype=np.float64).ravel()
    if arr.size < 2:
        return None
    bx, by = int(arr[0]), int(arr[1])
    timer = int(arr[2]) if arr.size > 2 else _DEFAULT_BOMB_TIMER
    owner_id = int(arr[3]) if arr.size > 3 else _DEFAULT_BOMB_OWNER
    return bx, by, timer, owner_id


def _bomb_radius_from_obs(players, owner_id):
    if owner_id < len(players):
        return 1 + int(players[int(owner_id)][4])
    return 1


def _explosion_tiles_for_bomb(grid, bx, by, radius):
    h, w = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == WALL:
                break
            tiles.add((tx, ty))
            if cell == BOX:
                break
    return tiles


def _get_danger_tiles(grid, bombs, players):
    danger_soon = set()
    danger_now = set()
    if bombs is None:
        return danger_soon, danger_now
    arr = np.asarray(bombs)
    if arr.size == 0:
        return danger_soon, danger_now
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        if timer <= 0:
            continue
        radius = _bomb_radius_from_obs(players, owner_id)
        blast = _explosion_tiles_for_bomb(grid, bx, by, radius)
        danger_soon |= blast
        if timer <= 2:  # Bom sắp nổ trong 1-2 tick tới
            danger_now |= blast
    return danger_soon, danger_now


def _can_escape_after_placing_bfs(grid, my_pos, occupied_enemies, danger_soon, bomb_radius, bomb_timer=_DEFAULT_BOMB_TIMER):
    my_blast = _explosion_tiles_for_bomb(grid, my_pos[0], my_pos[1], bomb_radius)
    combined_danger = set(danger_soon) | my_blast
    max_depth = min(bomb_timer - 1, 8)
    q = deque([(my_pos, 0)])
    seen = {my_pos}
    
    while q:
        pos, depth = q.popleft()
        if pos not in combined_danger and depth > 0:
            return True
        if depth >= max_depth:
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] in [0, ITEM_RADIUS, ITEM_CAPACITY] and (nx, ny) not in occupied_enemies:
                    if (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append(((nx, ny), depth + 1))
    return False


def _bfs_dist_to_nearest_item(grid, start_x, start_y, danger_now, max_radius=5):
    """FIXED: Radar quét Vật phẩm. Né các ô chuẩn bị nổ (danger_now)."""
    item_positions = np.argwhere((grid == ITEM_RADIUS) | (grid == ITEM_CAPACITY))
    if len(item_positions) == 0:
        return None

    items = {(int(p[0]), int(p[1])) for p in item_positions}
    H, W = grid.shape
    q = deque([(int(start_x), int(start_y), 0)])
    vis = {(int(start_x), int(start_y))}
    
    while q:
        cx, cy, d = q.popleft()
        if (cx, cy) in items:
            return d
        if d >= max_radius:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in vis:
                # Không đi vào tường, hòm và các ô sắp nổ lập tức
                if int(grid[nx, ny]) not in (WALL, BOX) and (nx, ny) not in danger_now:
                    vis.add((nx, ny))
                    q.append((nx, ny, d + 1))
    return None


def _bfs_dist_to_nearest_enemy(grid, players, danger_now, agent_id, max_radius=8):
    """FIXED: Radar quét Kẻ địch. Né các ô chuẩn bị nổ (danger_now)."""
    arr = np.asarray(players)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    H, W = grid.shape
    
    enemies = set()
    for pid in range(arr.shape[0]):
        if pid != agent_id and int(arr[pid][2]) == 1:
            enemies.add((int(arr[pid][0]), int(arr[pid][1])))
            
    if not enemies:
        return None 
        
    start_x, start_y = int(arr[agent_id][0]), int(arr[agent_id][1])
    q = deque([(start_x, start_y, 0)])
    vis = {(start_x, start_y)}
    
    while q:
        cx, cy, d = q.popleft()
        if (cx, cy) in enemies:
            return d
        if d >= max_radius:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in vis:
                # Không đi vào tường, hòm và các ô sắp nổ lập tức
                if int(grid[nx, ny]) not in (WALL, BOX) and (nx, ny) not in danger_now:
                    vis.add((nx, ny))
                    q.append((nx, ny, d + 1))
    return None 


def _competitive_item_advantage(grid, players, agent_id, ax, ay):
    item_positions = np.argwhere((grid == ITEM_RADIUS) | (grid == ITEM_CAPACITY))
    if len(item_positions) == 0:
        return 0.0

    arr = np.asarray(players)
    if arr.ndim == 1: arr = arr.reshape(1, -1)

    enemies = []
    for pid in range(arr.shape[0]):
        if pid != agent_id and int(arr[pid][2]) == 1:
            enemies.append((int(arr[pid][0]), int(arr[pid][1])))
    if not enemies:
        return 0.0 

    max_dist = float(grid.shape[0] + grid.shape[1])
    total_advantage = 0.0

    for ip in item_positions:
        ix, iy = int(ip[0]), int(ip[1])
        my_dist = abs(ax - ix) + abs(ay - iy)
        min_enemy_dist = min(abs(ex - ix) + abs(ey - iy) for ex, ey in enemies)
        if my_dist < min_enemy_dist:
            advantage = (min_enemy_dist - my_dist) / max_dist
            total_advantage += min(advantage, 1.0)
    return total_advantage


def _detect_chain_bomb(grid, players, bomb_x, bomb_y, bomb_radius, existing_bombs):
    if existing_bombs is None:
        return 0
    arr = np.asarray(existing_bombs)
    if arr.size == 0:
        return 0
    if arr.ndim == 1: arr = arr.reshape(1, -1)

    new_blast = _explosion_tiles_for_bomb(grid, bomb_x, bomb_y, bomb_radius)
    chain_count = 0
    seen_chains = set()

    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None: continue
        bx, by, timer, owner_id = parsed
        if (bx, by) == (bomb_x, bomb_y): continue 

        if (bx, by) in new_blast and (bx, by) not in seen_chains:
            chain_count += 1
            seen_chains.add((bx, by))

        if (bx, by) not in seen_chains:
            old_radius = _bomb_radius_from_obs(players, owner_id)
            old_blast = _explosion_tiles_for_bomb(grid, bx, by, old_radius)
            if (bomb_x, bomb_y) in old_blast:
                chain_count += 1
                seen_chains.add((bx, by))
    return chain_count


def _enemy_alive_count(players, agent_id):
    arr = np.asarray(players)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    return sum(1 for pid in range(arr.shape[0]) if pid != agent_id and int(arr[pid][2]) == 1)


def _min_blast_timer_at(obs, x, y):
    bombs = obs["bombs"]
    if bombs is None: return None
    arr = np.asarray(bombs)
    if arr.size == 0: return None
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    players = obs["players"]
    grid = obs["map"]
    best = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None: continue
        bx, by, timer, owner_id = parsed
        radius = _bomb_radius_from_obs(players, owner_id)
        if (int(x), int(y)) in _explosion_tiles_for_bomb(grid, bx, by, radius):
            best = int(timer) if best is None else min(best, int(timer))
    return best


def _get_own_blast_zone(obs, agent_id):
    bombs = obs["bombs"]
    if bombs is None: return set()
    arr = np.asarray(bombs)
    if arr.size == 0: return set()
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    players = obs["players"]
    grid = obs["map"]
    own_blast = set()
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None: continue
        bx, by, timer, owner_id = parsed
        if int(owner_id) != int(agent_id): continue
        radius = _bomb_radius_from_obs(players, owner_id)
        own_blast |= _explosion_tiles_for_bomb(grid, bx, by, radius)
    return own_blast


# =====================================================================
# 3. HÀM TÍNH PHẦN THƯỞNG CHÍNH
# =====================================================================
def compute_reward(prev_obs, curr_obs, agent_id, return_breakdown=False):
    if prev_obs is None:
        return (0.0, {}) if return_breakdown else 0.0

    prev_players = prev_obs["players"]
    curr_players = curr_obs["players"]

    prev_alive = int(prev_players[agent_id][2])
    curr_alive = int(curr_players[agent_id][2])

    if prev_alive == 1 and curr_alive == 0:
        death_val = float(REWARD_DICT["agent_death"])
        return (death_val, {"Death": death_val}) if return_breakdown else death_val

    if prev_alive == 0:
        return (0.0, {}) if return_breakdown else 0.0

    reward = 0.0
    breakdown = {}

    prev_ammo = int(prev_players[agent_id][3])
    curr_ammo = int(curr_players[agent_id][3])
    just_planted_action = (curr_ammo < prev_ammo)

    base_ratio = min(curr_ammo / 2.5, 1.0)
    hunt_w = base_ratio ** 2  
    farm_w = 1.0 - hunt_w

    prev_x, prev_y = int(prev_players[agent_id][0]), int(prev_players[agent_id][1])
    curr_x, curr_y = int(curr_players[agent_id][0]), int(curr_players[agent_id][1])

    prev_enemies_alive = _enemy_alive_count(prev_players, agent_id)
    curr_enemies_alive = _enemy_alive_count(curr_players, agent_id)

    enemy_death_reward = 0.0
    if curr_enemies_alive < prev_enemies_alive:
        own_blast = _get_own_blast_zone(prev_obs, agent_id)
        arr_p = np.asarray(prev_players)
        if arr_p.ndim == 1: arr_p = arr_p.reshape(1, -1)
        arr_c = np.asarray(curr_players)
        if arr_c.ndim == 1: arr_c = arr_c.reshape(1, -1)
        
        own_kills = 0
        for pid in range(arr_p.shape[0]):
            if pid == agent_id: continue
            if int(arr_p[pid][2]) == 1 and int(arr_c[pid][2]) == 0:
                ex, ey = int(arr_p[pid][0]), int(arr_p[pid][1])
                if (ex, ey) in own_blast:
                    own_kills += 1
                    
        if own_kills > 0:
            enemy_death_reward = REWARD_DICT["enemy_death"] * own_kills * (1.0 + 1.0 * hunt_w)
            reward += enemy_death_reward
            breakdown["Kill Enemy"] = enemy_death_reward

    if curr_enemies_alive == 0 and prev_enemies_alive > 0:
        win_reward = REWARD_DICT["win"]
        reward += win_reward
        breakdown["Win Match"] = win_reward

    # Trích xuất Danger zones cho hàm BFS và hàm Né bom
    danger_soon_prev, danger_now_prev = _get_danger_tiles(prev_obs["map"], prev_obs["bombs"], prev_players)
    danger_soon_curr, danger_now_curr = _get_danger_tiles(curr_obs["map"], curr_obs["bombs"], curr_players)

    if prev_x == curr_x and prev_y == curr_y:
        if not just_planted_action:
            still_pen = REWARD_DICT["standing_still"]
            if (curr_x, curr_y) in danger_soon_curr:
                still_pen *= 0.3 
            reward += still_pen
            breakdown["Standing Still"] = still_pen
    else:
        still_bonus = -REWARD_DICT["standing_still"]
        reward += still_bonus
        breakdown["Moving"] = still_bonus

    time_pen = REWARD_DICT["time_penalty"] * (1.0 + 1.0 * hunt_w)
    reward += time_pen
    breakdown["Time Penalty"] = time_pen

    # VẬT PHẨM (Tích hợp danger_now vào BFS)
    prev_item_d = _bfs_dist_to_nearest_item(prev_obs["map"], prev_x, prev_y, danger_now_prev, max_radius=5)
    curr_item_d = _bfs_dist_to_nearest_item(curr_obs["map"], curr_x, curr_y, danger_now_curr, max_radius=5)

    if prev_item_d is not None and curr_item_d is not None:
        d_diff = prev_item_d - curr_item_d
        approach_item_reward = REWARD_DICT["approach_item"] * d_diff * farm_w
        if approach_item_reward != 0.0:
            reward += approach_item_reward
            breakdown["Approach Item"] = approach_item_reward

    if prev_obs["map"][curr_x, curr_y] in [ITEM_RADIUS, ITEM_CAPACITY]:
        item_coll_rew = REWARD_DICT["item_collection"]
        reward += item_coll_rew
        breakdown["Collect Item"] = item_coll_rew

    item_advantage = _competitive_item_advantage(curr_obs["map"], curr_players, agent_id, curr_x, curr_y)
    if item_advantage > 0:
        compete_reward = REWARD_DICT["item_compete_bonus"] * item_advantage * (1.0 + 1.0 * farm_w)
        reward += compete_reward
        breakdown["Item Advantage"] = compete_reward

    # NÉ BOM SINH TỒN THÔNG MINH
    prev_in_danger = (prev_x, prev_y) in danger_soon_prev
    curr_in_danger = (curr_x, curr_y) in danger_soon_curr

    if not prev_in_danger and curr_in_danger and not just_planted_action:
        if prev_x != curr_x or prev_y != curr_y:
            enter_danger_pen = REWARD_DICT["danger_enter"]
            reward += enter_danger_pen
            breakdown["Enter Danger"] = enter_danger_pen

    # FIXED: Trò lừa "Múa Lửa" (Bỏ check tọa độ prev_x == curr_x)
    if curr_in_danger:
        if not just_planted_action:
            min_timer = _min_blast_timer_at(curr_obs, curr_x, curr_y)
            urgency = (8.0 / max(min_timer, 1)) if min_timer else 4.0
            linger_penalty = REWARD_DICT["post_bomb_linger"] * urgency
            reward += linger_penalty
            breakdown["Linger Danger"] = linger_penalty

    # TIẾP CẬN KẺ ĐỊCH (Tích hợp danger_now vào BFS)
    if prev_enemies_alive > 0 and curr_enemies_alive > 0:
        prev_enemy_d = _bfs_dist_to_nearest_enemy(prev_obs["map"], prev_players, danger_now_prev, agent_id, max_radius=8)
        curr_enemy_d = _bfs_dist_to_nearest_enemy(curr_obs["map"], curr_players, danger_now_curr, agent_id, max_radius=8)
        
        if prev_enemy_d is not None and curr_enemy_d is not None:
            d_diff = prev_enemy_d - curr_enemy_d
            approach_enemy_reward = REWARD_DICT["approach_enemy"] * d_diff * (1.0 + 2.0 * hunt_w)
            if approach_enemy_reward != 0.0:
                reward += approach_enemy_reward
                breakdown["Approach Enemy"] = approach_enemy_reward

    # ĐẶT BOM
    just_planted = False
    if curr_obs["bombs"] is not None:
        arr_b = np.asarray(curr_obs["bombs"])
        if arr_b.size > 0:
            if arr_b.ndim == 1: arr_b = arr_b.reshape(1, -1)
            for i in range(arr_b.shape[0]):
                parsed = _parse_bomb_row(arr_b[i])
                if parsed is not None:
                    bx, by, timer, owner_id = parsed
                    if bx == prev_x and by == prev_y and owner_id == agent_id and timer == _DEFAULT_BOMB_TIMER:
                        just_planted = True
                        break

    if just_planted:
        plant_base_rew = REWARD_DICT["bomb_plant_base"]
        reward += plant_base_rew
        breakdown["Plant Bomb Base"] = plant_base_rew

        enemies_set = {(int(p[0]), int(p[1])) for i, p in enumerate(prev_players) if i != agent_id and p[2] == 1}
        my_radius = _bomb_radius_from_obs(prev_players, agent_id)

        is_safe = _can_escape_after_placing_bfs(prev_obs["map"], (curr_x, curr_y), enemies_set, danger_soon_prev, my_radius)

        if is_safe:
            safe_plant_rew = REWARD_DICT["safe_bomb_plant"]
            reward += safe_plant_rew
            breakdown["Safe Bomb Plant"] = safe_plant_rew

            adjacent_cells = [
                prev_obs["map"][max(0, curr_x - 1), curr_y],
                prev_obs["map"][min(prev_obs["map"].shape[0] - 1, curr_x + 1), curr_y],
                prev_obs["map"][curr_x, max(0, curr_y - 1)],
                prev_obs["map"][curr_x, min(prev_obs["map"].shape[1] - 1, curr_y + 1)]
            ]
            if BOX in adjacent_cells:
                plant_near_box_rew = REWARD_DICT["plant_near_box"]
                reward += plant_near_box_rew
                breakdown["Plant Near Box"] = plant_near_box_rew

            my_blast = _explosion_tiles_for_bomb(prev_obs["map"], curr_x, curr_y, my_radius)
            enemy_in_blast = sum(1 for epos in enemies_set if epos in my_blast)
            if enemy_in_blast > 0:
                plant_enemy_reward = REWARD_DICT["plant_near_enemy"] * enemy_in_blast * (1.0 + 1.0 * hunt_w) 
                reward += plant_enemy_reward
                breakdown["Plant Near Enemy"] = plant_enemy_reward

            chain_count = _detect_chain_bomb(prev_obs["map"], prev_players, curr_x, curr_y, my_radius, prev_obs["bombs"])
            if chain_count > 0:
                chain_reward = REWARD_DICT["chain_bomb_plant"] * chain_count * (1.0 + 1.0 * hunt_w) 
                reward += chain_reward
                breakdown["Chain Bomb Plant"] = chain_reward
        else:
            suicide_rew = REWARD_DICT["suicide_bomb_plant"]
            reward += suicide_rew
            breakdown["Suicide Bomb Plant"] = suicide_rew

    # PHÁ HÒM
    prev_map = prev_obs["map"]
    curr_map = curr_obs["map"]
    own_blast = _get_own_blast_zone(prev_obs, agent_id)

    if own_blast:
        own_boxes_destroyed = 0
        box_mask = (prev_map == BOX) & (curr_map != BOX)
        destroyed_positions = np.argwhere(box_mask)
        for pos in destroyed_positions:
            if (int(pos[0]), int(pos[1])) in own_blast:
                own_boxes_destroyed += 1

        if own_boxes_destroyed > 0:
            box_reward = REWARD_DICT["box_destroyed"] * own_boxes_destroyed * (1.0 + 1.0 * farm_w)
            reward += box_reward
            breakdown["Destroy Box"] = box_reward

    surv_bonus = REWARD_DICT["survival_bonus"]
    reward += surv_bonus
    breakdown["Survival Bonus"] = surv_bonus

    return (float(reward), breakdown) if return_breakdown else float(reward)