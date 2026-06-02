"""Enhanced observation encoding for Bomberland DQN v2.

Produces 13 spatial channels + 7 auxiliary scalars (vs. original 9+3).
New channels: blast zone prediction, danger heatmap, box adjacency, BFS reachability.
"""
import numpy as np
from collections import deque

BOMB_MAX_TIMER = 7

class _Map:
    GRASS = 0; WALL = 1; BOX = 2; ITEM_RADIUS = 3; ITEM_CAPACITY = 4

class _Player:
    MAX_BOMB_RADIUS = 5; MAX_BOMB_CAPACITY = 5


def _blast_tiles(grid, bx, by, radius):
    H, W = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx*r, by + dy*r
            if not (0 <= tx < H and 0 <= ty < W):
                break
            c = int(grid[tx, ty])
            if c == _Map.WALL:
                break
            tiles.add((tx, ty))
            if c == _Map.BOX:
                break
    return tiles


def encode_obs(obs, agent_ids):
    """Encode observation into (13, H, W) map tensor + (7,) auxiliary vector."""
    if obs is None:
        raise ValueError("obs should not be None")

    uid = int(agent_ids[0])
    grid = obs["map"]
    players = obs["players"]
    bombs = obs["bombs"]
    H, W = grid.shape
    n_players = len(players)
    opp_ids = [int(agent_ids[i]) for i in range(1, len(agent_ids))] if len(agent_ids) > 1 \
              else [i for i in range(n_players) if i != uid]

    # --- Spatial channels ---
    chs = [(grid == v).astype(np.float32) for v in range(5)]  # 0-4: one-hot map

    my_x, my_y = int(players[uid][0]), int(players[uid][1])
    my_alive = int(players[uid][2])
    my_pos = np.zeros((H, W), dtype=np.float32)
    if my_alive:
        my_pos[my_x, my_y] = 1.0
    chs.append(my_pos)  # 5

    enemy_pos = np.zeros((H, W), dtype=np.float32)
    n_alive = 0
    min_dist = 999
    for oid in opp_ids:
        if int(players[oid][2]) == 1:
            ex, ey = int(players[oid][0]), int(players[oid][1])
            enemy_pos[ex, ey] = 1.0
            n_alive += 1
            if my_alive:
                min_dist = min(min_dist, abs(my_x - ex) + abs(my_y - ey))
    if min_dist == 999:
        min_dist = 0
    chs.append(enemy_pos)  # 6

    btimer = np.zeros((H, W), dtype=np.float32)
    bowned = np.zeros((H, W), dtype=np.float32)
    bzone = np.zeros((H, W), dtype=np.float32)
    dheat = np.zeros((H, W), dtype=np.float32)
    n_bombs = 0
    in_blast = 0.0

    barr = np.asarray(bombs)
    if barr.size > 0:
        if barr.ndim == 1:
            barr = barr.reshape(1, -1)
        for b in barr:
            bx, by = int(b[0]), int(b[1])
            timer = float(b[2]) if len(b) > 2 else 7.0
            owner = int(b[3]) if len(b) > 3 else 0
            n_bombs += 1
            btimer[bx, by] = max(btimer[bx, by], timer / BOMB_MAX_TIMER)
            bowned[bx, by] = 1.0 if owner == uid else -1.0
            rad = 1 + int(players[owner][4]) if owner < n_players else 1
            for tx, ty in _blast_tiles(grid, bx, by, rad):
                bzone[tx, ty] = 1.0
                dheat[tx, ty] = max(dheat[tx, ty], 1.0 / max(timer, 0.5))
        if my_alive and bzone[my_x, my_y] > 0:
            in_blast = 1.0

    chs.append(btimer)  # 7
    chs.append(bowned)  # 8
    chs.append(bzone)   # 9
    chs.append(dheat)   # 10

    # Box adjacency
    badj = np.zeros((H, W), dtype=np.float32)
    bm = (grid == _Map.BOX).astype(np.float32)
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        badj += np.roll(np.roll(bm, dx, 0), dy, 1)
    chs.append(badj / 4.0)  # 11

    # BFS reachability
    reach = np.zeros((H, W), dtype=np.float32)
    if my_alive:
        q = deque([(my_x, my_y, 0)])
        vis = {(my_x, my_y)}
        md = float(H + W)
        bomb_set = set()
        if barr.size > 0:
            for b in barr:
                bomb_set.add((int(b[0]), int(b[1])))
        while q:
            cx, cy, d = q.popleft()
            reach[cx, cy] = 1.0 - d / md
            for ddx, ddy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx+ddx, cy+ddy
                if 0 <= nx < H and 0 <= ny < W and (nx,ny) not in vis:
                    if int(grid[nx,ny]) not in (1,2) and (nx,ny) not in bomb_set:
                        vis.add((nx,ny))
                        q.append((nx, ny, d+1))
    chs.append(reach)  # 12

    map_feat = np.stack(chs, axis=0)  # (13, H, W)

    scalar = np.array([
        float(players[uid][3]) / _Player.MAX_BOMB_CAPACITY,
        float(players[uid][4]) / _Player.MAX_BOMB_RADIUS,
        n_alive / max(len(opp_ids), 1),
        min_dist / 24.0,
        in_blast,
        n_bombs / 10.0,
        float(my_alive),
    ], dtype=np.float32)

    return map_feat, scalar
