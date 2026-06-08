"""Hybrid Reward v5 — Smooth Conditional Reward Shaping for Bomberland DQN.

Continuous dynamic reward multipliers based on bombs_left:
  - hunt_w = min(bombs_left / 2, 1.0)  — linear ramp [0→1]
  - farm_w = 1 - hunt_w                — complementary
  - No hard threshold discontinuity; rewards scale smoothly.

v5 additions:
  - Competitive item pickup: bonus when agent is closer to item than enemies
  - Chain bomb detection: high reward for placing bombs that trigger chain explosions

IMPORTANT: No engine imports. All map constants hard-coded for submission.
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
    # ── Terminal: Death penalty PHẢI áp đảo mọi combo thưởng đặt bom ──
    "win": 5.0,                  # ↑ 3.0→5.0: Thưởng chiến thắng phải là mục tiêu tối thượng
    "enemy_death": 2.5,          # Giữ nguyên
    "agent_death": -5.0,         # ↑↑ -2.5→-5.0: PHẢI lớn hơn tổng combo bomb (~2.0)

    # ── Di chuyển & Chống núp lùm ──
    "standing_still": -0.05,     # ↓ -0.10→-0.05: Giảm để agent có thể suy nghĩ khi trong danger
    "time_penalty": -0.02,       # ↓ -0.03→-0.02: Giảm nhẹ noise nền

    # ── Chiến đấu — GIẢM để không lấn át death penalty ──
    "bomb_plant_base": 0.05,     # ↓ 0.10→0.05: Thưởng nhẹ cho hành vi đặt bom
    "plant_near_box": 0.20,      # ↓ 0.40→0.20: Giảm để tổng stack không quá cao
    "plant_near_enemy": 0.30,    # ↓ 0.50→0.30: Giảm nhưng vẫn khuyến khích
    "box_destroyed": 0.60,       # ↓ 0.80→0.60: Thưởng thực tế khi phá hòm
    "safe_bomb_plant": 0.30,     # ↓ 0.70→0.30: Giảm mạnh — tránh stack quá cao
    "suicide_bomb_plant": -2.0,  # ↑↑ -0.80→-2.0: PHẢI phạt nặng đặt bom tự sát
    "chain_bomb_plant": 0.50,    # ↓ 0.80→0.50: Giảm nhẹ

    # ── ★ MỚI: Post-Bomb Escape — Cơ chế né bom sau khi đặt ──
    "post_bomb_escape": 0.40,    # ★ Thưởng LỚN khi thoát blast zone bom mình vừa đặt
    "post_bomb_linger": -0.15,   # ★ Phạt LEO THANG mỗi step còn trong blast zone bom mình

    # ── Kinh tế & Vật phẩm ──
    "item_collection": 0.80,     # ↓ 1.0→0.80: Giảm nhẹ
    "approach_item": 0.10,       # ↓ 0.15→0.10
    "item_compete_bonus": 0.15,  # ↓ 0.20→0.15
    "survival_bonus": 0.02,      # ↑ 0.01→0.02: Tăng nhẹ thưởng sống sót

    # ── Nhận biết nguy hiểm — TĂNG MẠNH ──
    "danger_evasion": 0.40,      # ↑↑ 0.20→0.40: Thưởng mạnh khi né thoát vùng nguy hiểm
    "danger_enter": -0.20,       # ↑ -0.10→-0.20: Phạt mạnh hơn khi lao vào danger
    "own_blast_loiter": -0.10,   # ↑ -0.05→-0.10: Phạt mạnh hơn khi lảng vảng cạnh bom mình

    # ── Định vị không gian ──
    "approach_enemy": 0.08,      # ↓ 0.10→0.08: Giảm nhẹ
}

# =====================================================================
# 2. CÁC HÀM BỔ TRỢ HÌNH HỌC & GIẢ LẬP TÌM ĐƯỜNG BẰNG BFS
# =====================================================================
def _parse_bomb_row(b):
    """Phân tích mảng bom từ obs thành tuple có cấu trúc."""
    arr = np.asarray(b, dtype=np.float64).ravel()
    if arr.size < 2:
        return None
    bx, by = int(arr[0]), int(arr[1])
    timer = int(arr[2]) if arr.size > 2 else _DEFAULT_BOMB_TIMER
    owner_id = int(arr[3]) if arr.size > 3 else _DEFAULT_BOMB_OWNER
    return bx, by, timer, owner_id


def _bomb_radius_from_obs(players, owner_id):
    """Tính bán kính nổ thực tế (Bán kính gốc 1 + bonus từ item)."""
    if owner_id < len(players):
        return 1 + int(players[int(owner_id)][4])
    return 1


def _explosion_tiles_for_bomb(grid, bx, by, radius):
    """Vẽ vùng nguy hiểm hình chữ thập của quả bom, bị chặn bởi tường và hòm."""
    h, w = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == WALL:  # Tường không thể phá hủy -> chặn vụ nổ
                break
            tiles.add((tx, ty))
            if cell == BOX:   # Hòm gỗ bị phá hủy và chặn vụ nổ tại đó
                break
    return tiles


def _get_danger_tiles(grid, bombs, players):
    """Trích xuất từ Tactical Agent: Trả về tập hợp các ô đang bị bom đe dọa."""
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
        if timer <= 1:  # Sắp nổ trong step tiếp theo
            danger_now |= blast
    return danger_soon, danger_now


def _can_escape_after_placing_bfs(grid, my_pos, occupied_enemies, danger_soon, bomb_radius,
                                   bomb_timer=_DEFAULT_BOMB_TIMER):
    """
    THUẬT TOÁN ĐẮT GIÁ TỪ TACTICAL AGENT (CẢI TIẾN v2):
    Giả lập đặt bom tại chỗ và chạy BFS xem có KỊP thoát trước khi nổ không.
    v2: depth limit = min(timer - 1, 8) thay vì cố định 8.
    """
    # 1. Tự vẽ vụ nổ giả định của quả bom mình định đặt tại vị trí hiện tại
    my_blast = _explosion_tiles_for_bomb(grid, my_pos[0], my_pos[1], bomb_radius)
    # Vùng nguy hiểm hỗn hợp = Bom hiện có trên sân + Quả bom mình định đặt
    combined_danger = set(danger_soon) | my_blast

    # 2. BFS với depth limit = timer - 1 (phải thoát TRƯỚC khi nổ)
    max_depth = min(bomb_timer - 1, 8)
    q = deque([(my_pos, 0)])
    seen = {my_pos}
    while q:
        pos, depth = q.popleft()
        if pos not in combined_danger and depth > 0:
            return True  # Tìm thấy lối thoát an toàn VÀ kịp thời gian!
        if depth >= max_depth:
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                # Ô có thể đi được (Cỏ, Item) và không bị đối thủ chặn chân
                if grid[nx, ny] in [0, ITEM_RADIUS, ITEM_CAPACITY] and (nx, ny) not in occupied_enemies:
                    if (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append(((nx, ny), depth + 1))
    return False  # Đặt bom ở đây đồng nghĩa với tự sát!


def _manhattan_to_nearest_item(grid, x, y):
    """Tìm khoảng cách Manhattan ngắn nhất tới Vật phẩm gần nhất (mã 3 hoặc 4)."""
    ix, iy = int(x), int(y)
    item_positions = np.argwhere((grid == ITEM_RADIUS) | (grid == ITEM_CAPACITY))
    if len(item_positions) == 0:
        return None
    distances = [abs(ix - ip[0]) + abs(iy - ip[1]) for ip in item_positions]
    return min(distances)


def _competitive_item_advantage(grid, players, agent_id, ax, ay):
    """Tính lợi thế cạnh tranh item: thưởng khi agent gần item hơn TẤT CẢ đối thủ.

    Returns:
        Tổng điểm advantage cho tất cả items mà agent gần hơn địch.
        Mỗi item cho điểm = (enemy_dist - agent_dist) / max_dist, capped [0, 1].
        Trả về 0.0 nếu không có item hoặc không có lợi thế.
    """
    item_positions = np.argwhere((grid == ITEM_RADIUS) | (grid == ITEM_CAPACITY))
    if len(item_positions) == 0:
        return 0.0

    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Gom tọa độ đối thủ còn sống
    enemies = []
    for pid in range(arr.shape[0]):
        if pid != agent_id and int(arr[pid][2]) == 1:
            enemies.append((int(arr[pid][0]), int(arr[pid][1])))
    if not enemies:
        return 0.0  # Không có đối thủ → không có cạnh tranh

    max_dist = float(grid.shape[0] + grid.shape[1])  # Chuẩn hóa
    total_advantage = 0.0

    for ip in item_positions:
        ix, iy = int(ip[0]), int(ip[1])
        my_dist = abs(ax - ix) + abs(ay - iy)

        # Khoảng cách gần nhất của BẤT KỲ đối thủ nào tới item này
        min_enemy_dist = min(abs(ex - ix) + abs(ey - iy) for ex, ey in enemies)

        # Agent gần hơn → advantage dương
        if my_dist < min_enemy_dist:
            advantage = (min_enemy_dist - my_dist) / max_dist
            total_advantage += min(advantage, 1.0)

    return total_advantage


def _detect_chain_bomb(grid, players, bomb_x, bomb_y, bomb_radius, existing_bombs):
    """Phát hiện chuỗi nổ lan: bom mới đặt có blast zone chạm bom khác không?

    Cơ chế chain explosion trong Bomberland:
      Khi bom A nổ, vùng blast chạm tới bom B → bom B nổ ngay lập tức.
      Agent đặt bom ở vị trí mà blast zone bao phủ bom khác = tạo chain.

    Args:
        grid: bản đồ hiện tại
        players: mảng thông tin người chơi
        bomb_x, bomb_y: vị trí bom vừa đặt
        bomb_radius: bán kính nổ của bom vừa đặt
        existing_bombs: mảng numpy các bom đang trên sân (trước khi đặt bom mới)

    Returns:
        chain_count: số bom bị chạm bởi blast zone (0 = không chain)
    """
    if existing_bombs is None:
        return 0
    arr = np.asarray(existing_bombs)
    if arr.size == 0:
        return 0
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Vẽ blast zone của bom mới đặt
    new_blast = _explosion_tiles_for_bomb(grid, bomb_x, bomb_y, bomb_radius)

    chain_count = 0
    seen_chains = set()  # Tránh đếm trùng cùng 1 bom

    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        if (bx, by) == (bomb_x, bomb_y):
            continue  # Bỏ qua chính nó

        # Forward: bom cũ nằm trong blast zone bom mới → chain!
        if (bx, by) in new_blast and (bx, by) not in seen_chains:
            chain_count += 1
            seen_chains.add((bx, by))

        # Reverse: bom mới nằm trong blast zone bom cũ → cũng chain!
        if (bx, by) not in seen_chains:
            old_radius = _bomb_radius_from_obs(players, owner_id)
            old_blast = _explosion_tiles_for_bomb(grid, bx, by, old_radius)
            if (bomb_x, bomb_y) in old_blast:
                chain_count += 1
                seen_chains.add((bx, by))

    return chain_count


def _enemy_alive_count(players, agent_id):
    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return sum(1 for pid in range(arr.shape[0]) if pid != agent_id and int(arr[pid][2]) == 1)


def _manhattan_to_nearest_alive_enemy(players, agent_id, x, y):
    best = None
    ix, iy = int(x), int(y)
    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for pid in range(arr.shape[0]):
        if pid == agent_id or int(arr[pid][2]) != 1:
            continue
        d = abs(ix - int(arr[pid][0])) + abs(iy - int(arr[pid][1]))
        best = d if best is None else min(best, d)
    return best


def _min_own_blast_timer_at(obs, agent_id, x, y):
    bombs = obs["bombs"]
    if bombs is None: return None
    arr = np.asarray(bombs)
    if arr.size == 0 or arr.ndim == 1: return None

    players = obs["players"]
    grid = obs["map"]
    best = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None: continue
        bx, by, timer, owner_id = parsed
        if int(owner_id) != int(agent_id): continue
        radius = _bomb_radius_from_obs(players, owner_id)
        if (int(x), int(y)) in _explosion_tiles_for_bomb(grid, bx, by, radius):
            best = int(timer) if best is None else min(best, int(timer))
    return best


def _get_own_blast_zone(obs, agent_id):
    """Trả về tập hợp tất cả ô bị đe dọa bởi bom do agent sở hữu.

    Dùng cho cơ chế Post-Bomb Escape: thưởng khi agent thoát khỏi
    blast zone bom của mình sau khi đặt.
    """
    bombs = obs["bombs"]
    if bombs is None:
        return set()
    arr = np.asarray(bombs)
    if arr.size == 0:
        return set()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    players = obs["players"]
    grid = obs["map"]
    own_blast = set()

    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        if int(owner_id) != int(agent_id):
            continue
        radius = _bomb_radius_from_obs(players, owner_id)
        own_blast |= _explosion_tiles_for_bomb(grid, bx, by, radius)

    return own_blast


# =====================================================================
# 3. HÀM TÍNH PHẦN THƯỞNG CHÍNH (CONDITIONAL HYBRID REWARD FUNCTION)
# =====================================================================
def compute_reward(prev_obs, curr_obs, agent_id):
    """Compute shaped reward with smooth continuous multipliers.

    Smooth Conditional Reward Shaping (no hard threshold):
      hunt_w = min(bombs_left / 2.0, 1.0)  → [0, 1] linear ramp
      farm_w = 1.0 - hunt_w                → [1, 0] complementary

      approach_enemy  *= (1 + 2 × hunt_w)  → [1x, 3x]
      enemy_death     *= (1 + 1 × hunt_w)  → [1x, 2x]
      time_penalty    *= (1 + 1 × hunt_w)  → [1x, 2x]
      approach_item   *= (1 + 2 × farm_w)  → [1x, 3x]
      box_destroyed   *= (1 + 1 × farm_w)  → [1x, 2x]
      danger_evasion  *= (1 + 0.5 × farm_w) → [1x, 1.5x]
    """
    if prev_obs is None:
        return 0.0

    prev_players = prev_obs["players"]
    curr_players = curr_obs["players"]

    prev_alive = int(prev_players[agent_id][2])
    curr_alive = int(curr_players[agent_id][2])

    # -----------------------------------------------------------------
    # LUẬT SỐNG CÒN TỐI CAO (TERMINAL STATES)
    # -----------------------------------------------------------------
    if prev_alive == 1 and curr_alive == 0:
        return float(REWARD_DICT["agent_death"])  # Chết là dính phạt nặng, lập tức ngắt hàm

    reward = 0.0

    # ── Smooth Conditioning: linear ramp thay vì hard threshold ──
    curr_bombs_left = int(curr_players[agent_id][3])
    hunt_w = min(curr_bombs_left / 2.0, 1.0)  # [0→1], smooth at threshold=2
    farm_w = 1.0 - hunt_w                      # [1→0], complementary

    # Kiểm tra chỉ số hạ gục đối thủ và chiến thắng
    prev_enemies_alive = _enemy_alive_count(prev_players, agent_id)
    curr_enemies_alive = _enemy_alive_count(curr_players, agent_id)

    enemy_death_reward = 0.0
    if curr_enemies_alive < prev_enemies_alive:
        enemy_death_reward = REWARD_DICT["enemy_death"] * (prev_enemies_alive - curr_enemies_alive)
        # ── Smooth: enemy_death [1x → 2x] ──
        enemy_death_reward *= (1.0 + 1.0 * hunt_w)
    reward += enemy_death_reward

    if curr_enemies_alive == 0 and prev_enemies_alive > 0:
        reward += REWARD_DICT["win"]

    # Tọa độ di chuyển của Agent
    prev_x, prev_y = int(prev_players[agent_id][0]), int(prev_players[agent_id][1])
    curr_x, curr_y = int(curr_players[agent_id][0]), int(curr_players[agent_id][1])

    # -----------------------------------------------------------------
    # CHỐNG NÚP LÙM THỤ ĐỘNG (giảm penalty khi đang trong danger zone)
    # -----------------------------------------------------------------
    if prev_x == curr_x and prev_y == curr_y:
        # Giảm penalty standing_still khi trong danger → cho phép agent suy nghĩ
        still_pen = REWARD_DICT["standing_still"]
        if (curr_x, curr_y) in danger_soon_curr:
            still_pen *= 0.3  # Giảm 70% penalty khi đang trong vùng nguy hiểm
        reward += still_pen
    else:
        reward -= REWARD_DICT["standing_still"]  # Di chuyển sẽ triệt tiêu điểm phạt

    # ── Smooth: time_penalty [1x → 2x] ──
    time_pen = REWARD_DICT["time_penalty"] * (1.0 + 1.0 * hunt_w)
    reward += time_pen

    # Lấy thông tin vùng nguy hiểm toàn cục từ thuật toán hình học của Baseline
    danger_soon_prev, danger_now_prev = _get_danger_tiles(prev_obs["map"], prev_obs["bombs"], prev_players)
    danger_soon_curr, _ = _get_danger_tiles(curr_obs["map"], curr_obs["bombs"], curr_players)

    # -----------------------------------------------------------------
    # CƠ CHẾ ĐIỀU HƯỚNG ĂN VẬT PHẨM (MẮT THẦN TOÀN CỤC)
    # -----------------------------------------------------------------
    prev_item_d = _manhattan_to_nearest_item(prev_obs["map"], prev_x, prev_y)
    curr_item_d = _manhattan_to_nearest_item(curr_obs["map"], curr_x, curr_y)

    if prev_item_d is not None and curr_item_d is not None:
        approach_item_reward = REWARD_DICT["approach_item"] * (prev_item_d - curr_item_d)
        # ── Smooth: approach_item [1x → 3x] ──
        approach_item_reward *= (1.0 + 2.0 * farm_w)
        reward += approach_item_reward

    # Thưởng tĩnh khi ăn thành công vật phẩm
    if prev_obs["map"][curr_x, curr_y] in [ITEM_RADIUS, ITEM_CAPACITY]:
        reward += REWARD_DICT["item_collection"]

    # -----------------------------------------------------------------
    # CẠNH TRANH VẬT PHẨM: Thưởng khi agent gần item hơn đối thủ
    # -----------------------------------------------------------------
    item_advantage = _competitive_item_advantage(
        curr_obs["map"], curr_players, agent_id, curr_x, curr_y
    )
    if item_advantage > 0:
        compete_reward = REWARD_DICT["item_compete_bonus"] * item_advantage
        # ── Smooth: item_compete [1x → 2x] ở farming mode ──
        compete_reward *= (1.0 + 1.0 * farm_w)
        reward += compete_reward

    # -----------------------------------------------------------------
    # NÉ BOM SINH TỒN THÔNG MINH
    # -----------------------------------------------------------------
    prev_in_danger = (prev_x, prev_y) in danger_soon_prev
    curr_in_danger = (curr_x, curr_y) in danger_soon_curr

    if prev_in_danger and not curr_in_danger:
        # ── Smooth: danger_evasion [1x → 1.5x] ──
        evasion_reward = REWARD_DICT["danger_evasion"] * (1.0 + 0.5 * farm_w)
        reward += evasion_reward
    elif not prev_in_danger and curr_in_danger and (prev_x != curr_x or prev_y != curr_y):
        reward += REWARD_DICT["danger_enter"]   # Phạt nếu cố tình lao đầu vào vùng nguy hiểm

    # Phạt loiter khi đứng quá gần ngòi nổ quả bom của chính mình
    mt_own = _min_own_blast_timer_at(curr_obs, agent_id, curr_x, curr_y)
    if mt_own is not None:
        # ★ Phạt leo thang theo timer: càng gần nổ càng phạt nặng
        reward += REWARD_DICT["own_blast_loiter"] * float(max(1, 8 - mt_own))

    # -----------------------------------------------------------------
    # ★ MỚI: POST-BOMB ESCAPE — Thưởng/phạt né bom sau khi đặt
    # -----------------------------------------------------------------
    # Kiểm tra bom do agent sở hữu đang trên sân
    _own_blast_prev = _get_own_blast_zone(prev_obs, agent_id)
    _own_blast_curr = _get_own_blast_zone(curr_obs, agent_id)

    prev_in_own = (prev_x, prev_y) in _own_blast_prev if _own_blast_prev else False
    curr_in_own = (curr_x, curr_y) in _own_blast_curr if _own_blast_curr else False

    if prev_in_own and not curr_in_own:
        # Agent vừa thoát khỏi blast zone bom mình → THƯỞNG LỚN
        reward += REWARD_DICT["post_bomb_escape"]
    elif curr_in_own and (prev_x == curr_x and prev_y == curr_y):
        # Agent đứng yên trong blast zone bom mình → PHẠT LEO THANG
        own_timer = _min_own_blast_timer_at(curr_obs, agent_id, curr_x, curr_y)
        urgency = 1.0 / max(own_timer, 1) if own_timer else 0.5
        reward += REWARD_DICT["post_bomb_linger"] * (1.0 + urgency)

    # Thưởng hướng đi tiếp cận dồn ép kẻ địch
    if prev_enemies_alive > 0 and curr_enemies_alive > 0:
        prev_enemy_d = _manhattan_to_nearest_alive_enemy(prev_players, agent_id, prev_x, prev_y)
        curr_enemy_d = _manhattan_to_nearest_alive_enemy(curr_players, agent_id, curr_x, curr_y)
        if prev_enemy_d is not None and curr_enemy_d is not None:
            approach_enemy_reward = REWARD_DICT["approach_enemy"] * (prev_enemy_d - curr_enemy_d)
            # ── Smooth: approach_enemy [1x → 3x] ──
            approach_enemy_reward *= (1.0 + 2.0 * hunt_w)
            reward += approach_enemy_reward

    # -----------------------------------------------------------------
    # ĐẶT BOM ĐƯỢC CHẤM ĐIỂM BỞI THUẬT TOÁN BFS CỦA TACTICAL AGENT
    # -----------------------------------------------------------------
    prev_bombs_left = int(prev_players[agent_id][3])

    if curr_bombs_left < prev_bombs_left:  # Giây phút nút ĐẶT BOM được bấm
        # ★ THƯỞNG CƠ BẢN: Bất kỳ lần đặt bom nào cũng được thưởng nhẹ
        reward += REWARD_DICT["bomb_plant_base"]

        # Gom danh sách đối thủ còn sống để làm chướng ngại vật trong BFS
        enemies_set = {(int(p[0]), int(p[1])) for i, p in enumerate(prev_players) if i != agent_id and p[2] == 1}
        my_radius = _bomb_radius_from_obs(prev_players, agent_id)

        # Gọi thuật toán giả lập tìm đường sống của Tactical Baseline
        is_safe = _can_escape_after_placing_bfs(prev_obs["map"], (curr_x, curr_y), enemies_set, danger_soon_prev, my_radius)

        if is_safe:
            reward += REWARD_DICT["safe_bomb_plant"]  # Đặt bom an toàn, có đường lui

            # Thưởng thêm nếu vị trí đặt bom này mang lại giá trị kinh tế (nằm cạnh hòm gỗ)
            adjacent_cells = [
                prev_obs["map"][max(0, curr_x - 1), curr_y],
                prev_obs["map"][min(prev_obs["map"].shape[0] - 1, curr_x + 1), curr_y],
                prev_obs["map"][curr_x, max(0, curr_y - 1)],
                prev_obs["map"][curr_x, min(prev_obs["map"].shape[1] - 1, curr_y + 1)]
            ]
            if BOX in adjacent_cells:
                reward += REWARD_DICT["plant_near_box"]

            # ★ MỚI: Thưởng đặt bom gần kẻ địch (blast zone chạm vị trí enemy)
            my_blast = _explosion_tiles_for_bomb(prev_obs["map"], curr_x, curr_y, my_radius)
            enemy_in_blast = sum(1 for epos in enemies_set if epos in my_blast)
            if enemy_in_blast > 0:
                plant_enemy_reward = REWARD_DICT["plant_near_enemy"] * enemy_in_blast
                plant_enemy_reward *= (1.0 + 1.0 * hunt_w)  # [1x → 2x] hunting mode
                reward += plant_enemy_reward

            # ── CHAIN BOMB: Thưởng chuỗi nổ lan ──
            chain_count = _detect_chain_bomb(
                prev_obs["map"], prev_players,
                curr_x, curr_y, my_radius,
                prev_obs["bombs"]
            )
            if chain_count > 0:
                # Thưởng cao × số bom bị chain, nhân thêm hunt_w vì đây là chiến thuật tấn công
                chain_reward = REWARD_DICT["chain_bomb_plant"] * chain_count
                chain_reward *= (1.0 + 1.0 * hunt_w)  # [1x → 2x] ở hunting mode
                reward += chain_reward
        else:
            reward += REWARD_DICT["suicide_bomb_plant"]  # Đặt bom tự sát → phạt (đã giảm nhẹ)

    # Ghi nhận thành quả phá hòm thực tế sau vụ nổ
    prev_boxes = int(np.sum(prev_obs["map"] == BOX))
    curr_boxes = int(np.sum(curr_obs["map"] == BOX))
    if curr_boxes < prev_boxes:
        box_reward = REWARD_DICT["box_destroyed"] * (prev_boxes - curr_boxes)
        # ── Smooth: box_destroyed [1x → 2x] ──
        box_reward *= (1.0 + 1.0 * farm_w)
        reward += box_reward

    # Thưởng sống sót
    reward += REWARD_DICT["survival_bonus"]

    return float(reward)