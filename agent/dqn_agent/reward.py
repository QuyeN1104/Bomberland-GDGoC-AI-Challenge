import numpy as np
import sys
from pathlib import Path

# Tự động cấu hình đường dẫn hệ thống
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine import Map

_DEFAULT_BOMB_TIMER = 7
_DEFAULT_BOMB_OWNER = 0

# =====================================================================
# 1. BẢNG CẤU HÌNH PHẦN THƯỞNG (REWARD DICTIONARY) - ĐÃ CÂN BẰNG LẠI SCALE
# =====================================================================
REWARD_DICT = {
    # Điều kiện kết thúc (Terminal)
    "win": 3.0,             # Thưởng tối cao khi thắng trận
    "enemy_death": 2.0,     # Tăng mạnh (từ 1.5) để kích thích Agent chủ động đi săn
    "agent_death": -2.0,    # Giảm nhẹ (từ -3.0) để Agent bớt "sợ chết" mà không dám mạo hiểm

    # Di chuyển & Hiệu suất thời gian
    "standing_still": -0.05, # Phạt nặng hơn (từ -0.01) để triệt tiêu hành vi núp lùm cố định
    "time_penalty": -0.005,  # Phạt thời gian trên mỗi bước đi để ép Agent chạy nhanh hơn

    # Chiến đấu & Khai thác tài nguyên
    "plant_near_box": 0.15,  # Tăng mạnh (từ 0.08) để kích thích đặt bom khai phá đường đi
    "box_destroyed": 0.35,   # Tăng mạnh (từ 0.15) để ghi nhận thành quả phá hòm
    "safe_bomb_plant": 0.10, # Thưởng khi đặt bom ở vị trí an toàn, có đường lui

    # Vật phẩm (Kinh tế)
    "item_collection": 0.60, # TĂNG ĐỘT BIẾN (từ 0.12) để biến Vật phẩm thành mục tiêu siêu giá trị
    "approach_item": 0.05,   # THÊM MỚI: Thưởng động trên từng bước nếu đi lại gần Vật phẩm
    "survival_bonus": 0.001, # Giảm thưởng sống sót thụ động để tránh việc chỉ né bom mà không làm việc

    # Nhận biết nguy hiểm
    "danger_evasion": 0.15,
    "danger_enter": -0.08,
    "own_blast_loiter": -0.05,

    # Định vị không gian
    "approach_enemy": 0.025,
}

# =====================================================================
# 2. CÁC HÀM BỔ TRỢ TÍNH TOÁN TOÁN HỌC & KHÔNG GIAN
# =====================================================================
def _parse_bomb_row(b):
    """Phân tích cú pháp dòng dữ liệu bom từ obs thành tuple dạng số."""
    arr = np.asarray(b, dtype=np.float64).ravel()
    if arr.size < 2:
        return None
    bx, by = int(arr[0]), int(arr[1])
    timer = int(arr[2]) if arr.size > 2 else _DEFAULT_BOMB_TIMER
    owner_id = int(arr[3]) if arr.size > 3 else _DEFAULT_BOMB_OWNER
    return bx, by, timer, owner_id


def _bomb_radius_from_obs(players, owner_id):
    """Lấy bán kính nổ thực tế của bom (bán kính gốc 1 + bonus nhặt đồ)."""
    return 1 + int(players[int(owner_id)][4])


def _explosion_tiles_for_bomb(grid, bx, by, radius):
    """Tính toán danh sách tọa độ (x, y) nằm trong phạm vi ảnh hưởng của vụ nổ."""
    h, w = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == Map.WALL: # Gặp tường indestructible thì dừng
                break
            tiles.add((tx, ty))
            if cell == Map.BOX:  # Gặp hòm thì phá hòm rồi dừng nổ tiếp
                break
    return tiles


def _blast_status_at(obs, x, y):
    """Kiểm tra xem tọa độ hiện tại có đang nằm trong tầm nổ của quả bom nào không."""
    bombs = obs["bombs"]
    if bombs is None:
        return False, None
    arr = np.asarray(bombs)
    if arr.size == 0 or arr.ndim == 1:
        return False, None
        
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
        if (int(x), int(y)) in tiles:
            in_blast = True
            t = int(timer)
            min_timer = t if min_timer is None else min(min_timer, t)
    return in_blast, min_timer


def _any_bombs(obs):
    b = obs["bombs"]
    return b is not None and np.asarray(b).size > 0


def _enemy_alive_count(players, agent_id):
    """Đếm số lượng đối thủ còn sống trên sân."""
    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return sum(1 for pid in range(arr.shape[0]) if pid != agent_id and int(arr[pid][2]) == 1)


def _manhattan_to_nearest_alive_enemy(players, agent_id, x, y):
    """Tính khoảng cách Manhattan tới đối thủ còn sống gần nhất."""
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
    """Tìm thời gian đếm ngược nhỏ nhất của các quả bom do CHÍNH MÌNH đặt."""
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
    """Đếm số ô trống có thể đi được xung quanh để tính đường thoát thân."""
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
    """Đếm tổng số hòm gỗ hiện có trên bản đồ."""
    return int(np.sum(grid == Map.BOX))


def _manhattan_to_nearest_item(grid, x, y):
    """THÊM MỚI: Tìm khoảng cách ngắn nhất từ Robot tới Vật phẩm gần nhất (mã 3 hoặc 4)."""
    ix, iy = int(x), int(y)
    # Lấy tọa độ của tất cả các ô chứa Item Radius (3) hoặc Item Capacity (4)
    item_positions = np.argwhere((grid == 3) | (grid == 4))
    if len(item_positions) == 0:
        return None
    
    # Tính khoảng cách Manhattan từ vị trí hiện tại tới từng Item
    distances = [abs(ix - ip[0]) + abs(iy - ip[1]) for ip in item_positions]
    return min(distances)


# =====================================================================
# 3. HÀM TÍNH PHẦN THƯỞNG CHÍNH (MAIN REWARD FUNCTION)
# =====================================================================
def compute_reward(prev_obs, curr_obs, agent_id):
    if prev_obs is None:
        return 0.0

    prev_players = prev_obs["players"]
    curr_players = curr_obs["players"]
    
    prev_alive = int(prev_players[agent_id][2])
    curr_alive = int(curr_players[agent_id][2])

    reward = 0.0
    
    # -----------------------------------------------------------------
    # KIỂM TRA ĐIỀU KIỆN THẮNG / THUA CHIẾN THUẬT
    # -----------------------------------------------------------------
    if prev_alive == 1 and curr_alive == 0:
        return float(REWARD_DICT["agent_death"]) # Chết là dính phạt ngay, thoát hàm
    
    prev_enemies_alive = _enemy_alive_count(prev_players, agent_id)
    curr_enemies_alive = _enemy_alive_count(curr_players, agent_id)
    
    if curr_enemies_alive < prev_enemies_alive:
        reward += REWARD_DICT["enemy_death"] * (prev_enemies_alive - curr_enemies_alive)
    if curr_enemies_alive == 0 and prev_enemies_alive > 0:
        reward += REWARD_DICT["win"]

    # -----------------------------------------------------------------
    # ĐIỀU HÀNH DI CHUYỂN & CHỐNG NÚP LÙM
    # -----------------------------------------------------------------
    prev_x, prev_y = prev_players[agent_id][0], prev_players[agent_id][1]
    curr_x, curr_y = curr_players[agent_id][0], curr_players[agent_id][1]
    
    if prev_x == curr_x and prev_y == curr_y:
        reward += REWARD_DICT["standing_still"] # Phạt nặng nếu đứng im thụ động
    else:
        reward -= REWARD_DICT["standing_still"] # Di chuyển sẽ được bù lại điểm đứng im
    
    reward += REWARD_DICT["time_penalty"] # Chi phí thời gian cố định mỗi bước

    # -----------------------------------------------------------------
    # CƠ CHẾ ĐIỀU HƯỚNG ĂN VẬT PHẨM (GIẢI QUYẾT TRIỆT ĐỂ BỆNH "LƯỜI")
    # -----------------------------------------------------------------
    # 1. Thưởng động: Tiến lại gần vật phẩm có trên bản đồ
    prev_item_d = _manhattan_to_nearest_item(prev_obs["map"], prev_x, prev_y)
    curr_item_d = _manhattan_to_nearest_item(curr_obs["map"], curr_x, curr_y)
    
    if prev_item_d is not None and curr_item_d is not None:
        # Nếu hiệu số dương tức là khoảng cách đang giảm dần -> Bot đang đi đúng hướng ăn đồ
        reward += REWARD_DICT["approach_item"] * (prev_item_d - curr_item_d)

    # 2. Thưởng tĩnh: Giây phút giẫm chân ăn được vật phẩm
    stepped_on_tile = prev_obs["map"][curr_x, curr_y]
    if stepped_on_tile in [3, 4]: 
        reward += REWARD_DICT["item_collection"]
    else:
        # Kiểm tra dự phòng nếu map cập nhật chậm nhưng chỉ số người chơi đã tăng
        prev_radius_bonus = int(prev_players[agent_id][4])
        curr_radius_bonus = int(curr_players[agent_id][4])
        if curr_radius_bonus > prev_radius_bonus:
             reward += REWARD_DICT["item_collection"]

    # -----------------------------------------------------------------
    # QUẢN LÝ RỦI RO & NÉ BOM SINH TỒN
    # -----------------------------------------------------------------
    if _any_bombs(prev_obs) or _any_bombs(curr_obs):
        prev_in_blast, prev_timer = _blast_status_at(prev_obs, prev_x, prev_y)
        curr_in_blast, _ = _blast_status_at(curr_obs, curr_x, curr_y)
        if prev_in_blast and not curr_in_blast:
            urgency = 1.5 if (prev_timer is not None and prev_timer <= 3) else 1.0
            reward += REWARD_DICT["danger_evasion"] * urgency
        elif not prev_in_blast and curr_in_blast and (prev_x != curr_x or prev_y != curr_y):
            reward += REWARD_DICT["danger_enter"]

    # Phạt lảng vảng cạnh bom của chính mình khi ngòi nổ ngắn lại
    mt_own = _min_own_blast_timer_at(curr_obs, agent_id, curr_x, curr_y)
    if curr_alive == 1 and mt_own is not None:
        urgency = max(1, 8 - int(mt_own))
        reward += REWARD_DICT["own_blast_loiter"] * float(urgency)

    # Thưởng hướng di chuyển tiếp cận kẻ địch để dồn ép
    if curr_alive == 1 and prev_enemies_alive > 0 and curr_enemies_alive > 0:
        prev_d = _manhattan_to_nearest_alive_enemy(prev_players, agent_id, prev_x, prev_y)
        curr_d = _manhattan_to_nearest_alive_enemy(curr_players, agent_id, curr_x, curr_y)
        if prev_d is not None and curr_d is not None:
            reward += REWARD_DICT["approach_enemy"] * (prev_d - curr_d)

    # -----------------------------------------------------------------
    # PHÁ HÒM GỖ ĐỂ TẠO RA VẬT PHẨM TỰ NHIÊN
    # -----------------------------------------------------------------
    prev_bombs_left = int(prev_players[agent_id][3])
    curr_bombs_left = int(curr_players[agent_id][3])
    
    if curr_bombs_left < prev_bombs_left: # Khoảnh khắc Agent vừa bấm nút Đặt Bom
        adjacent_tiles = [
            prev_obs["map"][max(0, curr_x-1), curr_y],
            prev_obs["map"][min(prev_obs["map"].shape[0]-1, curr_x+1), curr_y],
            prev_obs["map"][curr_x, max(0, curr_y-1)],
            prev_obs["map"][curr_x, min(prev_obs["map"].shape[1]-1, curr_y+1)]
        ]
        if 2 in adjacent_tiles: # Thưởng ngay nếu đặt cạnh hòm gỗ
            reward += REWARD_DICT["plant_near_box"]
        
        n_free = _count_free_neighbors(curr_obs["map"], curr_x, curr_y, curr_obs["bombs"])
        if n_free >= 1: # Chỉ thưởng đặt bom nếu tính toán thấy có đường chạy
            reward += REWARD_DICT["safe_bomb_plant"]

    # Thưởng lớn khi hòm gỗ thực tế bị nổ tung biến mất khỏi bản đồ
    prev_boxes = _count_boxes(prev_obs["map"])
    curr_boxes = _count_boxes(curr_obs["map"])
    if curr_boxes < prev_boxes:
        reward += REWARD_DICT["box_destroyed"] * (prev_boxes - curr_boxes)

    # Thưởng sống sót cơ bản
    if curr_alive == 1:
        reward += REWARD_DICT["survival_bonus"]

    return float(reward)