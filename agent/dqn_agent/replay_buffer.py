"""Prioritized Experience Replay (PER) with SumTree and N-step returns.

Features:
  - O(log n) priority updates and sampling using an array-based binary SumTree.
  - Built-in N-step return calculation using a deque.
  - Safe episode transitions (reset_episode) to prevent cross-episode contamination.
"""
import numpy as np
import random
from collections import deque


class SumTree:
    """Binary tree where parent = sum of children, for O(log n) priority sampling."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        # Cây nhị phân dạng mảng có 2*capacity - 1 node
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Đẩy sự thay đổi giá trị từ node lá lên tận node gốc."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Tìm node lá tương ứng với giá trị tích lũy s."""
        left = 2 * idx + 1
        right = left + 1

        # Nếu đã chạm đáy (node lá)
        if left >= len(self.tree):
            return idx

        # Xử lý sai số dấu phẩy động (floating-point accuracy)
        if s <= self.tree[left] or np.isclose(s, self.tree[left]):
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Trả về tổng priority của toàn bộ cây (nằm ở node gốc)."""
        return self.tree[0]

    def add(self, priority):
        """Thêm một kinh nghiệm mới vào cây (ghi đè dạng vòng tròn)."""
        idx = self.write + self.capacity - 1
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """Cập nhật priority mới cho một node và lan truyền lên gốc."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Lấy ra (tree_index, priority, data_index) dựa trên giá trị ngẫu nhiên s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx


class PrioritizedReplayBuffer:
    """PER buffer with n-step return computation built in."""
    
    # PER hyperparameters
    PER_E = 0.01          # Hằng số nhỏ để tránh priority = 0
    PER_A = 0.6           # Mức độ ưu tiên (0 = uniform, 1 = strict priority)
    PER_B_START = 0.4     # Mức độ bù trừ bias ban đầu (sẽ tăng dần lên 1.0)

    def __init__(self, capacity, map_shape, aux_dim, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.beta = self.PER_B_START
        self.pos = 0

        # Phân bổ bộ nhớ cho các tensor trạng thái
        self.map_s = np.zeros((capacity, *map_shape), dtype=np.float32)
        self.aux_s = np.zeros((capacity, aux_dim), dtype=np.float32)
        self.nmap_s = np.zeros((capacity, *map_shape), dtype=np.float32)
        self.naux_s = np.zeros((capacity, aux_dim), dtype=np.float32)
        
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Buffer tạm thời để tính N-step return
        self._nstep_buf = deque(maxlen=n_step)

    def reset_episode(self):
        """Clear n-step buffer to prevent merging states across different episodes."""
        self._nstep_buf.clear()

    def __len__(self):
        return self.tree.n_entries

    def _priority(self, td):
        """Tính priority mới dựa trên TD error."""
        return (np.abs(td) + self.PER_E) ** self.PER_A

    def push(self, ms, axs, a, r, nms, naxs, done):
        """Lưu trữ một transition vào buffer n-step. 
        Khi đủ n-step hoặc game over, transition sẽ được đẩy vào bộ nhớ chính.
        """
        self._nstep_buf.append((ms, axs, a, r, nms, naxs, done))
        
        if len(self._nstep_buf) < self.n_step and not done:
            return
            
        if done:
            # Xả toàn bộ n-step buffer khi game over
            while self._nstep_buf:
                self._store_nstep()
                self._nstep_buf.popleft()
        else:
            self._store_nstep()

    def _store_nstep(self):
        """Tính toán tổng phần thưởng có chiết khấu cho N-step và lưu vào mảng NumPy."""
        R = 0.0
        for i, (_, _, _, r, _, _, d) in enumerate(self._nstep_buf):
            R += (self.gamma ** i) * r
            if d:
                break
                
        first = self._nstep_buf[0]
        last = self._nstep_buf[-1]
        
        idx = self.pos
        
        # Lưu state đầu tiên và state cuối cùng của chuỗi N-step
        self.map_s[idx] = first[0]
        self.aux_s[idx] = first[1]
        self.actions[idx] = first[2]
        self.rewards[idx] = R
        self.nmap_s[idx] = last[4]
        self.naux_s[idx] = last[5]
        self.dones[idx] = last[6]
        
        # Gán priority lớn nhất để đảm bảo transition mới được sample ít nhất 1 lần
        max_p = max(np.max(self.tree.tree[-self.capacity:]), 1.0)
        self.tree.add(max_p)
        
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Lấy mẫu ngẫu nhiên có trọng số (O(log N))."""
        idxs, tree_idxs, prios = [], [], []
        seg = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = seg * i
            b = seg * (i + 1)
            s = random.uniform(a, b)
            
            ti, p, di = self.tree.get(s)
            
            idxs.append(di % self.capacity)
            tree_idxs.append(ti)
            prios.append(p)
            
        ix = np.array(idxs)
        p = np.array(prios, dtype=np.float64)
        
        # Tính toán Importance Sampling (IS) weights để sửa bias
        prob = p / (self.tree.total() + 1e-8)
        w = (len(self) * prob + 1e-8) ** (-self.beta)
        w /= w.max()  # Chuẩn hóa về [0, 1] để ổn định gradient
        
        return (self.map_s[ix], self.aux_s[ix], self.nmap_s[ix], self.naux_s[ix],
                self.actions[ix], self.rewards[ix], self.dones[ix],
                np.array(tree_idxs), w.astype(np.float32))

    def update_priorities(self, tree_idxs, td_errors):
        """Cập nhật lại priority của các transition sau khi tính được hàm Loss."""
        for i, td in zip(tree_idxs, td_errors):
            self.tree.update(i, self._priority(td))

    def anneal_beta(self, frac):
        """Tăng dần Beta từ 0.4 lên 1.0 trong suốt quá trình train."""
        self.beta = min(1.0, self.PER_B_START + (1.0 - self.PER_B_START) * frac)