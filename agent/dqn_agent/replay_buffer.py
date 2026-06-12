"""Prioritized Experience Replay with SumTree and N-step returns."""
import numpy as np
import random
from collections import deque


class SumTree:
    """Binary tree where parent = sum of children, for O(log n) priority sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority):
        idx = self.write + self.capacity - 1
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        return idx, self.tree[idx], idx - self.capacity + 1


class PrioritizedReplayBuffer:
    """PER buffer with n-step return computation built in."""
    PER_E = 0.01
    PER_A = 0.6
    PER_B_START = 0.4

    def __init__(self, capacity, map_shape, aux_dim, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.beta = self.PER_B_START
        self.pos = 0

        self.map_s = np.zeros((capacity, *map_shape), dtype=np.float32)
        self.aux_s = np.zeros((capacity, aux_dim), dtype=np.float32)
        self.nmap_s = np.zeros((capacity, *map_shape), dtype=np.float32)
        self.naux_s = np.zeros((capacity, aux_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._nstep_buf = deque(maxlen=n_step)

    def reset_episode(self):
        """Flush n-step buffer between episodes to prevent cross-episode contamination."""
        self._nstep_buf.clear()

    def __len__(self):
        return self.tree.n_entries

    def _priority(self, td):
        return (np.abs(td) + self.PER_E) ** self.PER_A

    def push(self, ms, axs, a, r, nms, naxs, done):
        self._nstep_buf.append((ms, axs, a, r, nms, naxs, done))
        if len(self._nstep_buf) < self.n_step and not done:
            return
        if done:
            while self._nstep_buf:
                self._store_nstep()
                self._nstep_buf.popleft()
        else:
            self._store_nstep()

    def _store_nstep(self):
        R = 0.0
        for i, (_, _, _, r, _, _, d) in enumerate(self._nstep_buf):
            R += (self.gamma ** i) * r
            if d:
                break
        first = self._nstep_buf[0]
        last = self._nstep_buf[-1]
        idx = self.pos
        self.map_s[idx] = first[0]
        self.aux_s[idx] = first[1]
        self.actions[idx] = first[2]
        self.rewards[idx] = R
        self.nmap_s[idx] = last[4]
        self.naux_s[idx] = last[5]
        self.dones[idx] = last[6]
        max_p = max(np.max(self.tree.tree[-self.capacity:]), 1.0)
        self.tree.add(max_p)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idxs, tree_idxs, prios = [], [], []
        seg = self.tree.total() / batch_size
        for i in range(batch_size):
            s = random.uniform(seg * i, seg * (i + 1))
            ti, p, di = self.tree.get(s)
            idxs.append(di % self.capacity)
            tree_idxs.append(ti)
            prios.append(p)
        ix = np.array(idxs)
        p = np.array(prios, dtype=np.float64)
        prob = p / (self.tree.total() + 1e-8)
        w = (len(self) * prob + 1e-8) ** (-self.beta)
        w /= w.max()
        return (self.map_s[ix], self.aux_s[ix], self.nmap_s[ix], self.naux_s[ix],
                self.actions[ix], self.rewards[ix], self.dones[ix],
                np.array(tree_idxs), w.astype(np.float32))

    def update_priorities(self, tree_idxs, td_errors):
        for i, td in zip(tree_idxs, td_errors):
            self.tree.update(i, self._priority(td))

    def anneal_beta(self, frac):
        self.beta = min(1.0, self.PER_B_START + (1.0 - self.PER_B_START) * frac)
