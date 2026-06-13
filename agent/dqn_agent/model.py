"""Dueling DQN with Noisy Networks for Bomberland.

Features:
  - Factorised Gaussian NoisyNet layer (Fortunato et al., 2018) for adaptive exploration.
  - Two-branch Dueling DQN (Wang et al., 2016) to separate State Value and Advantage.
  - Dynamic CNN flattening to support varying map sizes (11x11, 15x15, etc.) safely.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer.
    
    Sinh nhiễu ngẫu nhiên vào trọng số (weights) trong quá trình huấn luyện
    để tự động hóa việc Exploration thay vì dùng Epsilon-greedy truyền thống.
    """
    def __init__(self, in_features, out_features, std_init=0.7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        mu_range = 1 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(std_init / math.sqrt(in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(std_init / math.sqrt(out_features))
        
        self.reset_noise()

    @staticmethod
    def _scale_noise(size):
        """Hàm tạo nhiễu Gaussian được factorize để giảm chi phí tính toán."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Lấy mẫu (sample) lại giá trị Epsilon mới cho mạng."""
        ei = self._scale_noise(self.in_features)
        eo = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eo.outer(ei))
        self.bias_epsilon.copy_(eo)

    def forward(self, x):
        """Chỉ cộng nhiễu khi mạng đang ở chế độ .train()"""
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class DuelingDQN(nn.Module):
    """
    Two-branch Dueling DQN:
      - Conv2D encoder for spatial channels
      - MLP encoder for auxiliary scalars
      - Value stream V(s) + Advantage stream A(s,a)
    """
    def __init__(self, map_shape, aux_dim, n_actions, noisy=True):
        super().__init__()
        c, h, w = map_shape
        self.noisy = noisy
        
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        
        # Tự động tính toán kích thước tensor sau khi qua CNN (safe cho mọi size map)
        with torch.no_grad():
            conv_out = self.map_encoder(torch.zeros(1, c, h, w)).reshape(1, -1).size(1)
            
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 64), nn.ReLU(), 
            nn.Linear(64, 64), nn.ReLU(),
        )
        
        feat = conv_out + 64
        L = NoisyLinear if noisy else nn.Linear
        
        self.value = nn.Sequential(L(feat, 256), nn.ReLU(), L(256, 1))
        self.advantage = nn.Sequential(L(feat, 256), nn.ReLU(), L(256, n_actions))

    def forward(self, map_x, aux_x):
        mf = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        af = self.aux_encoder(aux_x)
        f = torch.cat([mf, af], dim=1)
        
        v = self.value(f)
        a = self.advantage(f)
        
        # Dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return v + a - a.mean(dim=1, keepdim=True)

    def reset_noise(self):
        """Gọi hàm này sau mỗi bước tối ưu (optimizer.step()) để làm mới nhiễu."""
        if self.noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()