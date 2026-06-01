"""Dueling DQN with Noisy Networks."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer (Fortunato et al., 2018)."""
    def __init__(self, in_features, out_features, std_init=0.5):
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
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        ei = self._scale_noise(self.in_features)
        eo = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eo.outer(ei))
        self.bias_epsilon.copy_(eo)

    def forward(self, x):
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
        with torch.no_grad():
            conv_out = self.map_encoder(torch.zeros(1, c, h, w)).reshape(1, -1).size(1)
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
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
        return v + a - a.mean(dim=1, keepdim=True)

    def reset_noise(self):
        if self.noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
