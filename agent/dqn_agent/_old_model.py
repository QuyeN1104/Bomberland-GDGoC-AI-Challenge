"""Legacy DQNModel kept for loading old checkpoints."""
import torch
import torch.nn as nn

class DQNModel(nn.Module):
    def __init__(self, map_shape, aux_dim, output_dim):
        super().__init__()
        c, h, w = map_shape
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            conv_out_dim = self.map_encoder(torch.zeros(1, c, h, w)).reshape(1, -1).size(1)
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + 32, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, map_x, aux_x):
        mf = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        af = self.aux_encoder(aux_x)
        return self.head(torch.cat([mf, af], dim=1))
