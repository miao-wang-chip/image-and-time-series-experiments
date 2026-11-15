"""Four sequence encoders with an identical direct multi-user forecasting head."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def selective_scan(u, dt, A, B, C, D):
    """Real-valued Mamba-1 recurrence; (batch,time,inner), state=(batch,inner,N)."""
    state = u.new_zeros(u.shape[0], u.shape[2], A.shape[1])
    outputs = []
    for t in range(u.shape[1]):
        delta = dt[:, t, :, None]
        state = (delta * A).exp() * state + delta * B[:, t, None, :] * u[:, t, :, None]
        outputs.append((state * C[:, t, None, :]).sum(-1) + D * u[:, t])
    return torch.stack(outputs, 1)


class MambaBlock(nn.Module):
    """CPU implementation of Mamba-1: selective dt/B/C, causal conv, gating, skip.

    Follows the equations in Gu & Dao (2023); no CUDA extension required.
    This is a small independently implemented real-state block, not Mamba-2.
    """
    def __init__(self, width, state_size):
        super().__init__()
        inner = width * 2
        self.rank, self.n = math.ceil(width / 16), state_size
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, inner * 2, bias=False)
        self.conv = nn.Conv1d(inner, inner, 4, padding=3, groups=inner)
        self.select = nn.Linear(inner, self.rank + 2 * state_size, bias=False)
        self.delta = nn.Linear(self.rank, inner)
        dt = torch.exp(torch.empty(inner).uniform_(math.log(.001), math.log(.1)))
        with torch.no_grad():
            self.delta.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        nn.init.uniform_(self.delta.weight, -self.rank**-.5, self.rank**-.5)
        self.A_log = nn.Parameter(torch.arange(1, state_size + 1).float().log().repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.out = nn.Linear(inner, width, bias=False)

    def forward(self, x):
        u, z = self.in_proj(self.norm(x)).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2))
        dt, B, C = self.select(u).split([self.rank, self.n, self.n], -1)
        y = selective_scan(u, F.softplus(self.delta(dt)), -self.A_log.exp(), B, C, self.D)
        return x + self.out(y * F.silu(z))


class Forecaster(nn.Module):
    def __init__(self, family, users, cfg):
        super().__init__()
        d, h = cfg['width'], cfg['history']
        self.family = family
        self.project = nn.Linear(users, d)
        if family == 'cnn':
            self.encoder = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(),
                                         nn.Conv1d(d, d, 5, padding=4, dilation=2), nn.GELU())
        elif family == 'gru':
            self.encoder = nn.GRU(d, d, batch_first=True)
        elif family == 'transformer':
            layer = nn.TransformerEncoderLayer(d, 4, d * 2, dropout=.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            position = torch.arange(h).float()[:, None]
            freq = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
            pe = torch.zeros(h, d)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(position * freq), torch.cos(position * freq)
            self.register_buffer('position', pe)
        elif family == 'mamba':
            self.encoder = MambaBlock(d, cfg['state_size'])
        else:
            raise ValueError(family)
        # Every route uses all encoded timestamps, the same head and daily residual.
        self.norm = nn.LayerNorm(d)
        self.time_head = nn.Linear(h, cfg['horizon'])
        self.user_head = nn.Linear(d, users)
        nn.init.normal_(self.user_head.weight, std=.005)
        nn.init.zeros_(self.user_head.bias)

    def forward(self, x):
        z = self.project(x)
        if self.family == 'cnn':
            z = self.encoder(z.transpose(1, 2)).transpose(1, 2)
        elif self.family == 'gru':
            z = self.encoder(z)[0]
        elif self.family == 'transformer':
            z = self.encoder(z + self.position)
        else:
            z = self.encoder(z)
        z = self.time_head(self.norm(z).transpose(1, 2)).transpose(1, 2)
        return x[:, -96:] + self.user_head(z)
