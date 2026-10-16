"""World model v2 models and the baselines they are judged against.

All predict the next 5-minute frame of every cell from the past K frames, the platform's
actions in between and the next step's context:
    16 count channels   Poisson log-rate per cell (fleet by status / product, requests,
                        pickups, drop-offs, cancellations, declines, GPS samples)
    3 surge channels    surge - 1 per product (squared error)
    GPS speed           Gaussian mean and log-variance, scored only where cabs drove

Models
    persistence   next frame = this frame
    climatology   per cell, time of day and day type: mean of the training days
    local_cnn     3 conv layers over the last frame: local dynamics, no long range, no memory
    st_transformer  4x4 patches -> tokens; each block attends across the city within a
                  frame, then causally across time per patch (factorised space-time
                  attention, as in video world models: TimeSformer / Genie's ST-transformer)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import COUNT, OBS, SAMPLES, SPEED, SURGE

N_COUNT, N_SURGE = len(COUNT), len(SURGE)
N_OUT = N_COUNT + N_SURGE + 2
J_COUNT = [OBS.index(i) for i in COUNT]
J_SURGE = [OBS.index(i) for i in SURGE]


def split_out(o):
    """(..., N_OUT, G, G) -> log-rates, surge, speed mean (km/h), speed log-var."""
    return (o[..., :N_COUNT, :, :], o[..., N_COUNT:N_COUNT + N_SURGE, :, :],
            60 * o[..., -2, :, :], o[..., -1, :, :])


def losses(out, tgt, road_mask):
    """Per-term mean losses. tgt: raw target frames (B, K, C, G, G)."""
    lr, sg, mu, lv = split_out(out)
    lr = lr.clamp(-12, 8)
    y = tgt[:, :, COUNT]
    m = road_mask[:, None, None]                                          # (B,1,1,G,G)
    pois = ((lr.exp() - y * lr + torch.lgamma(y + 1)) * m).sum((0, 1, 3, 4)) / (m.sum() * lr.shape[1])
    ys = tgt[:, :, SURGE] / 100.0
    surge = (((sg - ys) ** 2) * m).sum() / (m.sum() * lr.shape[1] * N_SURGE)
    samp = tgt[:, :, SAMPLES] > 0
    sp = tgt[:, :, SPEED]
    lv = lv.clamp(0, 9)                                                   # log variance in (km/h)^2
    gauss = 0.5 * (lv + (sp - mu) ** 2 / lv.exp())
    speed = (gauss * samp).sum() / samp.sum().clamp(min=1)
    return {"poisson": pois, "surge": surge, "speed": speed}


def init_head_bias(bias: torch.Tensor, reps: int = 1):
    """Start at sensible outputs: low counts, no surge, 30 km/h with sd 8 km/h."""
    b = torch.zeros(N_OUT)
    b[:N_COUNT] = -3.0
    b[-2] = 0.5
    b[-1] = math.log(64.0)
    with torch.no_grad():
        bias.copy_(b.repeat_interleave(reps))


class Persistence(nn.Module):
    name = "persistence"

    def forward(self, inp, ctx, x_prev):
        B, K, C, G, _ = x_prev.shape
        out = torch.zeros(B, K, N_OUT, G, G, device=inp.device)
        out[:, :, :N_COUNT] = torch.log(x_prev[:, :, COUNT] + 0.1)
        out[:, :, N_COUNT:N_COUNT + N_SURGE] = x_prev[:, :, SURGE] / 100.0
        sp = x_prev[:, :, SPEED]
        out[:, :, -2] = torch.where(sp > 0, sp, torch.full_like(sp, 30.0)) / 60.0
        out[:, :, -1] = math.log(64.0)                                    # sd 8 km/h
        return out


class Climatology(nn.Module):
    """Per city, day type, time-of-day step and cell: training-day means."""
    name = "climatology"

    def __init__(self, tables: dict):
        super().__init__()
        self.tables = tables          # (city_id, weekend) -> (T, N_OUT, G, G) tensor

    def forward(self, inp, ctx, x_prev, keys):
        rows = []
        for (cid, _, t), c in zip(keys, ctx):
            wk = int(c[0, 4] > 0.5)
            rows.append(self.tables[(cid, wk)][t + 1: t + 1 + ctx.shape[1]])
        return torch.stack(rows)


class LocalCNN(nn.Module):
    name = "local_cnn"

    def __init__(self, c_in, width=64):
        super().__init__()
        self.ctx = nn.Linear(6, width)
        self.c1 = nn.Conv2d(c_in, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.c3 = nn.Conv2d(width, N_OUT, 1)
        init_head_bias(self.c3.bias)

    def forward(self, inp, ctx, x_prev=None):
        B, K, C, G, _ = inp.shape
        h = F.silu(self.c1(inp.flatten(0, 1)) + self.ctx(ctx.flatten(0, 1))[:, :, None, None])
        h = F.silu(self.c2(h))
        return self.c3(h).view(B, K, N_OUT, G, G)


class STBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv_s, self.o_s = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.qkv_t, self.o_t = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = heads

    def _attn(self, x, qkv, o, causal):
        B, N, D = x.shape
        q, k, v = qkv(x).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return o(y.transpose(1, 2).reshape(B, N, D))

    def forward(self, x):                                   # x: (B, K, P, D)
        B, K, P, D = x.shape
        s = x.reshape(B * K, P, D)
        s = s + self._attn(self.n1(s), self.qkv_s, self.o_s, False)
        t = s.view(B, K, P, D).transpose(1, 2).reshape(B * P, K, D)
        t = t + self._attn(self.n2(t), self.qkv_t, self.o_t, True)
        x = t.view(B, P, K, D).transpose(1, 2)
        return x + self.mlp(self.n3(x))


class STTransformer(nn.Module):
    name = "st_transformer"

    def __init__(self, c_in, G=64, patch=4, d=192, layers=6, heads=6, K=6, skip=False):
        """`skip` (step 4d): a per-cell path from each step's own inputs (3 x 3 neighbourhood)
        added to the output, so cell-exact effects such as "the drivers ordered away from this
        cell are no longer idle here" need not pass through the 4 x 4 patch tokens. Its last
        layer starts at zero, so a model fine-tuned with it starts as the model without it."""
        super().__init__()
        self.p, self.G = patch, G
        self.skip = None
        if skip:
            self.skip = nn.Sequential(nn.Conv2d(c_in, 96, 3, padding=1), nn.SiLU(), nn.Conv2d(96, 96, 1), nn.SiLU(),
                                      nn.Conv2d(96, N_OUT, 1))
            nn.init.zeros_(self.skip[-1].weight)
            nn.init.zeros_(self.skip[-1].bias)
        n = G // patch
        self.embed = nn.Conv2d(c_in, d, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, 1, n * n, d))
        self.tpos = nn.Parameter(torch.zeros(1, K, 1, d))
        self.ctx = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.tpos, std=0.02)
        self.blocks = nn.ModuleList(STBlock(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_OUT * patch * patch)
        init_head_bias(self.head.bias, patch * patch)               # head output is (N_OUT, p, p)

    def encode(self, inp, ctx):
        """Token states (B, K, P, d): the learned world state, used later by probes."""
        B, K, C, G, _ = inp.shape
        z = self.embed(inp.flatten(0, 1)).flatten(2).transpose(1, 2).view(B, K, -1, self.pos.shape[-1])
        z = z + self.pos + self.tpos[:, :K] + self.ctx(ctx)[:, :, None]
        for b in self.blocks:
            z = b(z)
        return self.norm(z)

    def forward(self, inp, ctx, x_prev=None):
        B, K = inp.shape[:2]
        z = self.encode(inp, ctx)
        n, p = self.G // self.p, self.p
        o = self.head(z).view(B, K, n, n, N_OUT, p, p)
        o = o.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, K, N_OUT, self.G, self.G)
        if self.skip is not None:
            o = o + self.skip(inp.flatten(0, 1)).view(B, K, N_OUT, self.G, self.G)
        return o
