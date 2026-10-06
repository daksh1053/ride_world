"""Learned world models of the ride-service world, and the baselines they are judged against.

All models predict the next step's zone counts p(obs_{k+1} | history, A_k, context)
with a Poisson likelihood per count (formulation eq. 29: minimise the NLL of observed
targets). Mapped onto the survey's anatomy (sec. 4):

    substrate   structured feature substrate: 6 counts per zone per 5 min
    coupling    action-conditioned rollout, step-wise (eq. 10, second line)
    backbone    per-zone networks with shared weights (so one model serves any city)
    regime      interactive simulator: open-loop rollouts under supplied actions

Models
    persistence   next counts = current counts
    seasonal      mean of the same zone / time of day / day type over training days
    glm           Poisson regression (linear log-rate)
    mlp           2 hidden layers, Markov in the last 3 steps
    gru           recurrent latent state per zone (PlaNet/Dreamer's deterministic path)
    gru_os        gru trained with overshooting: inside each window it is fed its own
                  predictions, as in rollouts (latent overshooting, PlaNet; the
                  train-on-own-rollouts idea behind Diffusion Forcing's stable rollouts)
    gru_cons      gru whose idle / busy heads conserve the fleet: it predicts each city
                  total as a change of the current total, then splits it across zones
                  with a softmax. A cab cannot appear or vanish by rounding error
                  (the survey's "physical plausibility" built into the substrate, 5.4)

Any model can drop its action inputs (`use_actions=False`) to measure interactability.
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import cities, wm_data

N_OBS, N_ACT, N_CTX, N_STATIC = len(wm_data.OBS), len(wm_data.ACT), 5, 4
N_FEAT = 3 * N_OBS + N_ACT + N_OBS + 1 + N_OBS + 1 + N_CTX + N_STATIC
EVENT_IDX = [0, 1, 2]          # counts over the step (new, pickups, cancels)
STATE_IDX = [3, 4, 5]          # snapshots at the boundary (idle, busy, pending)


# ---------------------------------------------------------------- data

@dataclass
class Run:
    city: str
    split: str
    date: str
    policy: str
    obs: torch.Tensor      # (K+1, Z, 6) counts
    act: torch.Tensor      # (K, Z, 3)
    ctx: torch.Tensor      # (K+1, 5)
    static: torch.Tensor   # (Z, 4)
    nbr: torch.Tensor      # (Z, k)
    log_fleet: float


def load_runs(keys, split, device, policies=None) -> list[Run]:
    out = []
    for k in keys:
        city = cities.get(k)
        d = city.data_dir / "wm"
        st = np.load(d / "static.npz")
        static = torch.tensor(st["static"], device=device)
        nbr = torch.tensor(st["nbr"], device=device)
        for f in sorted(glob.glob(str(d / f"{split}_*.npz"))):
            m = re.match(rf"{split}_(\d{{4}}-\d\d-\d\d)_(.+)\.npz", Path(f).name)
            if policies and m.group(2) not in policies:
                continue
            z = np.load(f)
            out.append(Run(k, split, m.group(1), m.group(2),
                           torch.tensor(z["obs"], device=device), torch.tensor(z["act"], device=device),
                           torch.tensor(z["ctx"], device=device), static, nbr,
                           float(np.log(city.market.n_drivers))))
    return out


def features(hist: torch.Tensor, act: torch.Tensor, ctx: torch.Tensor, run: Run) -> torch.Tensor:
    """hist (3, Z, 6) counts at k, k-1, k-2 (newest first); act (Z, 3); ctx (5,) -> (Z, F).

    Everything is computed from counts, so a rollout can feed its own predictions in.
    """
    Z = hist.shape[1]
    lh = torch.log1p(hist.clamp(min=0))
    own = lh.permute(1, 0, 2).reshape(Z, -1)                       # 18
    la = torch.log1p(act)                                          # 3
    nb = torch.log1p(hist[0][run.nbr].mean(1))                     # 6: neighbourhood now
    nb_in = torch.log1p(act[run.nbr, 1].sum(1, keepdim=True))      # 1: dispatches into neighbours
    city = torch.log1p(hist[0].sum(0)).expand(Z, -1)               # 6: city totals now
    fleet = torch.full((Z, 1), run.log_fleet, device=hist.device)  # 1
    return torch.cat([own, la, nb, nb_in, city, fleet, ctx.expand(Z, -1), run.static], 1)


# ---------------------------------------------------------------- models

CITY_IDLE, CITY_BUSY = 28 + 3, 28 + 4      # feature columns: log1p(city total idle / busy)
LOG_RATE_MAX = 9.0                           # e^9 ~ 8,000 per zone-step: numerical guard only


class Net(nn.Module):
    def __init__(self, kind: str, use_actions: bool = True, hidden: int = 128):
        super().__init__()
        self.kind, self.use_actions = kind, use_actions
        self.conserve = kind.startswith("gru_cons")
        if kind == "glm":
            self.body = nn.Identity()
            self.head = nn.Linear(N_FEAT, N_OBS)
        elif kind == "mlp":
            self.body = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
            self.head = nn.Linear(hidden, N_OBS)
        else:                                                   # gru, gru_os
            self.enc = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.SiLU())
            self.cell = nn.GRUCell(hidden, hidden // 2)
            self.head = nn.Sequential(nn.Linear(hidden // 2 + hidden, hidden), nn.SiLU(), nn.Linear(hidden, N_OBS))
            if self.conserve:                        # city-level change of total idle / busy
                self.total = nn.Sequential(nn.Linear(hidden // 2 + hidden, 64), nn.SiLU(), nn.Linear(64, 2))
        self.register_buffer("mu", torch.zeros(N_FEAT))
        self.register_buffer("sd", torch.ones(N_FEAT))

    @property
    def recurrent(self):
        return self.kind.startswith("gru")

    def mask(self, x):
        if not self.use_actions:
            x = x.clone()
            x[:, 18:21] = 0.0          # this step's actions
            x[:, 27] = 0.0             # dispatches into neighbours
        return (x - self.mu) / self.sd

    def init_state(self, Z, device):
        return torch.zeros(Z, self.cell.hidden_size, device=device) if self.recurrent else None

    def forward(self, x, h=None):
        """x (Z, F) raw features -> log-rate (Z, 6), next state. One call = one city-step."""
        raw = x
        x = self.mask(x)
        if not self.recurrent:
            return self.head(self.body(x)).clamp(max=LOG_RATE_MAX), None
        e = self.enc(x)
        h = self.cell(e, h)
        z = torch.cat([h, e], 1)
        out = self.head(z)
        if self.conserve:
            # totals now (from the city-total features) times a learned change, split by softmax
            now = torch.expm1(raw[0, [CITY_IDLE, CITY_BUSY]]).clamp(min=0.5)
            delta = self.total(z.mean(0)).clamp(-1.0, 1.0)
            tot = now * delta.exp()                                    # (2,)
            share = torch.log_softmax(out[:, 3:5], 0)                  # over zones
            out = torch.cat([out[:, :3], share + tot.log(), out[:, 5:]], 1)
        return out.clamp(max=LOG_RATE_MAX), h


def poisson_nll(log_rate, y):
    """Mean Poisson NLL per count, including the log(y!) constant (true NLL in nats)."""
    return (log_rate.exp() - y * log_rate + torch.lgamma(y + 1)).mean()


def fit_normaliser(model, runs, n=4000):
    xs = []
    g = torch.Generator().manual_seed(0)
    for _ in range(min(n, 400)):
        r = runs[torch.randint(len(runs), (1,), generator=g).item()]
        k = torch.randint(2, r.act.shape[0], (1,), generator=g).item()
        xs.append(features(r.obs[[k, k - 1, k - 2]], r.act[k], r.ctx[k], r))
    x = torch.cat(xs)
    model.mu.copy_(x.mean(0))
    model.sd.copy_(x.std(0).clamp(min=1e-3))


# ---------------------------------------------------------------- training

def precompute(runs) -> list[torch.Tensor]:
    """Teacher-forced features for every run: (K, Z, F), rows k < 2 left at zero."""
    out = []
    for r in runs:
        K, Z = r.act.shape[0], r.obs.shape[1]
        X = torch.zeros(K, Z, N_FEAT, device=r.obs.device)
        for k in range(2, K):
            X[k] = features(r.obs[[k, k - 1, k - 2]], r.act[k], r.ctx[k], r)
        out.append(X)
    return out


def train(model, runs, X, epochs=6, batch=16384, window=24, lr=2e-3, seed=0, log=print):
    """Teacher-forced maximum likelihood (eq. 29). Non-recurrent models: random
    zone-steps; recurrent models: windows of `window` steps, many runs side by side."""
    torch.manual_seed(seed)
    fit_normaliser(model, runs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    K = runs[0].act.shape[0]
    if not model.recurrent:
        Xf = torch.cat([x[2:].reshape(-1, N_FEAT) for x in X])
        Yf = torch.cat([r.obs[3:].reshape(-1, N_OBS) for r in runs])
        nb = len(Xf) // batch
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * nb)
        for ep in range(epochs):
            perm = torch.randperm(len(Xf), device=Xf.device)
            tot = 0.0
            for b in range(nb):
                sel = perm[b * batch:(b + 1) * batch]
                lr_k, _ = model(Xf[sel])
                loss = poisson_nll(lr_k, Yf[sel])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item()
            log(f"      epoch {ep + 1}/{epochs}  train NLL {tot / nb:.4f}")
        del Xf, Yf
        return model
    starts = [(i, s) for i, r in enumerate(runs) for s in range(2, K - window, window)]
    per = 1 if model.conserve else 8                           # runs side by side (zone softmax needs one city)
    steps = epochs * (len(starts) // per)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for ep in range(epochs):
        perm = torch.randperm(len(starts))
        tot, n = 0.0, 0
        for b in range(len(starts) // per):
            group = [starts[j] for j in perm[b * per:(b + 1) * per].tolist()]
            xw = torch.cat([X[i][s:s + window] for i, s in group], 1)          # (W, sumZ, F)
            yw = torch.cat([runs[i].obs[s + 1:s + 1 + window] for i, s in group], 1)
            h = model.init_state(xw.shape[1], xw.device)
            loss = 0.0
            for t in range(window):
                lr_k, h = model(xw[t], h)
                if t >= 2:                                     # first steps only warm the state
                    loss = loss + poisson_nll(lr_k, yw[t])
            loss = loss / (window - 2)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        log(f"      epoch {ep + 1}/{epochs}  train NLL {tot / n:.4f}")
    return model


def finetune_overshoot(model, runs, X, n_windows=600, window=24, free=12, p=0.7, lr=5e-4, seed=1, log=print):
    """Overshooting fine-tune: after `free` real steps, the model is fed its own predicted
    counts with probability p per step, so it learns to recover from its own errors
    (latent overshooting, PlaNet; the anti-drift motivation of Diffusion Forcing)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    K = runs[0].act.shape[0]
    tot = 0.0
    for w in range(n_windows):
        i = torch.randint(len(runs), (1,)).item()
        r = runs[i]
        s = torch.randint(2 + free, K - window, (1,)).item()
        h = model.init_state(r.obs.shape[1], r.obs.device)
        for k in range(s - free, s):                           # ground the state on real data
            _, h = model(X[i][k], h)
        hist = r.obs[[s, s - 1, s - 2]].clone()
        loss = 0.0
        for k in range(s, s + window):
            lr_k, h = model(features(hist, r.act[k], r.ctx[k], r), h)
            y = r.obs[k + 1]
            loss = loss + poisson_nll(lr_k, y)
            nxt = lr_k.exp() if torch.rand(1).item() < p else y
            hist = torch.cat([nxt[None], hist[:2]])
        loss = loss / window
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()
        if (w + 1) % 200 == 0:
            log(f"      overshoot {w + 1}/{n_windows}  NLL {tot / 200:.4f}")
            tot = 0.0
    return model


# ---------------------------------------------------------------- prediction

@torch.no_grad()
def rollout(model, r: Run, k0: int, H: int, teacher: bool = False, warm: int = 12):
    """Predicted mean counts for steps k0+1 .. k0+H, (H, Z, 6).

    Controlled rollout: actions and exogenous context come from the log; the model's
    own predicted counts are fed back (teacher=False). A recurrent model first warms
    its state on the `warm` real steps before k0 (re-grounding on observations,
    survey sec. 5.3), then imagines.
    """
    Z = r.obs.shape[1]
    h = model.init_state(Z, r.obs.device) if model is not None else None
    if model is not None and model.recurrent:
        for k in range(max(2, k0 - warm), k0):
            _, h = model(features(r.obs[[k, k - 1, k - 2]], r.act[k], r.ctx[k], r), h)
    hist = r.obs[[k0, k0 - 1, k0 - 2]].clone()
    out = []
    for k in range(k0, k0 + H):
        lr_k, h = model(features(hist, r.act[k], r.ctx[k], r), h)
        mean = lr_k.exp()
        out.append(mean)
        nxt = r.obs[k + 1] if teacher else mean
        hist = torch.cat([nxt[None], hist[:2]])
    return torch.stack(out)


def baseline_persistence(r: Run, k0: int, H: int):
    return r.obs[k0][None].expand(H, -1, -1).clone()


class Seasonal:
    """Mean of each zone's counts at the same step of the day over training days of the
    same city and day type (weekday / weekend)."""

    def __init__(self, train_runs):
        self.table = {}
        for r in train_runs:
            key = (r.city, bool(r.ctx[0, 2].item()))
            self.table.setdefault(key, []).append(r.obs)
        self.table = {k: torch.stack(v).mean(0) for k, v in self.table.items()}

    def predict(self, r: Run, k0: int, H: int):
        key = (r.city, bool(r.ctx[0, 2].item()))
        if key not in self.table:                      # city or day type never seen
            return None
        return self.table[key][k0 + 1:k0 + 1 + H]


def save(model, path, meta: dict):
    torch.save({"state": model.state_dict(), "kind": model.kind, "use_actions": model.use_actions,
                "meta": meta}, path)
    Path(str(path) + ".json").write_text(json.dumps(meta, indent=2))
