"""Stage 7: drivers as small agents that maximise their own earnings, choosing from menus the
platform's world-model agent gives them.

Every 30-s epoch the platform (world/wm3/agent.py Agent, its matching head) scores every allowed
(request, idle driver) pair; each idle driver is shown a MENU: up to K requests it may take (the
platform's K best for that driver), each with the fare the driver will get (the rider's quote,
surge included, minus the platform's commission), the pickup time, the trip time and where the
trip ends. The driver takes one option or declines all. Drivers choose at the same time; when two
take the same request the platform's higher-scored driver gets it (the other stays idle and sees
a new menu next epoch). A taken option is a ride the driver has accepted.

The driver's policy is one small network shared by every driver (each sees only its own
situation and its own menu):
    option j   MLP([option features j, own features])
    decline    MLP(own features) + b_decline
Its aim is the driver's own net earnings (fares minus the cost of the km driven) over the time it
works. Option features (all in units a driver would use):
    platform rank, pickup min, trip min, fare / median fare, fare per engaged hour, km cost,
    net per engaged hour, intercity, surge, at the drop-off zone: requests in the last 30 min per
    idle driver and minutes to the driver's home zone
Own features: hours online, hours since the usual shift start, today's earnings, minutes idle, own
zone's requests per idle driver, cost per km (relative), mode.

Everything is computed from the entity-level state (an imagined world, or the simulator's logs
rebuilt into one, state.DayLog), so the same driver policy runs in both.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import rollout as RO

K_MENU = 4
OPT_F = 12
SELF_F = 9
RO.KIND.setdefault("driver", max(RO.KIND.values()) + 1)


class DriverPolicy(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.opt = nn.Sequential(nn.Linear(OPT_F + SELF_F, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.dec = nn.Sequential(nn.Linear(SELF_F, h), nn.GELU(), nn.Linear(h, 1))
        self.b_decline = nn.Parameter(torch.tensor(-3.0))
        self.w_rank = nn.Parameter(torch.tensor(-2.0))       # starts as "take the platform's first option"
        for m in (self.opt[-1], self.dec[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, opt, own, mask):
        """opt (B, K, OPT_F), own (B, SELF_F), mask (B, K) -> logits (B, 1 + K) [decline, options]."""
        x = torch.cat([opt, own[:, None].expand(-1, opt.shape[1], -1)], -1)
        lo = self.opt(x)[..., 0] + self.w_rank * opt[..., 0]
        lo = lo.masked_fill(~mask, float("-inf"))
        ld = self.dec(own)[..., 0] + self.b_decline
        return torch.cat([ld[:, None], lo], -1)


def menus(mi, lg, K=K_MENU):
    """mi: matching inputs (agent.match_inputs), lg: the platform's (n_req, 1 + m) logits.
    Returns {candidate driver column c: array of request rows (best first)} for drivers with any
    allowed request, and the score matrix (n_req, m) used to settle conflicts."""
    sc = np.where(mi["feas"], lg[:, 1:], -np.inf)
    out = {}
    for c in range(sc.shape[1]):
        ok = np.flatnonzero(np.isfinite(sc[:, c]))
        if len(ok):
            out[c] = ok[np.argsort(-sc[ok, c], kind="stable")][:K]
    return out, sc


def features(W, mi, menu):
    """Option and own features for every driver with a menu, from imagined world W (the
    simulator's state rebuilt, when live). Returns cols (n,), opt (n, K, OPT_F), own (n, SELF_F),
    mask (n, K), fare (n, K) in money."""
    R, g = W.R, W.g
    t = W.t
    h = int(t // 3600) % 24
    tt = R.zone_tt[W.daytype, h]
    dist = R.zone_dist[W.daytype, h]
    E0 = W.money
    com = R.market.commission
    Z = R.Z
    # zone prospects: requests in the last 30 min per idle driver (all modes)
    Hh = np.array(W.hist) if W.hist else np.zeros((0, 4))
    rec = np.bincount(Hh[(Hh[:, 0] >= t - 1800) & (Hh[:, 3] > 0), 1].astype(int), minlength=Z) if len(Hh) else np.zeros(Z)
    idle_z = np.bincount(g.zone[W.node[W.st == RO.IDLE]], minlength=Z)
    prospect = rec / (idle_z + 1.0)
    cols = np.array(sorted(menu), np.int64)
    n = len(cols)
    opt = np.zeros((n, K_MENU, OPT_F), np.float32)
    mask = np.zeros((n, K_MENU), bool)
    fare = np.zeros((n, K_MENU))
    own = np.zeros((n, SELF_F), np.float32)
    kap_med = float(np.median(W.kappa))
    for i, c in enumerate(cols):
        d = int(mi["drv"][c])
        dz = int(g.zone[W.node[d]])
        hz = int(W.home_zone[d]) if hasattr(W, "home_zone") else dz
        for j, r in enumerate(menu[c]):
            q = W.req[int(mi["req"][r])]
            oz = int(g.zone[q["origin"]])
            if q["ext"] >= 0:
                ez = int(g.zone[R.gw_pos[q["ext"]]])
                trip = tt[oz, ez] / 60 + R.ext[q["ext"]].outside_min
                km = dist[oz, ez] / 1000 + R.ext[q["ext"]].outside_km
                end = ez
            else:
                end = int(g.zone[q["dest"]])
                trip = tt[oz, end] / 60
                km = dist[oz, end] / 1000
            pick = float(mi["eta"][r, c])
            km_pick = dist[dz, oz] / 1000
            f = q["quote"] * (1 - com)
            cost = W.kappa[d] * (km + km_pick)
            eng = max(pick + trip, 1.0) / 60
            opt[i, j] = [j, pick / 10, trip / 30, f / E0, f / E0 / eng / 4, cost / E0, (f - cost) / E0 / eng / 4,
                         float(q["ext"] >= 0), q["surge"] - 1, np.log1p(prospect[end]), tt[end, hz] / 3600, 1.0]
            mask[i, j] = True
            fare[i, j] = f
        online_h = (t - W.online_since[d]) / 3600 if np.isfinite(W.online_since[d]) else 0.0
        sh = ((t / 3600 - W.d_shift[d] + 12) % 24) - 12
        own[i] = [online_h / 8, sh / 8, W.income[d] / E0 / 10, (t - W.idle_since[d]) / 600, np.log1p(prospect[dz]),
                  W.kappa[d] / kap_med - 1, *np.eye(3)[W.d_mode[d]]]
    return cols, opt, own, mask, fare


@torch.no_grad()
def driver_logits(pol, opt, own, mask, device):
    if not len(opt):
        return np.zeros((0, 1 + K_MENU))
    t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return pol(t(opt), t(own), t(mask)).float().cpu().numpy()


def choose(lg, seed, e, drivers, force=None):
    """Keyed draws per driver: 0 = decline, 1 + j = option j. force: {driver: action}."""
    if not len(lg):
        return np.zeros(0, np.int64)
    p = np.exp(lg - lg.max(1, keepdims=True))
    p = np.nan_to_num(p / p.sum(1, keepdims=True))
    a = RO.pick(p, RO.keyed_uniform(seed, e, "driver", drivers)).astype(np.int64)
    if force:
        for i, d in enumerate(drivers):
            if int(d) in force and (force[int(d)] == 0 or np.isfinite(lg[i, force[int(d)]])):
                a[i] = force[int(d)]
    return a


def settle(mi, menu, cols, a, sc):
    """Simultaneous choices -> offers (driver, request) that the drivers have accepted; a request
    taken by several drivers goes to the platform's best-scored one."""
    want = {}
    for i, c in enumerate(cols):
        if a[i] > 0:
            r = int(menu[c][a[i] - 1])
            want.setdefault(r, []).append(int(c))
    offers, lost = [], 0
    for r, cs in want.items():
        best = max(cs, key=lambda c: sc[r, c])
        offers.append((int(mi["drv"][best]), int(mi["req"][r])))
        lost += len(cs) - 1
    return offers, lost
