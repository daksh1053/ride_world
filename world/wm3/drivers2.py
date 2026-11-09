"""Stage 7b: richer driver agents, in a market where taking the first ride at once is not the best.

What changes from stage 7 (world/wm3/drivers.py):

Market (set per run with configure()):
  * exclusive menus  each request is on ONE driver's menu (its best-scored driver's, up to K = 6),
                     so every option a driver sees is one it gets if it takes it. In stage 7 a
                     request was on several menus and went to the platform's best driver, so a
                     driver's options after the first were mostly ones it would lose anyway.
  * driver pay       a budget-neutral pay scheme (world/wm3/pay.py) instead of the rider's fare
                     minus commission; riders still pay the same fares.
  * availability     a share of the drivers taken off the road (rollout.ImaginedWorld.cut_supply).

Drivers (one policy shared by all, each driver with its own type and situation):
  * types   private preferences, drawn once per driver from the city's seed: a daily income target
            (money above it is worth only above_w < 1 per unit: reference-dependent drivers, as in
            Camerer et al. 1997 / Farber 2015), a shift length, and a pull towards home near the end
            of the shift (cost per minute of drive home); plus the known cost per km.
  * actions wait, take option j of the menu (j < 6), or drive to one of M = 3 nearby zones the driver
            picks itself (the zones within 15 min with the most recent requests per idle driver).
            Drivers with a menu decide every 30 s; idle drivers without one decide every 5 min
            (between the platform's repositioning rounds) whether to wait or move.
  * policy  tokens for the driver, each option and each move; two self-attention layers let the
            driver weigh its options against each other; logits per token. It starts as stage 7's
            start policy: take the first option, never move.
  * aim     its utility over the next 45 minutes: u(income) with the income target, minus the home
            pull at the end if its shift is (nearly) over. Income counts pay minus the cost of km.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import pay as PY
from . import rollout as RO
from . import drivers as D1

K = 6
M_MOVES = 3
OPT_F = 16
MOVE_F = 9
OWN_F = 19
FREE_EVERY = 10          # epochs (5 min) between decisions of idle drivers without a menu
FREE_PHASE = 5           # ... at e % 10 == 5, between the platform's repositioning rounds (e % 10 == 0)
MOVE_MAX_MIN = 15.0
RO.KIND.setdefault("driver2", max(RO.KIND.values()) + 1)
RO.KIND.setdefault("mix", max(RO.KIND.values()) + 1)

CFG = {"scheme": None, "key": None, "market": None}


def configure(key, market, scheme, K_menu=K, share=1, batch=1):
    """The market for this process: pay scheme; menus of length K_menu, each request on the menus of
    its `share` best-scored drivers, sent every `batch` epochs."""
    global K
    K = int(K_menu)
    CFG.update(key=key, market=market, scheme=scheme)
    D1.set_menu_size(K, "exclusive", share, batch)


# ---------------------------------------------------------------------- driver types
@dataclass
class Types:
    target: np.ndarray       # daily net income target, money
    above_w: np.ndarray      # value of a unit of money above the target (below: 1)
    shift_h: np.ndarray      # hours the driver means to work
    home_pull: np.ndarray    # cost per minute of drive home at the end of the shift, money


HOURLY = {"sf_hd": 1.3, "pune_hd": 1.2}        # typical net per online hour, median fares (stage 5-7 runs)


def driver_types(key, nD, money, seed=2026):
    g = np.random.default_rng([seed, nD, sum(map(ord, key))])
    shift = np.clip(g.normal(8.0, 1.5, nD), 4.0, 12.0)
    target = HOURLY[key] * money * shift * np.exp(g.normal(0.0, 0.3, nD))
    above = 0.3 + 0.6 * g.beta(4, 4, nD)
    pull = 0.03 * money * np.exp(g.normal(0.0, 0.5, nD))
    return Types(target, above, shift, pull)


def types_of(W):
    if getattr(W, "dtypes", None) is None or len(W.dtypes.target) != W.nD:
        W.dtypes = driver_types(W.key, W.nD, W.money)
    return W.dtypes


def utility(T, d, income):
    """u(income) with the target: slope 1 below, above_w above (money)."""
    x = income - T.target[d]
    return np.where(x < 0, income, T.target[d] + T.above_w[d] * x)


def shift_left_h(W, d):
    T = types_of(W)
    on = W.online_since[d]
    if not np.isfinite(on):
        return T.shift_h[d]
    return T.shift_h[d] - (W.t - on) / 3600


def home_minutes(W, d, node=None):
    R, g = W.R, W.g
    h = int(W.t // 3600) % 24
    if node is None:
        node = int(W.leg[d].nodes[-1]) if W.leg[d] is not None else int(W.node[d])
    return R.zone_tt[W.daytype, h, int(g.zone[node]), int(W.home_zone[d])] / 60


def home_cost(W, d):
    """The pull home at the end: cost of the drive home if the shift ends within 30 min."""
    if shift_left_h(W, d) > 0.5:
        return 0.0
    return float(types_of(W).home_pull[d] * home_minutes(W, d))


# ---------------------------------------------------------------------- policy
class DriverPolicy2(nn.Module):
    def __init__(self, h=64, layers=2, heads=4):
        super().__init__()
        self.own = nn.Sequential(nn.Linear(OWN_F, h), nn.GELU(), nn.Linear(h, h))
        self.opt = nn.Sequential(nn.Linear(OPT_F + OWN_F, h), nn.GELU(), nn.Linear(h, h))
        self.mov = nn.Sequential(nn.Linear(MOVE_F + OWN_F, h), nn.GELU(), nn.Linear(h, h))
        self.typ = nn.Parameter(torch.zeros(3, h))
        layer = nn.TransformerEncoderLayer(h, heads, 2 * h, dropout=0.0, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.ModuleList([nn.Linear(h, 1) for _ in range(3)])        # wait, option, move
        for m in self.head:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.b_wait = nn.Parameter(torch.tensor(-3.0))
        self.b_move = nn.Parameter(torch.tensor(-8.0))     # moves ~2% at a free decision
        self.w_rank = nn.Parameter(torch.tensor(-2.0))     # starts as "take the platform's first option"

    def forward(self, opt, mov, own, omask, mmask):
        """opt (B, K, OPT_F), mov (B, M, MOVE_F), own (B, OWN_F), masks (B, K) / (B, M) ->
        logits (B, 1 + K + M): [wait, options, moves]."""
        B, k, m = opt.shape[0], opt.shape[1], mov.shape[1]
        oe = own[:, None]
        x = torch.cat([self.own(own)[:, None] + self.typ[0],
                       self.opt(torch.cat([opt, oe.expand(-1, k, -1)], -1)) + self.typ[1],
                       self.mov(torch.cat([mov, oe.expand(-1, m, -1)], -1)) + self.typ[2]], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=opt.device), ~omask, ~mmask], 1)
        y = self.mix(x, src_key_padding_mask=pad)
        lw = self.head[0](y[:, 0])[:, 0] + self.b_wait
        lo = self.head[1](y[:, 1:1 + k])[..., 0] + self.w_rank * opt[..., 0]
        lm = self.head[2](y[:, 1 + k:])[..., 0] + self.b_move
        lo = lo.masked_fill(~omask, float("-inf"))
        lm = lm.masked_fill(~mmask, float("-inf"))
        return torch.cat([lw[:, None], lo, lm], -1)


def asap_policy():
    """The exact ASAP rule as a policy: always the platform's first option, never wait while
    offered a ride, never move (the soft starting policy declines ~5% and moves ~2%)."""
    p = DriverPolicy2().eval()
    with torch.no_grad():
        p.b_wait.fill_(-30.0)
        p.b_move.fill_(-40.0)
        p.w_rank.fill_(-20.0)
    return p


class Mixed:
    """Different policies for different drivers: policies[i] for drivers with group[d] == i."""

    def __init__(self, policies, group):
        self.policies, self.group = policies, np.asarray(group)


# ---------------------------------------------------------------------- features
def _prospects(W):
    t, g, Z = W.t, W.g, W.R.Z
    Hh = np.array(W.hist) if W.hist else np.zeros((0, 4))
    rec = np.bincount(Hh[(Hh[:, 0] >= t - 1800) & (Hh[:, 3] > 0), 1].astype(int), minlength=Z) if len(Hh) else np.zeros(Z)
    idle_z = np.bincount(g.zone[W.node[W.st == RO.IDLE]], minlength=Z)
    pend = np.bincount([g.zone[q["origin"]] for q in W.req.values() if q["status"] == "pending"], minlength=Z)
    return rec / (idle_z + 1.0), pend


def features(W, drivers, menu_rows, mi):
    """For deciding drivers (ids) with their menus (lists of request ids, possibly empty): option,
    move and own features, masks, the move zones and the options' expected pay."""
    R, g, t = W.R, W.g, W.t
    h = int(t // 3600) % 24
    tt, dist = R.zone_tt[W.daytype, h], R.zone_dist[W.daytype, h]
    E0 = W.money
    T = types_of(W)
    prospect, pend = _prospects(W)
    n = len(drivers)
    opt = np.zeros((n, K, OPT_F), np.float32)
    mov = np.zeros((n, M_MOVES, MOVE_F), np.float32)
    own = np.zeros((n, OWN_F), np.float32)
    om = np.zeros((n, K), bool)
    mm = np.zeros((n, M_MOVES), bool)
    zones = np.full((n, M_MOVES), -1, np.int64)
    payx = np.zeros((n, K))
    kap_med = float(np.median(W.kappa))
    sch, key, market = CFG["scheme"], CFG["key"], CFG["market"]
    trips = getattr(W, "trips", [])
    for i, d in enumerate(drivers):
        d = int(d)
        dz = int(g.zone[W.node[d]])
        hz = int(W.home_zone[d])
        rows = menu_rows[i]
        for j, k in enumerate(rows[:K]):
            q = W.req[int(k)]
            oz = int(g.zone[q["origin"]])
            if q["ext"] >= 0:
                e = R.ext[q["ext"]]
                end = int(g.zone[R.gw_pos[q["ext"]]])
                trip, km_in, out_km = tt[oz, end] / 60 * R.speed[q["mode"]] + e.outside_min, max(dist[oz, end], 800) / 1000, e.outside_km
            else:
                end = int(g.zone[q["dest"]])
                trip, km_in, out_km = tt[oz, end] / 60 * R.speed[q["mode"]], max(dist[oz, end], 800) / 1000, 0.0
            c = int(np.flatnonzero(mi["drv"] == d)[0])
            r = int(np.flatnonzero(mi["req"] == k)[0])
            pick = float(mi["eta"][r, c])
            km_pick = dist[dz, oz] / 1000
            p = float(PY.pay(sch, key, market, q["mode"], km_in, out_km, trip, q["surge"], pick + 1.0))
            cost = W.kappa[d] * (km_pick + km_in + 2 * out_km)
            eng = max(pick + 1 + trip, 1.0)
            opt[i, j] = [j, pick / 10, trip / 30, p / E0, cost / E0, (p - cost) / E0, (p - cost) / E0 / eng * 30,
                         q["surge"] - 1, float(q["ext"] >= 0), np.log1p(prospect[end]), tt[end, hz] / 3600,
                         float(end == hz), (km_in + out_km) / 10, np.log1p(pend[end]), q["quote"] / E0, 1.0]
            om[i, j] = True
            payx[i, j] = p
        near = np.flatnonzero((tt[dz] / 60 <= MOVE_MAX_MIN) & (np.arange(R.Z) != dz))
        if len(near):
            best = near[np.argsort(-(prospect[near] + 0.2 * pend[near]), kind="stable")][:M_MOVES]
            for j, z in enumerate(best):
                km = dist[dz, z] / 1000
                mov[i, j] = [tt[dz, z] / 60 / MOVE_MAX_MIN, np.log1p(prospect[z]), W.surge[z, W.d_mode[d]] - 1,
                             tt[z, hz] / 3600, W.kappa[d] * km / E0, np.log1p(pend[z]), float(z == hz),
                             np.log1p(prospect[z]) - np.log1p(prospect[dz]), 1.0]
                mm[i, j] = True
                zones[i, j] = z
        online_h = (t - W.online_since[d]) / 3600 if np.isfinite(W.online_since[d]) else 0.0
        inc = float(W.income[d])
        last_h = sum(1 for x in trips if x["driver"] == d and x["t"] > t - 3600)
        own[i] = [online_h / 8, shift_left_h(W, d) / 4, inc / T.target[d], T.target[d] / E0 / 20, T.above_w[d],
                  T.home_pull[d] / E0 * 30, tt[dz, hz] / 3600, (t - W.idle_since[d]) / 600, np.log1p(prospect[dz]),
                  W.surge[dz, W.d_mode[d]] - 1, W.kappa[d] / kap_med - 1, *np.eye(3)[W.d_mode[d]], last_h / 3,
                  float(len(rows) == 0), np.log1p(pend[dz]), float(dz == hz), 1.0]
    return opt, mov, own, om, mm, zones, payx


@torch.no_grad()
def logits(pol, drivers, opt, mov, own, om, mm):
    if not len(opt):
        return np.zeros((0, 1 + K + M_MOVES))
    tt = lambda x: torch.from_numpy(x)  # noqa: E731
    if isinstance(pol, Mixed):
        out = np.zeros((len(opt), 1 + K + M_MOVES), np.float32)
        grp = pol.group[np.asarray(drivers, int)]
        for gi, p in enumerate(pol.policies):
            s = np.flatnonzero(grp == gi)
            if len(s):
                out[s] = p(tt(opt[s]), tt(mov[s]), tt(own[s]), tt(om[s]), tt(mm[s])).float().numpy()
        return out
    return pol(tt(opt), tt(mov), tt(own), tt(om), tt(mm)).float().numpy()


def choose(lg, seed, e, drivers, force=None):
    """Keyed draws per driver: 0 = wait, 1 + j = option j, 1 + K + m = move m. force {driver: action}."""
    if not len(lg):
        return np.zeros(0, np.int64)
    p = np.exp(lg - lg.max(1, keepdims=True))
    p = np.nan_to_num(p / p.sum(1, keepdims=True))
    a = RO.pick(p, RO.keyed_uniform(seed, e, "driver2", drivers)).astype(np.int64)
    if force:
        for i, d in enumerate(drivers):
            if int(d) in force and np.isfinite(lg[i, force[int(d)]]):
                a[i] = force[int(d)]
    return a


# ---------------------------------------------------------------------- one round
def round_(W, mi, lg_match):
    """Every driver with a menu (exclusive menus from the platform's matching logits), and at free
    epochs every other idle driver, chooses. Returns (offers [(driver, request)], moves [(driver,
    zone)]) and records the round in W.last_menu."""
    menu, sc = D1.menus(mi, lg_match) if mi is not None and W.e % D1.BATCH == 0 else ({}, None)
    rows = {int(mi["drv"][c]): [int(mi["req"][r]) for r in rr] for c, rr in menu.items()} if mi is not None else {}
    deciders = sorted(rows)
    if W.e % FREE_EVERY == FREE_PHASE:
        free = [int(d) for d in np.flatnonzero(W.st == RO.IDLE) if int(d) not in rows]
        deciders = sorted(set(deciders) | set(free))
    if not deciders:
        W.last_menu = None
        return [], []
    menus = [rows.get(d, []) for d in deciders]
    opt, mov, own, om, mm, zones, payx = features(W, deciders, menus, mi if mi is not None else {"drv": np.zeros(0), "req": np.zeros(0)})
    lg = logits(W.drivers, deciders, opt, mov, own, om, mm)
    force = W.override_driver.get(W.e) if hasattr(W, "override_driver") else None
    a = choose(lg, W.policy_seed, W.e, deciders, force)
    offers, moves, declined = [], [], []
    for i, d in enumerate(deciders):  # exclusive menus: a taken option is the driver's ride
        if 1 <= a[i] <= K:
            offers.append((d, menus[i][a[i] - 1]))
            continue
        if a[i] > K:
            moves.append((d, int(zones[i, a[i] - 1 - K])))
        for k in menus[i]:              # waiting or moving declines the menu: not shown to this driver again
            declined.append((d, int(k)))
            W.req[int(k)]["excluded"].add(d)
    lost = 0
    if len(offers) > len({k for _, k in offers}):        # shared menus: a request taken by several drivers
        col = {int(d): c for c, d in enumerate(mi["drv"])}  # goes to the platform's best-scored one
        row = {int(k): r for r, k in enumerate(mi["req"])}
        best = {}
        for d, k in offers:
            if k not in best or sc[row[k], col[d]] > sc[row[k], col[best[k]]]:
                best[k] = d
        lost = len(offers) - len(best)
        offers = [(d, k) for k, d in best.items()]
    W.n_lost = getattr(W, "n_lost", 0) + lost
    W.preaccept = set(offers)
    W.last_menu = {"drv": np.array(deciders), "opt": opt, "mov": mov, "own": own, "om": om, "mm": mm, "lg": lg, "a": a,
                   "e": W.e, "zones": zones, "pay": payx, "n_opt": om.sum(1), "declined": declined}
    W.n_declined = getattr(W, "n_declined", 0) + int(((a == 0) & om.any(1)).sum())
    W.n_self_moves = getattr(W, "n_self_moves", 0) + len(moves)
    return offers, moves


def self_move(W, d, z):
    """Driver d drives to zone z on its own (a self_reposition leg, as the simulator's drivers do)."""
    if W.st[d] != RO.IDLE:
        return False
    W._status(d, RO.REPO, W.t)
    W.repo_zone[d] = z
    W.log.append((W.t, "driver_self_reposition", d, -1, int(z), 0.0))
    W._start_leg(d, W.R.zone_centre[z], W.t, "self_reposition")
    return True


# ---------------------------------------------------------------------- the driver's aim
def driver_utility(W, d, t0, km0, inc0):
    """Driver d's utility from t0 to now in W (money): u(today's income at t0 + net earned since,
    under the pay scheme) - u(income at t0) - the pull home if its shift is (nearly) over."""
    rec = PY.focal_record(W, d, t0, km0)
    v = PY.focal_value(rec, CFG["scheme"], CFG["key"], CFG["market"], W.kappa[d])
    T = types_of(W)
    return float(utility(T, d, inc0 + v) - utility(T, d, inc0)) - home_cost(W, d), v
