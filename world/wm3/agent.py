"""World model 3 as an agent: the platform's repositioning choices made by the model itself.

Every decision epoch (5 min) each idle driver who has waited at least IDLE_MIN_S and got no offer
this epoch is either left where it is or ordered to one zone within REACH_S of its zone. The
choice is a categorical over [stay, zone_0 .. zone_Z-1] from the model's driver and zone token
states:
    stay      MLP(h_driver) + a bias + logsumexp of the zone scores (P(move) starts near P_MOVE0)
    zone z    q(h_driver) . k(h_zone) / sqrt(d) + w_tt * minutes to z  (unreachable zones masked)
Matching (offers) is the platform's rule (HDDispatch with rebalancing switched off), or, from
stage 6, the agent's own (match="agent"): every 30-s epoch the pending requests, oldest first,
each pick one idle driver of their mode within the pickup limit (not one who declined them, not
one an earlier request took) or "hold" (wait for the next epoch):
    driver j  qm(h_request) . km(h_driver) / sqrt(d) + w_eta * pickup minutes
    hold      MLP(h_request) + b_hold
so a whole assignment has a probability (the product of the requests' choices) for RL.

The same Agent works in three places:
  - inside an imagined world (world/wm3/rollout.py), for training with GRPO
  - inside the simulator (LiveAgentPolicy): the simulator's logs so far are rebuilt into the
    same entity-level state (state.DayLog), so the agent sees exactly what it saw in training
  - as a world model: Agent.wm is a WorldModel3 and keeps all its prediction heads
Random draws are keyed per (seed, epoch, driver) like the rollout's, so two worlds that differ
only in one driver's choice draw the same luck for everyone else.
"""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ..hd.sim import HDDispatch
from ..policies import Action, Policy
from . import data as WD
from . import features as FT
from . import rollout as RO
from . import state as S

DECIDE_EVERY = 10            # epochs (5 min), as the platform's own rebalancing
IDLE_MIN_S = 300.0
REACH_S = 900.0
P_MOVE0 = 0.03              # starting chance that a deciding driver is moved
STAY_BIAS = math.log((1 - P_MOVE0) / P_MOVE0)
IDLE = 1
RO.KIND.setdefault("policy", max(RO.KIND.values()) + 1)
RO.KIND.setdefault("match", max(RO.KIND.values()) + 1)
MATCH_KEYS = ("mq.", "mk.", "hold.", "w_eta", "b_hold")
B_HOLD0 = -4.0                # the rule never holds a request that has a driver in reach


class Agent(nn.Module):
    def __init__(self, wm):
        super().__init__()
        self.wm = wm
        d = wm.d
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.stay = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.w_tt = nn.Parameter(torch.zeros(()))
        self.b_stay = nn.Parameter(torch.tensor(STAY_BIAS))
        # matching (stage 6)
        self.mq = nn.Linear(d, d)
        self.mk = nn.Linear(d, d)
        self.hold = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.w_eta = nn.Parameter(torch.tensor(-0.5))
        self.b_hold = nn.Parameter(torch.tensor(B_HOLD0))

    def match_logits(self, b, rpos, dpos, feas, eta):
        """b: collated snapshots; rpos: (B, n) positions of the deciding requests among the request
        tokens (-1 = padding); dpos: (B, m) positions of the candidate drivers among the driver
        tokens; feas: (B, n, m) bool; eta: (B, n, m) pickup minutes. Returns (B, n, 1 + m) logits
        [hold, driver_1..driver_m], -inf where not allowed (before masking drivers taken earlier)."""
        hD, hR, _, _ = self.wm.encode(b)
        hD, hR = hD.float(), hR.float()
        d = hD.shape[-1]
        hr = torch.gather(hR, 1, rpos.clamp(min=0)[..., None].expand(-1, -1, d))
        hd = torch.gather(hD, 1, dpos.clamp(min=0)[..., None].expand(-1, -1, d))
        pair = torch.einsum("bnd,bmd->bnm", self.mq(hr), self.mk(hd)) / math.sqrt(d) + self.w_eta * eta
        pair = pair.masked_fill(~feas, float("-inf"))
        hold = self.hold(hr)[..., 0] + self.b_hold
        return torch.cat([hold[..., None], pair], -1)

    def logits(self, b, pos, cand, tt):
        """b: a collated batch of one snapshot per element; pos: (B, n) long positions of the
        deciding drivers among the driver tokens (-1 = padding); cand: (B, n, Z) bool; tt: (B, n, Z)
        minutes. Returns (B, n, 1 + Z) logits, -inf where not allowed."""
        hD, _, hZ, _ = self.wm.encode(b)
        hD, hZ = hD.float(), hZ.float()
        B, n = pos.shape
        hd = torch.gather(hD, 1, pos.clamp(min=0)[..., None].expand(-1, -1, hD.shape[-1]))
        mv = torch.einsum("bnd,bzd->bnz", self.q(hd), self.k(hZ)) / math.sqrt(hD.shape[-1]) + self.w_tt * tt / 10
        mv = mv.masked_fill(~cand, float("-inf"))
        lse = torch.logsumexp(mv, -1)
        lse = torch.where(cand.any(-1), lse, torch.zeros_like(lse))
        # stay is scored against all moves together, so P(move) starts near P_MOVE0 whatever the
        # number of reachable zones
        stay = self.stay(hd)[..., 0] + self.b_stay + lse
        return torch.cat([stay[..., None], mv], -1)


def decision_inputs(W, f):
    """Who decides in imagined world W (features f), with candidate zones and travel minutes."""
    R = W.R
    on = f["D_id"]
    offered = {i for i, _, _ in W.offers_now}
    ok = (W.st[on] == IDLE) & (W.t - W.idle_since[on] >= IDLE_MIN_S) & ~np.isin(on, list(offered))
    pos = np.flatnonzero(ok)
    h = int(W.t // 3600) % 24
    ztt = R.zone_tt[W.daytype, h]
    dz = W.g.zone[W.node[on[pos]]]
    tt = ztt[dz]                                                    # (n, Z) seconds
    cand = tt <= REACH_S
    cand[np.arange(len(pos)), dz] = False
    return {"pos": pos, "drv": on[pos], "cand": cand, "tt": (tt / 60).astype(np.float32)}


def snap_of(W, f):
    return {"key": W.key, **{k: f[k] for k in ("G", "D", "F", "R", "Z")}}


def collate_decisions(snaps, decs, device):
    """One batch of snapshots plus padded decision arrays."""
    b = WD.collate([{**s, **_empty_targets(s)} for s in snaps], device)
    n = max(1, max(len(d["pos"]) for d in decs))
    Z = snaps[0]["Z"].shape[0]
    B = len(snaps)
    pos = np.full((B, n), -1, np.int64)
    cand = np.zeros((B, n, Z), bool)
    tt = np.zeros((B, n, Z), np.float32)
    for i, d in enumerate(decs):
        m = len(d["pos"])
        pos[i, :m], cand[i, :m], tt[i, :m] = d["pos"], d["cand"], d["tt"]
    t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return b, t(pos), t(cand), t(tt)


def _empty_targets(s):
    nD, nF, nR, nZ = len(s["D"]), len(s["F"]), len(s["R"]), len(s["Z"])
    return {"D_ev": np.zeros(nD, np.int8), "D_repo": np.full(nD, -1), "D_offer": np.full(nD, -1),
            "D_order": np.full(nD, -1), "F_on": np.zeros(nF, np.int8), "R_cancel": np.zeros(nR, np.int8),
            "R_board": np.zeros(nR, np.int8), "Z_new": np.zeros((nZ, 3), np.int16),
            "NEW": np.zeros((0, len(FT.NEW_COLS)), np.float32)}


@torch.no_grad()
def probs(agent, W, f, dec):
    """(n, 1 + Z) choice probabilities for one world."""
    if not len(dec["pos"]):
        return np.zeros((0, W.R.Z + 1))
    dev = next(agent.parameters()).device
    b, pos, cand, tt = collate_decisions([snap_of(W, f)], [dec], dev)
    with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
        lg = agent.logits(b, pos, cand, tt)
    return torch.softmax(lg[0].float(), -1).cpu().numpy()


def choose(p, seed, e, drv):
    """Keyed draws: action index per deciding driver (0 = stay, 1 + z = go to zone z)."""
    if not len(drv):
        return np.zeros(0, np.int64)
    return RO.pick(p, RO.keyed_uniform(seed, e, "policy", drv)).astype(np.int64)


def to_dispatches(dec, a):
    return [(int(d), int(x - 1)) for d, x in zip(dec["drv"], a) if x > 0]


# ---------------------------------------------------------------------- matching (stage 6)
def load_agent_state(agent, state):
    """Load an agent checkpoint; checkpoints from before stage 6 have no matching head (it keeps
    its fresh initialisation)."""
    missing, unexpected = agent.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith(MATCH_KEYS)] + list(unexpected)
    assert not bad, f"agent checkpoint mismatch: {bad[:5]}"
    return agent


def match_inputs(obs, f):
    """Who can be offered to whom now (an Observation, from an imagined world or the simulator, and
    the features f of the same moment): the pending requests, oldest first, and the idle drivers;
    a pair is allowed when the modes agree, the pickup ETA (the platform's GPS estimate) is within
    the limit and the driver has not declined that request (as HDDispatch). Only requests with an
    allowed driver decide. None if there is nothing to decide."""
    pend, idle = obs.pending, obs.idle
    if not len(pend) or not len(idle):
        return None
    rp = {int(k): j for j, k in enumerate(f["R_id"])}
    dp = {int(d): j for j, d in enumerate(f["D_id"])}
    pend = pend[pend.request_id.isin(list(rp))].sort_values(["t_request", "request_id"], kind="stable")
    idle = idle[idle.driver_id.isin(list(dp))]
    if not len(pend) or not len(idle):
        return None
    req, drv = pend.request_id.to_numpy().astype(np.int64), idle.driver_id.to_numpy().astype(np.int64)
    rm, dm = pend["mode"].to_numpy(), idle["mode"].to_numpy()
    eta = np.full((len(req), len(drv)), np.inf)
    for j in np.unique(rm):
        ri, di = np.flatnonzero(rm == j), np.flatnonzero(dm == j)
        if len(ri) and len(di):
            e = obs.fleet["eta"](idle.node.to_numpy()[di][:, None], pend.origin_node.to_numpy()[ri][None, :], int(j))
            eta[np.ix_(ri, di)] = np.asarray(e, float).T
    for c, ex in enumerate(pend.excluded):
        if ex:
            eta[c, np.isin(drv, list(ex))] = np.inf
    feas = eta <= obs.max_pickup_s
    rk, dk = feas.any(1), feas.any(0)
    if not rk.any():
        return None
    feas, eta = feas[rk][:, dk], eta[rk][:, dk]
    req, drv = req[rk], drv[dk]
    return {"req": req, "rpos": np.array([rp[int(k)] for k in req], np.int64), "drv": drv,
            "dpos": np.array([dp[int(d)] for d in drv], np.int64), "feas": feas,
            "eta": np.where(feas, eta / 60, 0.0).astype(np.float32)}


def collate_matches(snaps, mis, device):
    """One batch of snapshots plus padded matching inputs."""
    b = WD.collate([{**s, **_empty_targets(s)} for s in snaps], device)
    n = max(1, max(len(m["req"]) for m in mis))
    m_ = max(1, max(len(m["drv"]) for m in mis))
    B = len(snaps)
    rpos, dpos = np.full((B, n), -1, np.int64), np.full((B, m_), -1, np.int64)
    feas, eta = np.zeros((B, n, m_), bool), np.zeros((B, n, m_), np.float32)
    for i, m in enumerate(mis):
        a, c = len(m["req"]), len(m["drv"])
        rpos[i, :a], dpos[i, :c], feas[i, :a, :c], eta[i, :a, :c] = m["rpos"], m["dpos"], m["feas"], m["eta"]
    t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return b, t(rpos), t(dpos), t(feas), t(eta)


def _softmax_np(l):
    l = l - np.max(l)
    p = np.exp(l)
    return p / p.sum()


def choose_match(lg, seed, e, req, force=None):
    """Sequential keyed draws: request j (oldest first) picks from [hold] + the drivers not taken by
    requests before it. lg: (n, 1 + m) logits. force: {request_id: action} for forced choices
    (GRPO branches; a forced driver already taken falls back to hold). Returns (n,) actions
    (0 = hold, 1 + c = candidate driver c)."""
    n, m1 = lg.shape
    used = np.zeros(m1 - 1, bool)
    a = np.zeros(n, np.int64)
    u = RO.keyed_uniform(seed, e, "match", req)
    for j in range(n):
        l = lg[j].astype(np.float64).copy()
        l[1:][used] = -np.inf
        if force is not None and int(req[j]) in force:
            x = int(force[int(req[j])])
            a[j] = x if np.isfinite(l[x]) else 0
        else:
            a[j] = int(RO.pick(_softmax_np(l)[None], u[[j]])[0])
        if a[j] > 0:
            used[a[j] - 1] = True
    return a


def used_before(a, m):
    """(n, m) bool: the candidate drivers taken by the requests before each one."""
    U = np.zeros((len(a), m), bool)
    for j in range(1, len(a)):
        U[j] = U[j - 1]
        if a[j - 1] > 0:
            U[j, a[j - 1] - 1] = True
    return U


def seq_logprobs(lg, U):
    """lg: (..., n, 1 + m) logits, U: (..., n, m) taken-before masks -> log-probabilities of each
    request's choice given the drivers still free (torch)."""
    mask = torch.cat([torch.zeros_like(U[..., :1]), U], -1)
    return torch.log_softmax(lg.masked_fill(mask, float("-inf")), -1)


def to_offers(mi, a):
    return [(int(mi["drv"][x - 1]), int(k)) for k, x in zip(mi["req"], a) if x > 0]


def rule_targets(mi, offers):
    """The rule's offers as the agent's actions (0 = hold) and how many it made outside the agent's
    choice set (e.g. the airport queue beyond the pickup limit)."""
    col = {int(d): c for c, d in enumerate(mi["drv"])}
    of = {int(k): int(d) for d, k in offers}
    a = np.zeros(len(mi["req"]), np.int64)
    miss = 0
    for j, k in enumerate(mi["req"]):
        d = of.get(int(k))
        if d is None:
            continue
        c = col.get(d)
        if c is not None and mi["feas"][j, c]:
            a[j] = 1 + c
        else:
            miss += 1
    return a, miss


@torch.no_grad()
def match_logits_many(items, device):
    """items: [(agent, snap, mi)] -> list of (n, 1 + m) numpy logits (one batch per agent)."""
    out = [None] * len(items)
    for aid in {id(x[0]) for x in items}:
        grp = [i for i, x in enumerate(items) if id(x[0]) == aid]
        agent = items[grp[0]][0]
        for c in range(0, len(grp), 8):
            g = grp[c: c + 8]
            b, rpos, dpos, feas, eta = collate_matches([items[i][1] for i in g], [items[i][2] for i in g], device)
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                lg = agent.match_logits(b, rpos, dpos, feas, eta)
            lg = lg.float().cpu().numpy()
            for j, i in enumerate(g):
                mi = items[i][2]
                out[i] = lg[j, :len(mi["req"]), :1 + len(mi["drv"])]
    return out


def menu_round(W, mi, lg, f):
    """Stage 7 (world/wm3/drivers.py): every idle driver with an allowed request sees a menu (the
    platform's K best requests for it, with the fare it will get) and takes one or declines; the
    taken rides are accepted offers. Records the round in W.last_menu."""
    from . import drivers as DR
    menu, sc = DR.menus(mi, lg)
    cols, opt, own, mask, fare = DR.features(W, mi, menu)
    dl = DR.driver_logits(W.drivers, opt, own, mask, torch.device("cpu"))
    drv = mi["drv"][cols]
    a = DR.choose(dl, W.policy_seed, W.e, drv, W.override_driver.get(W.e) if hasattr(W, "override_driver") else None)
    offers, lost = DR.settle(mi, menu, cols, a, sc)
    W.preaccept = set(offers)
    # the driver each request would go to if every driver showing it took it (platform's best)
    owner = {}
    for c in cols:
        for r in menu[c]:
            if r not in owner or sc[r, c] > sc[r, owner[r]]:
                owner[r] = c
    winnable = np.array([any(owner[r] == c for r in menu[c]) for c in cols], bool)
    W.last_menu = {"drv": drv, "opt": opt, "own": own, "mask": mask, "fare": fare, "lg": dl, "a": a, "e": W.e,
                   "offers": offers, "lost": lost, "winnable": winnable}
    W.n_declined = getattr(W, "n_declined", 0) + int((a == 0).sum())
    W.n_lost = getattr(W, "n_lost", 0) + lost
    return offers


# ---------------------------------------------------------------------- imagined worlds
class AgentWorld(RO.ImaginedWorld):
    """An imagined world whose platform matches with HDDispatch and repositions with an agent
    (or with nothing / the platform's own rule, for baselines)."""

    def setup(self, agent=None, policy_seed=0, mode="agent", match="rule"):
        """mode: 'agent' (the agent repositions), 'rule' (the platform's own rebalancing every
        5 min), 'none' (no repositioning). match: 'rule' (HDDispatch's min-ETA matching) or
        'agent' (the agent's matching, every epoch)."""
        self.agent, self.policy_seed, self.mode, self.match_mode = agent, policy_seed, mode, match
        self.override_match = {}                             # {epoch: {request: action}} forced (branches)
        self.last_match = None
        self.n_held = 0
        self.match = HDDispatch(reposition_every=DECIDE_EVERY if mode == "rule" else 0, max_moves=60)
        self.match._n = int(self.e) - 1                      # the rule rebalances when e % 10 == 0
        self.override = {}                                   # {epoch: actions} forced choices (branches)
        self.n_moved = 0
        self.km = np.zeros(self.nD)                          # km driven since setup (operating costs)
        return self

    def _start_leg(self, d, target, t, kind, req=-1):
        leg = super()._start_leg(d, target, t, kind, req)
        if hasattr(self, "km"):
            self.km[d] += leg.length_m / 1000
        return leg

    def step_agent(self):
        """One 30-s step (as step_many on this world alone). Returns the repositioning decision
        record at decision epochs, else None; the matching record is in self.last_match."""
        return step_many([self])[0]

    def _on_board(self, t, k, push, t1):
        q = self.req.get(k)
        created = q["created"] if q is not None else None
        n = len(self.log)
        super()._on_board(t, k, push, t1)
        if len(self.log) > n and self.log[-1][1] == "pickup":
            self.wait_s = getattr(self, "wait_s", 0.0) + (t - created)

    def _dispatch(self, disp):
        """Repositioning orders on top of this step's offers (as apply_action)."""
        offered = {i for i, _, _ in self.offers_now}
        seen = set()
        for i, z in disp:
            if i in offered or i in seen or self.st[i] != IDLE or not (0 <= z < self.R.Z):
                continue
            seen.add(i)
            self._status(i, RO.REPO, self.t)
            self.repo_zone[i] = z
            self.count["orders"][i] += 1
            self.orders_now.append((i, int(z)))
            self.log.append((self.t, "dispatch_sent", i, -1, int(z), 0.0))


SHARED = ("model", "R", "g", "router", "road_cache", "agent", "L")


def detach(W):
    """The world's own state without the shared heavy parts (for pickling and cloning)."""
    keep = {k: getattr(W, k, None) for k in SHARED}
    for k in SHARED:
        setattr(W, k, None)
    return keep


def attach(W, keep):
    for k, v in keep.items():
        setattr(W, k, v)
    return W


def clone(W):
    keep = detach(W)
    try:
        W2 = copy.deepcopy(W)
    finally:
        attach(W, keep)
    return attach(W2, keep)


def save_start(W, path):
    import pickle
    keep = detach(W)
    try:
        Path(path).write_bytes(pickle.dumps(W, protocol=pickle.HIGHEST_PROTOCOL))
    finally:
        attach(W, keep)


def load_start(path, model, rules, router=None):
    import pickle
    from ..routing import LiveRouter
    W = pickle.loads(Path(path).read_bytes())
    if router is None:
        router = LiveRouter(rules.g.net, rules.g.traffic)
    router.set_day(W.day)
    return attach(W, {"model": model, "R": rules, "g": rules.g, "router": router, "road_cache": {}, "agent": None, "L": None})


# ---------------------------------------------------------------------- the live simulator
class LiveAgentPolicy(Policy):
    """Runs inside world/hd/sim.HDWorld. Matching: HDDispatch (no rebalancing), or the agent's
    (match="agent", every epoch). Repositioning (mode): the agent's every DECIDE_EVERY epochs,
    the rule's or none. Whenever the agent decides, the simulator's logs so far are rebuilt into
    the entity-level state (state.DayLog, as stage 1, in live mode) and the agent decides from it.
    Before t_start it is the platform's usual rule, so all runs are identical until t_start."""

    name = "wm3_agent"
    version = "2"

    def __init__(self, agent, graph, rules, prior, t_start, seed=0, base_kw=None, mode="agent", match="rule", drivers=None):
        """drivers: a stage-7 DriverPolicy -- with match="agent", idle drivers then get menus and
        choose (world/wm3/drivers.py); the rides they take are accepted (world/hd/sim.py preaccepted)."""
        self.drivers = drivers
        self.agent, self.g, self.rules, self.prior = agent, graph, rules, prior
        self.t_start, self.seed, self.mode, self.match_mode = t_start, seed, mode, match
        self.base = HDDispatch(**(base_kw or {}))
        self.match = HDDispatch(reposition_every=0)
        self.rule = HDDispatch(**(base_kw or {}))
        self.world = None                                    # set by the runner
        self.day_dir = None                                  # name gives the date and season seed
        self.decisions = []
        self.matches = []
        self.menus = []

    def act(self, obs):
        if obs.t < self.t_start:
            self.rule._n += 1                                 # keep the rule's 5-min phase
            return self.base.act(obs)
        e = int(round(obs.t / S.DT))
        if self.match_mode == "rule":
            if self.mode == "rule":
                return self.rule.act(obs)
            act = self.match.act(obs)
            if self.mode == "none" or e % DECIDE_EVERY:
                return act
            W = self.imagined_now(e)
            offers = act.offers
        else:
            self.rule._n += 1
            W = self.imagined_now(e)
            f = W.features()
            mi = match_inputs(obs, f)
            offers = []
            if mi is not None:
                dev = next(self.agent.parameters()).device
                lg = match_logits_many([(self.agent, snap_of(W, f), mi)], dev)[0]
                if self.drivers is not None:                 # menus; the drivers choose (stage 7)
                    W.policy_seed, W.drivers = self.seed, self.drivers
                    offers = menu_round(W, mi, lg, f)
                    a = np.zeros(len(mi["req"]), np.int64)
                    taken = {k: d for d, k in offers}
                    col = {int(d): c for c, d in enumerate(mi["drv"])}
                    for j, k in enumerate(mi["req"]):
                        if int(k) in taken:
                            a[j] = 1 + col[taken[int(k)]]
                    self.world.preaccepted = getattr(self.world, "preaccepted", set()) | set(offers)
                    self.menus.append({"t": obs.t, "n_menus": len(W.last_menu["drv"]), "declined": int((W.last_menu["a"] == 0).sum()),
                                       "lost": W.last_menu["lost"], "taken": len(offers)})
                else:
                    a = choose_match(lg, self.seed, e, mi["req"])
                    offers = to_offers(mi, a)
                self.matches.append({"t": obs.t, "n_req": len(mi["req"]), "n_drv": len(mi["drv"]), "n_offers": len(offers),
                                     "n_pending": len(obs.pending), "eta_min": float(np.mean([mi["eta"][j, x - 1] for j, x in enumerate(a) if x > 0]))
                                     if len(offers) else np.nan})
            if self.mode == "rule":                          # the rule's rebalancing on top of the agent's offers
                used = {d for d, _ in offers}
                o2 = copy.copy(obs)
                o2.idle = obs.idle[~obs.idle.driver_id.isin(list(used))]
                disp = self.rule._rebalance(o2, o2.idle) if e % DECIDE_EVERY == 0 and len(o2.idle) else []
                return Action(offers=offers, dispatches=disp)
            if self.mode == "none" or e % DECIDE_EVERY:
                return Action(offers=offers)
        W.apply_action(Action(offers=offers))
        f = W.features()
        dec = decision_inputs(W, f)
        p = probs(self.agent, W, f, dec)
        a = choose(p, self.seed, e, dec["drv"])
        disp = to_dispatches(dec, a)
        self.decisions.append({"t": obs.t, "n_deciding": len(dec["drv"]), "n_moved": len(disp)})
        return Action(offers=offers, dispatches=disp)

    def imagined_now(self, e):
        """The entity-level state at epoch e from the simulator's tables so far (in memory)."""
        tb = self.world.tables()
        tb["drivers"], tb["drivers_latent"] = self.drivers_today, self.latent_today
        L = S.DayLog(self.g, self.day_dir, tables=tb, live_e=e)
        W = RO.ImaginedWorld(self.agent.wm, self.rules, L, self.prior, e, None, seed=self.seed)
        W.L = None
        return W


# ---------------------------------------------------------------------- lockstep worlds (one GPU batch)
@torch.no_grad()
def step_many(worlds):
    """Advance several imagined worlds of one city by one 30-s step together: the agent's and the
    world model's forward passes are batched. Same result as step_agent() on each world.
    Returns the decision records (or None) per world."""
    recs = [None] * len(worlds)
    am = [i for i, W in enumerate(worlds) if getattr(W, "match_mode", "rule") == "agent"]
    for i, W in enumerate(worlds):
        W.last_match = None
        if i in am:
            continue
        obs = W.observe()
        rec_m = getattr(W, "record_match", 0) and W.e % W.record_match == 0
        if rec_m:                                    # the rule's offers as agent actions (imitation targets)
            f = W.features()
            mi = match_inputs(obs, f)
        act = W.match.act(obs)
        if rec_m and mi is not None:
            a, miss = rule_targets(mi, act.offers)
            W.last_match = {"snap": snap_of(W, f), "mi": mi, "a": a, "e": W.e, "miss": miss, "n_rule": len(act.offers)}
        W.apply_action(Action(offers=act.offers))
        W._pending_disp = act.dispatches
    if am:                                           # the agent matches (one GPU batch per agent)
        fs, mis = {}, {}
        for i in am:
            fs[i] = worlds[i].features()
            mis[i] = match_inputs(worlds[i].observe(), fs[i])
        live = [i for i in am if mis[i] is not None]
        dev = next(worlds[live[0]].agent.parameters()).device if live else None
        lgs = match_logits_many([(worlds[i].agent, snap_of(worlds[i], fs[i]), mis[i]) for i in live], dev) if live else []
        for i, lg in zip(live, lgs):
            W, mi = worlds[i], mis[i]
            if getattr(W, "drivers", None) is not None:          # stage 7: menus, the drivers choose
                offers = menu_round(W, mi, lg, fs[i])
            else:
                a = choose_match(lg, W.policy_seed, W.e, mi["req"], W.override_match.get(W.e))
                W.last_match = {"snap": snap_of(W, fs[i]), "mi": mi, "lg": lg, "a": a, "e": W.e}
                offers = to_offers(mi, a)
                W.n_held += int((a == 0).sum())
            W.apply_action(Action(offers=offers))
        for i in am:
            W = worlds[i]
            if i not in live:
                W.apply_action(Action())
            W._pending_disp = []
            if W.mode == "rule":                     # the rule's rebalancing on top of the agent's offers
                W.match._n += 1
                obs = W.observe()
                if W.match.reposition_every and W.match._n % W.match.reposition_every == 0 and len(obs.idle):
                    W._pending_disp = W.match._rebalance(obs, obs.idle)
            else:
                W.match._n += 1
    for i, W in enumerate(worlds):                 # the rule's choices as agent actions (imitation targets)
        if W.mode == "rule" and getattr(W, "record", False) and W.e % DECIDE_EVERY == 0:
            f = W.features()
            dec = decision_inputs(W, f)
            a = np.zeros(len(dec["drv"]), np.int64)
            at = {int(d): j for j, d in enumerate(dec["drv"])}
            for d, z in W._pending_disp:
                if int(d) in at and 0 <= z < W.R.Z and dec["cand"][at[int(d)], z]:
                    a[at[int(d)]] = 1 + int(z)
            recs[i] = {"snap": snap_of(W, f), "dec": dec, "a": a, "e": W.e, "n_rule": len(W._pending_disp)}
    deciding = [i for i, W in enumerate(worlds) if W.mode == "agent" and W.e % DECIDE_EVERY == 0]
    if deciding:
        fs = {i: worlds[i].features() for i in deciding}
        decs = {i: decision_inputs(worlds[i], fs[i]) for i in deciding}
        live = [i for i in deciding if len(decs[i]["pos"])]
        ps = {i: np.zeros((0, worlds[i].R.Z + 1)) for i in deciding}
        for aid in {id(worlds[i].agent) for i in live}:             # one batch per agent
            grp = [i for i in live if id(worlds[i].agent) == aid]
            agent = worlds[grp[0]].agent
            dev = next(agent.parameters()).device
            b, pos, cand, tt = collate_decisions([snap_of(worlds[i], fs[i]) for i in grp], [decs[i] for i in grp], dev)
            lg = agent.logits(b, pos, cand, tt)
            pr = torch.softmax(lg.float(), -1).cpu().numpy()
            for j, i in enumerate(grp):
                ps[i] = pr[j, :len(decs[i]["pos"])]
        for i in deciding:
            W, dec, p = worlds[i], decs[i], ps[i]
            a = W.override[W.e] if W.e in W.override else choose(p, W.policy_seed, W.e, dec["drv"])
            recs[i] = {"snap": snap_of(W, fs[i]), "dec": dec, "p": p, "a": a, "e": W.e}
            W._pending_disp = to_dispatches(dec, a)
    for W in worlds:
        W._dispatch(W._pending_disp)
        W.n_moved += len(W.orders_now)
    fs = [W.features() for W in worlds]
    env = worlds[0].model
    dev = next(env.parameters()).device
    b = WD.collate([{**snap_of(W, f), **_empty_targets(snap_of(W, f))} for W, f in zip(worlds, fs)], dev)
    with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
        out = env(b)
    hZd_all = out.get("hZd", out["h"][2]).float()
    for j, (W, f) in enumerate(zip(worlds, fs)):
        nD, nF, nR = len(f["D"]), len(f["F"]), len(f["R"])
        cut = {"ev": nD, "offer": nD, "order": nD, "repo": nD, "online": nF, "cancel": nR, "board": nR, "rate": None}
        P = {k: (out[k][j][:n] if n is not None else out[k][j]).float().cpu() for k, n in cut.items()}
        W.hZd = hZd_all[j]
        W._sample_and_apply(f, P, out["h"][2][j].float())
    return recs


def rollout_many(worlds, H, chunk=8):
    """H lockstep steps (forward passes batched over `chunk` worlds); returns the decision records
    of each world (list per world)."""
    recs = [[] for _ in worlds]
    for _ in range(H):
        for c in range(0, len(worlds), chunk):
            for i, r in enumerate(step_many(worlds[c: c + chunk])):
                if r is not None:
                    recs[c + i].append(r)
    return recs


def welfare(W, inc0, km0=None):
    """Drivers' proportional-fairness welfare gain: sum over drivers of log(1 + net / E0), now minus
    at the start, with net = today's income (inc0 at the start) minus the cost of the km driven since
    then; E0 = the median quote. A ride counts more for a driver who has earned little today."""
    E0 = W.money
    km = W.km - (0 if km0 is None else km0)
    net = np.maximum(W.income - W.kappa * km, -0.5 * E0)
    return float(np.log1p(net / E0).sum() - np.log1p(np.maximum(inc0, -0.5 * E0) / E0).sum())


def outcome(W, log0, inc0, wait0=0.0):
    """What happened in an imagined world since log index log0 (incomes inc0 at that time)."""
    typ = [x[1] for x in W.log[log0:]]
    inc = W.income - inc0
    on = W.st != 0
    x = np.sort(inc[on]) if on.any() else np.zeros(1)
    gini = float((2 * np.arange(1, len(x) + 1) - len(x) - 1) @ x / (len(x) * max(x.sum(), 1e-9))) if len(x) > 1 else 0.0
    pk = typ.count("pickup")
    return {"pickups": pk, "cancels": typ.count("customer_cancelled"), "expired": typ.count("request_expired"),
            "wait_min": (getattr(W, "wait_s", 0.0) - wait0) / 60 / max(pk, 1), "moved": W.n_moved,
            "followed": typ.count("dispatch_accepted"), "income": float(inc.sum()), "gini": gini,
            "km": float(W.km.sum()), "cost": float((W.kappa * W.km).sum()), "welfare": welfare(W, inc0)}
