"""World model 3 as an agent: the platform's repositioning choices made by the model itself.

Every decision epoch (5 min) each idle driver who has waited at least IDLE_MIN_S and got no offer
this epoch is either left where it is or ordered to one zone within REACH_S of its zone. The
choice is a categorical over [stay, zone_0 .. zone_Z-1] from the model's driver and zone token
states:
    stay      MLP(h_driver) + a bias + logsumexp of the zone scores (P(move) starts near P_MOVE0)
    zone z    q(h_driver) . k(h_zone) / sqrt(d) + w_tt * minutes to z  (unreachable zones masked)
Matching (offers) stays the platform's rule (HDDispatch with rebalancing switched off).

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


# ---------------------------------------------------------------------- imagined worlds
class AgentWorld(RO.ImaginedWorld):
    """An imagined world whose platform matches with HDDispatch and repositions with an agent
    (or with nothing / the platform's own rule, for baselines)."""

    def setup(self, agent=None, policy_seed=0, mode="agent"):
        """mode: 'agent' (the agent repositions), 'rule' (the platform's own rebalancing every
        5 min), 'none' (no repositioning)."""
        self.agent, self.policy_seed, self.mode = agent, policy_seed, mode
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
        """One 30-s step: match, (every 5 min) reposition, then predict and apply the events.
        Returns the decision record {snap, dec, p, a} at decision epochs, else None."""
        act = self.match.act(self.observe())
        self.apply_action(Action(offers=act.offers))
        disp, rec = act.dispatches, None
        if self.mode == "agent" and self.e % DECIDE_EVERY == 0:
            f = self.features()
            dec = decision_inputs(self, f)
            p = probs(self.agent, self, f, dec)
            a = self.override[self.e] if self.e in self.override else choose(p, self.policy_seed, self.e, dec["drv"])
            rec = {"snap": snap_of(self, f), "dec": dec, "p": p, "a": a, "e": self.e}
            disp = to_dispatches(dec, a)
        self._dispatch(disp)
        self.n_moved += len(self.orders_now)
        self.advance()
        return rec

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
    """Runs inside world/hd/sim.HDWorld. Matching: HDDispatch (no rebalancing). Every
    DECIDE_EVERY epochs from t_start, the simulator's logs so far are rebuilt into the
    entity-level state (state.DayLog, as stage 1) and the agent orders idle drivers.
    Before t_start (and with agent=None) it is the platform's usual rule, so runs with
    different agents are identical until t_start."""

    name = "wm3_agent"
    version = "1"

    def __init__(self, agent, graph, rules, prior, t_start, seed=0, base_kw=None, mode="agent"):
        self.agent, self.g, self.rules, self.prior = agent, graph, rules, prior
        self.t_start, self.seed, self.mode = t_start, seed, mode
        self.base = HDDispatch(**(base_kw or {}))
        self.match = HDDispatch(reposition_every=0)
        self.rule = HDDispatch(**(base_kw or {}))
        self.world = None                                    # set by the runner
        self.day_dir = None                                  # name gives the date and season seed
        self.decisions = []

    def act(self, obs):
        if obs.t < self.t_start:
            self.rule._n += 1                                 # keep the rule's 5-min phase
            return self.base.act(obs)
        if self.mode == "rule":
            return self.rule.act(obs)
        act = self.match.act(obs)
        e = int(round(obs.t / S.DT))
        if self.mode == "none" or e % DECIDE_EVERY:
            return act
        W = self.imagined_now(e)
        W.apply_action(Action(offers=act.offers))
        f = W.features()
        dec = decision_inputs(W, f)
        p = probs(self.agent, W, f, dec)
        a = choose(p, self.seed, e, dec["drv"])
        disp = to_dispatches(dec, a)
        self.decisions.append({"t": obs.t, "n_deciding": len(dec["drv"]), "n_moved": len(disp)})
        return Action(offers=act.offers, dispatches=disp, info=act.info)

    def imagined_now(self, e):
        """The entity-level state at epoch e from the simulator's tables so far."""
        tb = self.world.tables()
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            d = Path(tmp) / self.day_dir.parent.name / self.day_dir.name
            d.mkdir(parents=True)
            for name, df in tb.items():
                df.to_parquet(d / f"{name}.parquet", index=False)
            self.drivers_today.to_parquet(d / "drivers.parquet", index=False)
            self.latent_today.to_parquet(d / "drivers_latent.parquet", index=False)
            L = S.DayLog(self.g, d)
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
    for W in worlds:
        act = W.match.act(W.observe())
        W.apply_action(Action(offers=act.offers))
        W._pending_disp = act.dispatches
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
