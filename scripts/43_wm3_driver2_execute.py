"""World model 3, stage 7b: run the richer driver agents in the simulator, in the tuned market.

    .venv/bin/python scripts/43_wm3_driver2_execute.py --city sf_hd --seed0 700 --n-seeds 8 --drivers drv2

For each seed: the warmed-up season of stage 5 (scripts/27_wm3_rl_execute.py: fresh drivers and
riders, never seen in training), one test day, all arms from the same state and identical until t0.
At t0 the market of stage 7b starts (outputs/wm3/stage7b/market.json): a share of the drivers idle
or on a break log off for the rest of the day (availability), demand is multiplied by the market's demand factor from t0 on, and drivers are paid by the tuned
scheme (applied to the simulator's finished rides afterwards; riders pay the same fares).
    single   the stage-6 platform (mbc_pre) offers one ride at a time; the simulator's people
             accept or reject as they do in all training data
    asap     exclusive menus; every driver takes the first option, never moves
    agents   exclusive menus; every driver uses the trained driver policy
    mixed    exclusive menus; 25% of the drivers (the same in every seed's draw as in imagination)
             use the trained policy, the rest ASAP
Measured over [t0, t0 + 2 h]: riders picked up, wait, cancellations; drivers' pay, utility and
net per online hour under the tuned scheme; the 25%'s gain in 'mixed' over the same drivers in
'asap'; declines and drivers' own moves.

Writes data/<city>/wm3_rl/exec7b/rec_seed<k>_<drivers>.json (+ day logs).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from world import cities
from world.hd import sim as HS
from world.wm3 import agent as AG
from world.wm3 import drivers2 as D2
from world.wm3 import features as FT
from world.wm3 import pay as PY
from world.wm3 import rollout as RO
from world.wm3 import state as S

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EX = _mod("ex27", "27_wm3_rl_execute.py")
TR = EX.TR
G42 = _mod("g42", "42_wm3_driver2_grpo.py")
_quiet = EX._quiet          # the cached warm-up seasons were pickled from script 27 run as __main__
ARMS = ("single", "asap", "agents", "mixed")
FOLDER = "wm3_rl/exec7b"


class Live7b(AG.LiveAgentPolicy):
    """The stage-6 platform with stage 7b drivers (or single offers), plus the supply cut at t0."""

    def __init__(self, agent, *a, drivers=None, supply=0.0, **kw):
        super().__init__(agent, *a, mode="agent", match="agent", drivers=drivers, **kw)
        self.driver_v = 2 if drivers is not None else 1
        self.supply, self.cut_done, self.banned = supply, False, set()

    def act(self, obs):
        if not self.cut_done and obs.t >= self.t_start:
            self.cut_done = True
            w = self.world
            cand = np.flatnonzero((w.d_status == HS.IDLE) | (w.d_status == HS.BREAK))
            g = np.random.default_rng(self.seed * 101 + int(self.supply * 100))     # as ImaginedWorld.cut_supply
            out = [int(d) for d in cand if g.random() < self.supply]
            self.banned.update(out)
            for d in out:
                w.on_shift_end(obs.t, d)
            obs = copy.copy(obs)
            obs.idle = obs.idle[~obs.idle.driver_id.isin(out)]
            self.removed = out
        return super().act(obs)


def run_real(state, date, pol, t1, day_dir, demand=1.0):
    """EX.run_day_with with banned drivers kept offline (as scripts/36_wm3_shock.py) and, from t0,
    a demand surge: (demand - 1) times the day's requests drawn again (own random numbers, the same
    in every arm) and those from t0 on added, so all arms are identical before t0."""
    orig = HS.HDWorld.__init__
    gen = EX.DM.generate

    def generate(city, geo, date_, cust, clat, rng, **kw):
        req = gen(city, geo, date_, cust, clat, rng, **kw)
        if demand <= 1:
            return req
        g = np.random.default_rng(pol.seed * 7919 + pd.Timestamp(date_).dayofyear)
        extra = gen(city, geo, date_, cust, clat, g, **{**kw, "scale": kw.get("scale", 1.0) * (demand - 1)})
        extra = extra[extra.t_request >= pol.t_start]
        return pd.concat([req, extra]).sort_values("t_request").reset_index(drop=True)

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        on_online = self.on_online

        def online(t, i, shift_end):
            if i in pol.banned:
                return
            on_online(t, i, shift_end)
        self.on_online = online
    HS.HDWorld.__init__ = init
    EX.HDWorld = HS.HDWorld
    EX.DM.generate = generate
    try:
        return EX.run_day_with(state, date, pol, t1 / 3600 + 0.25, day_dir)
    finally:
        HS.HDWorld.__init__ = orig
        EX.DM.generate = gen


def sim_trips(L, t0, t1):
    """Finished rides in (t0, t1] from a simulator day as trip records (world/wm3/pay.py)."""
    r = L.req
    r = r[(r.t_dropoff > t0) & (r.t_dropoff <= t1) & (r.driver_id >= 0)]
    key = L.g.city.key
    ext = r.external_dest.fillna(-1).astype(int).to_numpy()
    mk = cities.get(key).market
    out_km = np.array([mk.external[e].outside_km if e >= 0 else 0.0 for e in ext])
    return pd.DataFrame({"t": r.t_dropoff.to_numpy(), "driver": r.driver_id.astype(int).to_numpy(),
                         "mode": r["mode"].map({m: i for i, m in enumerate(S.MODES)}).to_numpy(),
                         "accept_t": r.t_accept.to_numpy(), "pickup_t": r.t_pickup.to_numpy(),
                         "km_in": np.maximum(r.distance_km.to_numpy() - out_km, 0.0), "outside_km": out_km,
                         "minutes": ((r.t_dropoff - r.t_pickup) / 60).to_numpy(), "surge": r.surge.to_numpy(), "ext": ext})


def driver_metrics(L, t0, t1, scheme):
    """Per driver over the window: net pay under the scheme (finished rides; km of legs started in
    the window, intercity km twice), utility with the driver types, online hours."""
    key, market = L.g.city.key, cities.get(L.g.city.key).market
    tr = sim_trips(L, t0, t1)
    v = np.zeros(L.nD)
    np.add.at(v, tr.driver.to_numpy(), PY.trips_pay(scheme, key, market, tr))
    lg = L.legs[(L.legs.t_start >= t0) & (L.legs.t_start < t1)]
    km = np.bincount(lg.driver_id, weights=lg.length_m / 1000, minlength=L.nD)[:L.nD]
    np.add.at(km, tr.driver.to_numpy(), 2 * tr.outside_km.to_numpy())
    kappa = L.drivers.cost_per_km.to_numpy()
    net = v - kappa * km
    e0, e1 = int(t0 // S.DT), int(t1 // S.DT)
    money = float(np.median(L.req.quote))
    T = D2.driver_types(key, L.nD, money)
    inc0 = L.income[:, e0].astype(float)
    u = D2.utility(T, np.arange(L.nD), inc0 + net) - D2.utility(T, np.arange(L.nD), inc0)
    hrs = (L.status[:, e0:e1] != 0).sum(1) * S.DT / 3600
    return u, net, hrs, money, float(v.sum())


def run_seed(key, seed, args, scheme, supply, policies):
    city = cities.get(key)
    root = city.data_dir / FOLDER
    root.mkdir(parents=True, exist_ok=True)
    rf = root / f"rec_seed{seed}_{args.drivers}.json"
    date = EX.test_date(key, seed)
    t0 = float([8, 12, 17][seed % 3] * 3600)
    t1 = t0 + args.window_h * 3600
    rec = json.loads(rf.read_text()) if rf.exists() else {"key": key, "seed": seed, "date": str(date.date()), "t0": t0, "t1": t1,
                                                          "supply_cut": supply, "demand": args.demand, "drivers": args.drivers, "arms": {}}
    state = EX.warm_state(key, seed, city.data_dir / EX.FOLDER, EX.WARMUP)
    gr = S.CityGraph(city)
    rules = RO.CityRules(gr)
    prior = FT.Prior()
    for d in sorted(p for p in (city.data_dir / EX.FOLDER / f"seed{seed}").iterdir() if p.is_dir()):
        prior.update_light(d, 5000)
    tabs = TR.tables()
    agent = EX.load_agent(args.platform, tabs)
    nD = len(state.drv)
    grp = (RO.keyed_uniform(7, 0, "mix", np.arange(nD)) < G42.MIX_SHARE).astype(int)
    per = {}
    thr = min(0.25, 0.5 * args.window_h)          # drivers online at least this long (h) count in per-hour figures
    for arm in ARMS:
        day_dir = root / f"seed{seed}_{arm}_{args.drivers}" / str(date.date())
        if arm not in rec["arms"]:
            ta = time.time()
            drivers = {"single": None, "asap": policies["asap"], "agents": policies["agents"],
                       "mixed": D2.Mixed([policies["asap"], policies["agents"]], grp)}[arm]
            pol = Live7b(agent, gr, rules, prior, t_start=t0, seed=seed, drivers=drivers, supply=supply)
            run_real(copy.deepcopy(state), date, pol, t1, day_dir, args.demand)
            L = S.DayLog(gr, day_dir)
            rec["arms"][arm] = {**EX.real_metrics(L, t0, t1, L.drivers.cost_per_km.to_numpy()), "removed": len(pol.removed),
                                "menus": pol.menus, "minutes": (time.time() - ta) / 60}
        L = S.DayLog(gr, day_dir)
        u, net, hrs, money, paid = driver_metrics(L, t0, t1, scheme)
        per[arm] = (u, net, hrs)
        act = hrs > thr
        m = rec["arms"][arm]
        mn = m.get("menus") or []
        m.update({"u_per_h": float(u[act].sum() / hrs[act].sum() / money), "pay_net_per_h": float(net[act].sum() / hrs[act].sum() / money),
                  "pay_net_per_h_p10": float(np.percentile(net[act] / hrs[act], 10) / money), "paid": paid,
                  "declined": int(sum(x.get("declined", 0) for x in mn)), "self_moves": int(sum(x.get("moves", 0) for x in mn))})
        print(f"  {key} seed {seed} {arm}: pickups {m['pickups']} wait {m['wait_min']:.2f} u/h {m['u_per_h']:.3f} "
              f"net/h {m['pay_net_per_h']:.3f} declined {m['declined']} moves {m['self_moves']} ({m.get('minutes', 0):.1f} min)", flush=True)
        rf.write_text(json.dumps(rec, indent=1, default=float))
    f = (grp == 1) & (per["asap"][2] > thr) & (per["mixed"][2] > thr)
    o = (grp == 0) & (per["asap"][2] > thr)
    money = float(np.median(S.DayLog(gr, root / f"seed{seed}_asap_{args.drivers}" / str(date.date())).req.quote))
    rec["arms"]["mixed"].update({
        "focal_u_gain_per_h": float((per["mixed"][0][f] - per["asap"][0][f]).sum() / per["asap"][2][f].sum() / money),
        "focal_net_gain_per_h": float((per["mixed"][1][f] - per["asap"][1][f]).sum() / per["asap"][2][f].sum() / money),
        "others_net_gain_per_h": float((per["mixed"][1][o] - per["asap"][1][o]).sum() / per["asap"][2][o].sum() / money)})
    rf.write_text(json.dumps(rec, indent=1, default=float))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["sf_hd", "pune_hd"], required=True)
    ap.add_argument("--seed0", type=int, default=700)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--drivers", default="drv2", help="trained driver policy (data/wm3_models/<name>.pt)")
    ap.add_argument("--platform", default="mbc_pre")
    ap.add_argument("--window-h", type=float, default=2.0)
    args = ap.parse_args()
    scheme, supply, K, share, batch = G42.market_config(args.city)
    args.demand = G42.market_demand(args.city)
    D2.configure(args.city, cities.get(args.city).market, scheme, K, share, batch)
    p0 = D2.asap_policy()
    ck = torch.load(TR.MODELS / f"{args.drivers}.pt", map_location="cpu")
    p1 = D2.DriverPolicy2(**ck.get("arch", {})).eval()
    p1.load_state_dict(ck["state"])
    for seed in range(args.seed0, args.seed0 + args.n_seeds):
        run_seed(args.city, seed, args, scheme, supply, {"asap": p0, "agents": p1})


if __name__ == "__main__":
    main()
