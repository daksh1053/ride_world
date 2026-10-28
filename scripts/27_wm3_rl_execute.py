"""World model 3, stage 5c: execute the agents' choices in the simulator.

    .venv/bin/python scripts/27_wm3_rl_execute.py --city sf_hd --seed0 700 --n-seeds 4
    .venv/bin/python scripts/27_wm3_rl_execute.py --summarise

For each seed a fresh season (new drivers and riders, never seen in training) runs WARMUP days with
the platform's usual dispatch, so the drivers have records; then one test-split day is run once per
platform, all from the same copied season state and identical until t0:
    none        no repositioning after t0 (matching only)
    rule        the platform's own rebalancing (HDDispatch, as in all training data)
    <agent>     the platform's matching + the agent's repositioning orders every 5 min (world/wm3/agent.py
                LiveAgentPolicy: the simulator's logs so far are rebuilt into the entity-level state
                and the agent decides from it, exactly as in imagination)
    <agent>+m   (stage 6) the agent's matching every 30 s + its repositioning every 5 min
    <agent>+m-none   the agent's matching, no repositioning
Measured over [t0, t0 + WINDOW_H]: riders picked up, cancellations, expired requests, waiting time,
drivers moved and followed, drivers' income, km and driving costs, the drivers' welfare gain
(agent.welfare) and the reward the agents were trained for (pickups + welfare).

The world model's view of the same choice: from the real state at t0 (the rule run's logs) it
imagines the window under every platform (DRAWS draws each, the same random numbers across
platforms), so its predicted differences can be compared with the simulator's.

Writes data/<city>/wm3_rl/exec/rec_seed<k>.json (+ the arms' day logs) and, with --summarise,
outputs/wm3/stage5/27_*.png / .csv / .json.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

if "--cpu" in sys.argv:                       # before torch: keep the GPU free (e.g. for training)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from world import cities
from world.hd import demand as DM
from world.hd.run import Season
from world.hd.sim import HDDispatch, HDWorld
from world.sim import SimConfig
from world.wm3 import agent as AG
from world.wm3 import features as FT
from world.wm3 import model as WM
from world.wm3 import rollout as RO
from world.wm3 import state as S

_spec = importlib.util.spec_from_file_location("tr3", Path(__file__).with_name("18_wm3_train.py"))
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)
FOLDER = "wm3_rl/exec"
WARMUP = 3
WINDOW_H = 2.0
DRAWS = 3


def load_agent(tag, tabs):
    """A trained agent, or 'untrained' (wm3_exo with a fresh choice head)."""
    if tag == "untrained":
        torch.manual_seed(0)
        return AG.Agent(TR.load("wm3_exo", tabs)[0]).to(TR.DEV).eval()
    ck = torch.load(TR.MODELS / f"{tag}.pt", map_location=TR.DEV)
    a = AG.Agent(WM.WorldModel3(tabs, **ck["arch"])).to(TR.DEV)
    AG.load_agent_state(a, ck["state"])
    return a.eval()


def load_drivers(tag):
    from world.wm3 import drivers as DR
    pol = DR.DriverPolicy().eval()
    if tag != "start":
        pol.load_state_dict(torch.load(TR.MODELS / f"{tag}.pt", map_location="cpu")["state"])
    return pol


def run_day_with(season, date, policy, horizon_h, day_dir):
    """Season.run_day, but the policy gets the live world (LiveAgentPolicy needs it)."""
    date = pd.Timestamp(date)
    day_seed = season.seed * 1000 + date.dayofyear
    cfg = SimConfig(date=str(date.date()), seed=day_seed, horizon_s=horizon_h * 3600)
    cond = season.data.traffic.sample_day(cfg.date, np.random.default_rng(day_seed + 7))
    rain = 0.0 if cond.rain is None else cond.rain[2] * min(1.0, (cond.rain[1] - cond.rain[0]) / 43200)
    clat = season.clat.copy()
    clat["activity"] = clat.activity * clat.loyalty
    scale = float(season.clat.loyalty.mean()) * (1 + 0.15 * rain)
    rng = np.random.default_rng(day_seed)
    req = DM.generate(season.city, season.geo, date, season.cust, clat, rng, sf=season.sf, rain=rain, scale=scale)
    req = req[req.t_request < cfg.horizon_s].reset_index(drop=True)
    req["request_id"] = np.arange(len(req))
    dlat = season.dlat.reset_index(drop=True).copy()
    from world.hd import people as PP
    dlat["days_per_week"] = np.clip(dlat.days_per_week * PP.DAY_WEIGHT[date.weekday()], 0, 7)
    drv_today = season.drv.reset_index(drop=True)
    dlat_today = season.dlat.reset_index(drop=True)
    world = HDWorld(season.city, season.data.net, season.data.nodes, season.data.zones, season.data.traffic,
                    drv_today, dlat, season.cust, season.clat, req, season.data.gw_pos, policy, cfg,
                    airport_zone=season.airport_zone)
    policy.world, policy.drivers_today, policy.latent_today, policy.day_dir = world, drv_today, dlat_today, day_dir
    world.run(log=lambda *a: None)
    tb = world.tables()
    day_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tb.items():
        df.to_parquet(day_dir / f"{name}.parquet", index=False)
    drv_today.to_parquet(day_dir / "drivers.parquet", index=False)
    dlat_today.to_parquet(day_dir / "drivers_latent.parquet", index=False)
    return tb


def arm_spec(arm):
    """arm name -> (agent tag or None, repositioning mode, matching mode); '<agent>+m@<drivers>'
    (stage 7): menus from <agent>, every driver choosing with the driver policy <drivers>
    (data/wm3_models/<drivers>.pt, or 'start': take the platform's first option)."""
    if "@" in arm:
        return arm_spec(arm.split("@")[0])
    if arm in ("rule", "none"):
        return None, arm, "rule"
    if arm.endswith("+m-none"):
        return arm[:-7], "none", "agent"
    if arm.endswith("+m"):
        return arm[:-2], "agent", "agent"
    return arm, "agent", "rule"


def real_metrics(L, t0, t1, cost_per_km):
    e0, e1 = int(t0 // S.DT), int(t1 // S.DT)
    ev = L.ev[(L.ev.t_event > t0) & (L.ev.t_event <= t1)]
    cnt = lambda t: int((ev.type == t).sum())  # noqa: E731
    r = L.req[(L.req.t_pickup > t0) & (L.req.t_pickup <= t1)]
    lg = L.legs[(L.legs.t_start >= t0) & (L.legs.t_start < t1)]
    km = np.bincount(lg.driver_id, weights=lg.length_m / 1000, minlength=L.nD)[:L.nD]
    inc0, inc1 = L.income[:, e0].astype(float), L.income[:, e1].astype(float)
    E0 = float(np.median(L.req.quote))
    net = np.maximum(inc1 - cost_per_km * km, -0.5 * E0)
    welfare = float(np.log1p(net / E0).sum() - np.log1p(np.maximum(inc0, -0.5 * E0) / E0).sum())
    on = (L.status[:, e0:e1 + 1] != 0).any(1)
    x = np.sort((inc1 - inc0)[on])
    gini = float((2 * np.arange(1, len(x) + 1) - len(x) - 1) @ x / (len(x) * max(x.sum(), 1e-9))) if len(x) > 1 else 0.0
    pk = cnt("pickup")
    hrs = (L.status[:, e0:e1] != 0).sum(1) * S.DT / 3600                     # online hours in the window
    netd = (inc1 - inc0) - cost_per_km * km
    act = hrs > 0.25
    o = L.offers[(L.offers.t_sent > t0) & (L.offers.t_sent <= t1)]
    return {"offers": len(o), "rejected": int((o.response == "rejected").sum()), "timeout": int((o.response == "timeout").sum()),
            "offer_eta_min": float(o.eta_est_s.mean() / 60) if len(o) else np.nan,
            "pickups": pk, "cancels": cnt("customer_cancelled"), "expired": cnt("request_expired"),
            "wait_min": float(((r.t_pickup - r.t_request) / 60).mean()) if len(r) else np.nan,
            "moved": cnt("dispatch_sent"), "followed": cnt("dispatch_accepted"), "income": float((inc1 - inc0).sum()),
            "gini": gini, "km": float(km.sum()), "cost": float((cost_per_km * km).sum()), "welfare": welfare,
            "net_per_h": float(netd[act].sum() / hrs[act].sum() / E0) if act.any() else np.nan,
            "net_per_h_p10": float(np.percentile(netd[act] / hrs[act], 10) / E0) if act.any() else np.nan,
            "driver_hours": float(hrs.sum()),
            "reward": pk + welfare}


def _quiet(*a, **k):
    pass


def warm_state(key, seed, root, warmup):
    """The season after its warm-up days (cached as a pickle)."""
    import pickle
    f = root / f"state_seed{seed}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    season = Season(key, seed=seed, policy_factory=HDDispatch, log=_quiet)
    date = test_date(key, seed)
    for k in range(warmup, 0, -1):
        season.run_day(date - pd.Timedelta(days=k), out_dir=root / f"seed{seed}")
    f.write_bytes(pickle.dumps(season, protocol=pickle.HIGHEST_PROTOCOL))
    return season


def test_date(key, seed):
    city = cities.get(key)
    split = json.loads((city.data_dir / "wm2" / "split.json").read_text())
    g = np.random.default_rng(seed)
    return pd.Timestamp(g.choice(pd.date_range(split["test_from"], split["test_until"])))


def run_seed(key, seed, args):
    """Runs whatever is missing for this seed: the warm-up, then each platform's real run and its
    imagined counterpart (agents only when their checkpoints exist). Results accumulate in rec_seed<k>.json."""
    city = cities.get(key)
    root = city.data_dir / args.folder
    root.mkdir(parents=True, exist_ok=True)
    rf = root / f"rec_seed{seed}.json"
    date = test_date(key, seed)
    t0 = float([8, 12, 17][seed % 3] * 3600)
    t1 = t0 + args.window_h * 3600
    rec = json.loads(rf.read_text()) if rf.exists() else {"key": key, "seed": seed, "date": str(date.date()), "t0": t0, "t1": t1,
                                                          "warmup": args.warmup, "arms": {}, "imagined": {}}
    state = warm_state(key, seed, root, args.warmup)
    if args.warmup_only:
        return rec
    gr = S.CityGraph(city)
    rules = RO.CityRules(gr)
    prior = FT.Prior()
    for d in sorted(p for p in (root / f"seed{seed}").iterdir() if p.is_dir()):
        prior.update_light(d, 5000)
    tabs = TR.tables()
    arms = ["rule", "none"] + [a for a in args.agents if arm_spec(a)[0] == "untrained" or (TR.MODELS / f"{arm_spec(a)[0]}.pt").exists()]
    agents = {}
    for arm in arms:
        day_dir = root / f"seed{seed}_{arm}" / str(date.date())
        if arm in rec["arms"]:
            if "net_per_h" not in rec["arms"][arm] and day_dir.exists():       # metrics added in stages 6-7
                L = S.DayLog(gr, day_dir)
                rec["arms"][arm].update(real_metrics(L, t0, t1, L.drivers.cost_per_km.to_numpy()))
                rf.write_text(json.dumps(rec, indent=1, default=float))
            continue
        ta = time.time()
        tag, mode, match = arm_spec(arm)
        if tag is not None:
            agents[tag] = agents.get(tag) or load_agent(tag, tabs)
        pol = AG.LiveAgentPolicy(agents.get(tag), gr, rules, prior, t_start=t0, seed=seed, mode=mode, match=match,
                                 drivers=load_drivers(arm.split("@")[1]) if "@" in arm else None)
        run_day_with(copy.deepcopy(state), date, pol, t1 / 3600 + 0.25, day_dir)
        L = S.DayLog(gr, day_dir)
        rec["arms"][arm] = {**real_metrics(L, t0, t1, L.drivers.cost_per_km.to_numpy()),
                            "decisions": pol.decisions, "matches": pol.matches, "menus": pol.menus, "minutes": (time.time() - ta) / 60}
        print(f"  {key} seed {seed} {arm}: " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                                         for k, v in rec["arms"][arm].items() if k not in ("decisions", "matches", "menus")), flush=True)
        rf.write_text(json.dumps(rec, indent=1, default=float))
    if args.no_imagine:
        return rec
    todo = [a for a in arms if a not in rec["imagined"]]
    if todo:
        env = TR.load("wm3_exo", tabs)[0].eval()
        Lb = S.DayLog(gr, root / f"seed{seed}_rule" / str(date.date()))
        e0 = int(t0 // S.DT)
        H = int(args.window_h * 3600 // S.DT)
        chunk = {"sf_hd": 8, "pune_hd": 2}[key]
        for arm in todo:
            draws = []
            for s in [11, 12, 13][:args.draws]:
                W = AG.AgentWorld(env, rules, Lb, prior, e0, None, seed=s)
                W.L = None
                log0, inc0 = len(W.log), W.income.copy()
                tag, mode, match = arm_spec(arm)
                if tag is None:
                    W.setup(None, s, mode)
                else:
                    agents[tag] = agents.get(tag) or load_agent(tag, tabs)
                    W.setup(agents[tag], s, mode, match=match)
                    if "@" in arm:                                   # stage 7: menus, agent drivers
                        W.drivers, W.override_driver = load_drivers(arm.split("@")[1]), {}
                AG.rollout_many([W], H, chunk)
                m = AG.outcome(W, log0, inc0)
                draws.append({"draw": s, **m, "reward": m["pickups"] + m["welfare"], "held": getattr(W, "n_held", 0)})
                torch.cuda.empty_cache()
            rec["imagined"][arm] = draws
            im = pd.DataFrame(draws)[["pickups", "welfare", "reward", "moved"]].mean()
            print(f"  {key} seed {seed} imagined {arm}: reward {im.reward:.1f} pickups {im.pickups:.1f} welfare {im.welfare:.1f} moved {im.moved:.0f}", flush=True)
            rf.write_text(json.dumps(rec, indent=1, default=float))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["sf_hd", "pune_hd"])
    ap.add_argument("--seed0", type=int, default=700)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--agents", nargs="+", default=["rl_pre", "rl_scratch"])
    ap.add_argument("--window-h", type=float, default=WINDOW_H)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--folder", default=FOLDER)
    ap.add_argument("--warmup-only", action="store_true")
    ap.add_argument("--no-imagine", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        summarise()
        return
    for seed in range(args.seed0, args.seed0 + args.n_seeds):
        run_seed(args.city, seed, args)


def summarise():
    pass


if __name__ == "__main__":
    main()
