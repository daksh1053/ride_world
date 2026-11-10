"""Stage 7b, step 1: is "take the first ride you are offered, at once" a driver's best choice?

    .venv/bin/python scripts/40_wm3_market_probe.py --menu shared --supply 0.0
    .venv/bin/python scripts/40_wm3_market_probe.py --menu exclusive --supply 0.5

Inside the world model (wm3_exo), with the stage-6 platform (mbc_pre) giving every idle driver a
menu of its K best requests and every driver taking its first option (the stage-7 start policy),
one driver's choice at one moment is branched: copies of the imagined city that differ only in
that driver declining or taking option 1..K (everyone else uses the same random numbers). Each
copy runs H steps and saves what that driver did: its finished rides (trip records with pickup
time, km, minutes, surge), the ride in progress at the end and the km driven. Driver pay is not
fixed here: scripts/41_wm3_market_tune.py prices the same rides under any pay scheme
(world/wm3/pay.py), so many schemes are compared on one set of imagined branches.

--demand y: a demand surge; the world model's new-request rates are multiplied by y.

--supply x: availability. A share x of the drivers idle or on a break at the start log off and stay
off (the supply shock of stage 8, which the world model gets right in direction and size).

--menu: 'exclusive' (each request on one driver's menu only) or 'shared' (stage 7's K best per driver).

Writes data/<city>/wm3_rl/market_probe/probe_<menu>_s<supply>.pkl (one record per branched choice).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.multiprocessing as mp

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
CHUNK = {"sf_hd": 8, "pune_hd": 4}
_W = {}


def _tr():
    spec = importlib.util.spec_from_file_location("tr3", ROOT / "scripts" / "18_wm3_train.py")
    TR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TR)
    return TR


def worker_init(key, env_tag, platform, K, menu, share, batch):
    from world import cities
    from world.routing import LiveRouter
    from world.wm3 import agent as AG
    from world.wm3 import drivers as DR
    from world.wm3 import rollout as RO
    from world.wm3 import state as S
    torch.set_num_threads(2)
    DR.set_menu_size(K, menu, share, batch)
    TR = _tr()
    tabs = TR.tables()
    env, _ = TR.load(env_tag, tabs)
    env.eval()
    ck = torch.load(TR.MODELS / f"{platform}.pt", map_location=TR.DEV)
    plat = AG.load_agent_state(AG.Agent(TR.make_model(tabs, **ck["arch"])).to(TR.DEV), ck["state"]).eval()
    g = S.CityGraph(cities.get(key))
    rules = RO.CityRules(g)
    _W.update(key=key, env=env, plat=plat, rules=rules, router=LiveRouter(g.net, g.traffic), p0=DR.DriverPolicy().eval())


def option_details(W, reqs, opt):
    """The requests on a menu as plain numbers: mode, zones, intercity, surge, rider quote, the
    expected trip km and minutes (zone tables) and the pickup minutes the driver is shown."""
    R, g = W.R, W.g
    h = int(W.t // 3600) % 24
    out = []
    for j, k in enumerate(reqs):
        if k < 0:
            continue
        q = W.req[int(k)]
        oz = int(g.zone[q["origin"]])
        dz = int(g.zone[q["dest"]]) if q["ext"] < 0 else int(g.zone[R.gw_pos[q["ext"]]])
        e = R.ext[q["ext"]] if q["ext"] >= 0 else None
        out.append({"j": j, "req": int(k), "mode": int(q["mode"]), "oz": oz, "dz": dz, "ext": int(q["ext"]),
                    "surge": float(q["surge"]), "quote": float(q["quote"]), "pick_min": float(opt[j, 1] * 10),
                    "km_in": max(R.zone_dist[W.daytype, h, oz, dz], 800) / 1000,
                    "outside_km": e.outside_km if e else 0.0,
                    "minutes": R.zone_tt[W.daytype, h, oz, dz] / 60 * R.speed[q["mode"]] + (e.outside_min if e else 0.0)})
    return out


def run_probe(task):
    from world.wm3 import agent as AG
    from world.wm3 import pay as PY
    try:
        W = AG.load_start(task["start"], _W["env"], _W["rules"], _W["router"])
        seed = task["seed"]
        rng = np.random.default_rng(seed)
        W.setup(_W["plat"], policy_seed=seed, mode="agent", match="agent")
        W.drivers, W.override_driver, W.trips = _W["p0"], {}, []
        removed = W.cut_supply(task["supply"], seed) if task["supply"] > 0 else []
        W.demand_mult = task.get("demand", 1.0)
        for _ in range(int(rng.integers(task["warm"]))):
            AG.step_many([W])
        from world.wm3 import drivers as DR
        while W.e % DR.BATCH:                         # batched menus: branch at an epoch that sends them
            AG.step_many([W])
        probe = AG.clone(W)
        AG.step_many([probe])
        mr = getattr(probe, "last_menu", None)
        if mr is None or not len(mr["drv"]) or mr["e"] != W.e:
            return None
        win = np.flatnonzero(mr["winnable"])
        if not len(win):
            return None
        i = int(rng.choice(win))
        d = int(mr["drv"][i])
        acts = np.r_[0, 1 + np.flatnonzero(mr["mask"][i])].astype(int)
        t0, km0 = W.t, float(W.km[d])
        ws = []
        for x in acts:
            Wx = AG.clone(W)
            Wx.override_driver = {W.e: {d: int(x)}}
            ws.append(Wx)
        AG.rollout_many(ws, task["H"], CHUNK[_W["key"]])
        recs = [PY.focal_record(Wx, d, t0, km0) for Wx in ws]
        pk = [sum(1 for x in Wx.log if x[1] == "pickup") for Wx in ws]
        idle = int((mr["mask"].any(1)).sum())
        return {"key": _W["key"], "start": str(task["start"]), "seed": seed, "supply": task["supply"], "demand": W.demand_mult, "menu": task["menu"], "removed": len(removed),
                "t": t0, "driver": d, "kappa": float(W.kappa[d]), "money": W.money, "acts": acts, "opt": mr["opt"][i],
                "own": mr["own"][i], "mask": mr["mask"][i], "options": option_details(W, mr["reqs"][i], mr["opt"][i]),
                "branches": recs, "pickups": np.array(pk), "n_menus": idle, "online": int((W.st != 0).sum()),
                "a_base": int(mr["a"][i])}
    finally:
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supply", type=float, default=0.0)
    ap.add_argument("--groups", type=int, nargs=2, default=[80, 60], help="branched choices: SF, Pune")
    ap.add_argument("--H", type=int, default=90, help="imagined 30-s steps after the choice")
    ap.add_argument("--K", type=int, default=6, help="menu size")
    ap.add_argument("--menu", default="exclusive", choices=["shared", "exclusive"], help="menu design (world/wm3/drivers.py)")
    ap.add_argument("--share", type=int, default=1, help="exclusive menus: each request on its N best drivers' menus")
    ap.add_argument("--batch", type=int, default=1, help="menus every N epochs (30 s each)")
    ap.add_argument("--demand", type=float, default=1.0, help="demand surge: new-request rates times this")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--platform", default="mbc_pre")
    ap.add_argument("--env", default="wm3_exo")
    ap.add_argument("--workers", type=int, nargs=2, default=[2, 2])
    ap.add_argument("--cities", nargs="+", default=["sf_hd", "pune_hd"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    from world import cities
    rng = np.random.default_rng(args.seed + int(args.supply * 1000))
    ctx = mp.get_context("spawn")
    nw = dict(zip(["sf_hd", "pune_hd"], args.workers))
    ng = dict(zip(["sf_hd", "pune_hd"], args.groups))
    pools, res = {}, {}
    t0 = time.time()
    for k in args.cities:
        starts = sorted((cities.get(k).data_dir / "wm3_rl" / "starts").glob("train_*.pkl"))
        tasks = [{"start": starts[rng.integers(len(starts))], "seed": int(rng.integers(1 << 30)), "supply": args.supply, "demand": args.demand, "menu": args.menu,
                  "H": args.H, "warm": args.warm} for _ in range(ng[k])]
        pools[k] = ctx.Pool(nw[k], initializer=worker_init, initargs=(k, args.env, args.platform, args.K, args.menu, args.share, args.batch))
        res[k] = pools[k].imap_unordered(run_probe, tasks)
    for k in args.cities:
        out = []
        for n, r in enumerate(res[k], 1):
            if r is not None:
                out.append(r)
            print(f"  {k}: {n}/{ng[k]} choices branched, {len(out)} usable, {(time.time() - t0) / 60:.1f} min", flush=True)
        d = cities.get(k).data_dir / "wm3_rl" / "market_probe"
        d.mkdir(parents=True, exist_ok=True)
        tag = args.menu + (f"{args.share}" if args.share > 1 else "") + (f"b{args.batch}" if args.batch > 1 else "")
        f = d / (f"probe_{tag}_s{args.supply:.1f}" + (f"_d{args.demand:.1f}" if args.demand != 1 else "") + ".pkl")
        f.write_bytes(pickle.dumps(out))
        print(f"wrote {f} ({len(out)} records)", flush=True)
        pools[k].close()


if __name__ == "__main__":
    main()
