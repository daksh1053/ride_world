"""World model 3, stage 7b: richer driver agents (world/wm3/drivers2.py) trained with GRPO inside the
world model, in the market chosen by scripts/41_wm3_market_tune.py (outputs/wm3/stage7b/market.json:
pay scheme, availability, exclusive menus).

    .venv/bin/python scripts/42_wm3_driver2_grpo.py --tag drv2

The platform (stage-6 agent mbc_pre, frozen) scores every allowed (request, idle driver) pair each
30-s epoch; each request goes on the menu of its best-scored driver (up to 6 per menu, menus sent every
`batch` epochs as market.json says) with the pay
the driver will get; it also repositions idle drivers every 5 min. Each driver waits, takes an
option or drives to one of 3 nearby zones of its choice; idle drivers without a menu decide every
5 min whether to wait or move. All drivers share one policy (each with its own type: income
target, shift length, home pull, cost per km).

One GRPO group is ONE driver's decision at one moment: copies of the imagined city that differ only
in that driver's action (every allowed one); everyone else uses the current policy with the same
random numbers. Reward: that driver's utility over the next H steps (drivers2.driver_utility), in
median fares. Advantage = reward - group mean, over the s.d. of the city's groups of the iteration.
Update: PPO-clipped surrogate + beta * KL(policy || starting policy).

Every --eval-every iterations, 60 imagined minutes from held-out states under:
    asap      every driver takes the platform's first option at once, never moves (drivers2.asap_policy)
    agents    every driver uses the current policy
    mixed     25% of the drivers use the current policy, the rest ASAP
Measured: pickups, rider wait, drivers' utility and net pay per online hour (all; and the 25% in
'mixed' against the same drivers under 'asap': the gain of one driver leaving ASAP).

Writes data/wm3_models/<tag>.pt, outputs/wm3/stage7b/train_<tag>.csv, eval_<tag>.csv.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = "stage7b"
CHUNK = {"sf_hd": 8, "pune_hd": 4}
if os.environ.get("DRV2_CHUNK"):          # worlds per world-model batch, "SF,Pune" (smaller saves GPU memory)
    CHUNK = dict(zip(["sf_hd", "pune_hd"], map(int, os.environ["DRV2_CHUNK"].split(","))))
ARMS = ("asap", "agents", "mixed")
MIX_SHARE = 0.25
_W = {}


def _tr():
    spec = importlib.util.spec_from_file_location("tr3", ROOT / "scripts" / "18_wm3_train.py")
    TR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TR)
    return TR


def market_config(key):
    """The market of stage 7b for a city (outputs/wm3/stage7b/market.json) -> (scheme, supply, K, share, batch)."""
    from world import cities
    from world.wm3 import pay as PY
    m = json.loads((cities.OUTPUTS / "wm3" / OUT_NAME / "market.json").read_text())
    c = m[key]
    return PY.PayScheme(**c["scheme"]), float(c["supply_cut"]), int(m.get("K", 6)), int(c.get("share", 1)), int(c.get("batch", 1))


def market_demand(key):
    """The demand surge of the market (new-request rates times this; 1 = none)."""
    from world import cities
    m = json.loads((cities.OUTPUTS / "wm3" / OUT_NAME / "market.json").read_text())
    return float(m[key].get("demand", 1.0))


def worker_init(key, env_tag, platform, start_policy, arch):
    from world import cities
    from world.routing import LiveRouter
    from world.wm3 import agent as AG
    from world.wm3 import drivers2 as D2
    from world.wm3 import rollout as RO
    from world.wm3 import state as S
    torch.set_num_threads(2)
    scheme, supply, K, share, batch = market_config(key)
    D2.configure(key, cities.get(key).market, scheme, K, share, batch)
    TR = _tr()
    tabs = TR.tables()
    env, _ = TR.load(env_tag, tabs)
    env.eval()
    ck = torch.load(TR.MODELS / f"{platform}.pt", map_location=TR.DEV)
    plat = AG.load_agent_state(AG.Agent(TR.make_model(tabs, **ck["arch"])).to(TR.DEV), ck["state"]).eval()
    g = S.CityGraph(cities.get(key))
    rules = RO.CityRules(g)
    p0 = D2.DriverPolicy2(**arch)
    p0.load_state_dict(torch.load(start_policy, map_location="cpu"))
    _W.update(key=key, env=env, plat=plat, rules=rules, router=LiveRouter(g.net, g.traffic), supply=supply, demand=market_demand(key),
              pol=D2.DriverPolicy2(**arch).eval(), p0=p0.eval(), asap=D2.asap_policy(), version=-1)


def _policy(path, version):
    if _W["version"] != version:
        _W["pol"].load_state_dict(torch.load(path, map_location="cpu"))
        _W["version"] = version
    return _W["pol"]


def setup(W, drivers, seed):
    W.setup(_W["plat"], policy_seed=seed, mode="agent", match="agent")
    W.driver_v, W.drivers, W.override_driver, W.trips = 2, drivers, {}, []
    W.n_declined, W.n_self_moves = 0, 0
    W.demand_mult = _W["demand"]
    if _W["supply"] > 0 and not getattr(W, "banned", None):
        W.cut_supply(_W["supply"], seed)
    return W


def run_group(task):
    from world.wm3 import agent as AG
    from world.wm3 import drivers2 as D2
    try:
        pol = _policy(task["path"], task["version"])
        W = AG.load_start(task["start"], _W["env"], _W["rules"], _W["router"])
        seed = task["seed"]
        rng = np.random.default_rng(seed)
        setup(W, pol, seed)
        for _ in range(int(rng.integers(task["warm"]))):
            AG.step_many([W])
        # a free epoch now and then, so that waiting / moving without a menu is trained too
        if rng.random() < task["p_free"]:
            while W.e % D2.FREE_EVERY != D2.FREE_PHASE:
                AG.step_many([W])
        else:                                          # batched menus: branch at an epoch that sends them
            while W.e % D2.D1.BATCH:
                AG.step_many([W])
        probe = AG.clone(W)
        AG.step_many([probe])
        mr = getattr(probe, "last_menu", None)
        if mr is None or not len(mr["drv"]) or mr["e"] != W.e:
            return None
        ok = np.isfinite(mr["lg"])
        n_act = ok.sum(1)
        cand = np.flatnonzero(n_act >= 2)
        if not len(cand):
            return None
        with_menu = cand[mr["om"][cand].any(1)]
        pool = with_menu if len(with_menu) and rng.random() < 0.75 else cand
        i = int(rng.choice(pool))
        d = int(mr["drv"][i])
        acts = np.flatnonzero(ok[i]).astype(int)
        t0, km0, inc0 = W.t, float(W.km[d]), float(W.income[d])
        ws = []
        for x in acts:
            Wx = AG.clone(W)
            Wx.override_driver = {W.e: {d: int(x)}}
            ws.append(Wx)
        AG.rollout_many(ws, task["H"], CHUNK[_W["key"]])
        u = np.array([D2.driver_utility(Wx, d, t0, km0, inc0) for Wx in ws])
        lg = mr["lg"][i]
        p = np.exp(lg - lg[np.isfinite(lg)].max())
        p = np.nan_to_num(p / p.sum())
        pk = [sum(1 for x in Wx.log if x[1] == "pickup") for Wx in ws]
        return {"key": _W["key"], "opt": mr["opt"][i], "mov": mr["mov"][i], "own": mr["own"][i], "om": mr["om"][i],
                "mm": mr["mm"][i], "p_old": p, "acts": acts, "rewards": u[:, 0] / W.money, "value": u[:, 1] / W.money,
                "pickups": np.array(pk), "has_menu": bool(mr["om"][i].any()), "n_opt": int(mr["om"][i].sum())}
    finally:
        torch.cuda.empty_cache()


def utilities(W, t0, km0, inc0):
    """Every driver's utility and net pay since t0 (money), vectorised over drivers."""
    from world.wm3 import drivers2 as D2
    from world.wm3 import pay as PY
    T = D2.types_of(W)
    tr = pd.DataFrame([x for x in W.trips if x["t"] > t0])
    v = np.zeros(W.nD)
    out_km = np.zeros(W.nD)
    if len(tr):
        p = PY.trips_pay(D2.CFG["scheme"], D2.CFG["key"], D2.CFG["market"], tr)
        np.add.at(v, tr.driver.to_numpy(), p)
        np.add.at(out_km, tr.driver.to_numpy(), 2 * tr.outside_km.to_numpy())
    km = W.km - km0 + out_km
    for d in np.flatnonzero(np.isin(W.st, [3, 4, 6])):          # PICKUP, OCC, OUTSIDE: the ride in progress
        rec = PY.focal_record(W, int(d), t0, km0[d])
        c = rec["cur"]
        if c is not None:
            v[d] += c["frac"] * float(PY.trips_pay(D2.CFG["scheme"], D2.CFG["key"], D2.CFG["market"], pd.DataFrame([c]))[0])
            km[d] += c["frac"] * (c["pick_km"] + c["km_in"] + 2 * c["outside_km"]) - rec["km_cur_started"]
    net = v - W.kappa * km
    u = D2.utility(T, np.arange(W.nD), inc0 + net) - D2.utility(T, np.arange(W.nD), inc0)
    u -= np.array([D2.home_cost(W, d) if W.st[d] != 0 else 0.0 for d in range(W.nD)])
    return u, net


def run_eval(task):
    from world.wm3 import agent as AG
    from world.wm3 import drivers2 as D2
    from world.wm3 import rollout as RO
    try:
        pol = _policy(task["path"], task["version"])
        W = AG.load_start(task["start"], _W["env"], _W["rules"], _W["router"])
        seed = task["seed"]
        grp = (RO.keyed_uniform(seed, 0, "mix", np.arange(W.nD)) < MIX_SHARE).astype(int)
        ws = []
        for m in ARMS:
            drivers = {"asap": _W["asap"], "agents": pol, "mixed": D2.Mixed([_W["asap"], pol], grp)}[m]
            Wx = setup(AG.clone(W), drivers, seed)
            Wx.arm = m
            ws.append(Wx)
        t0 = ws[0].t
        log0 = len(ws[0].log)
        km0, inc0 = [Wx.km.copy() for Wx in ws], [Wx.income.copy() for Wx in ws]
        online = np.zeros((len(ws), W.nD))
        for _ in range(task["H"]):
            AG.step_many(ws)
            for j, Wx in enumerate(ws):
                online[j] += (Wx.st != RO.OFF) * 30 / 3600
        res, per = [], {}
        for j, Wx in enumerate(ws):
            m = AG.outcome(Wx, log0, Wx.income * 0 + inc0[j])
            u, net = utilities(Wx, t0, km0[j], inc0[j])
            hrs = online[j]
            act = hrs > 0.25
            per[Wx.arm] = (u, net, hrs)
            res.append({"key": _W["key"], "start": str(task["start"]), "mode": Wx.arm, "pickups": m["pickups"],
                        "wait_min": m["wait_min"], "cancels": m["cancels"], "expired": m["expired"],
                        "u_per_h": float(u[act].sum() / hrs[act].sum()) / Wx.money, "net_per_h": float(net[act].sum() / hrs[act].sum()) / Wx.money,
                        "net_per_h_p10": float(np.percentile(net[act] / hrs[act], 10)) / Wx.money,
                        "declined": Wx.n_declined, "self_moves": Wx.n_self_moves, "driver_hours": float(hrs.sum())})
        # the 25% who leave ASAP in 'mixed', against the same drivers when everyone is ASAP
        f = (grp == 1) & (per["asap"][2] > 0.25) & (per["mixed"][2] > 0.25)
        for r in res:
            if r["mode"] == "mixed":
                r["focal_u_gain_per_h"] = float((per["mixed"][0][f] - per["asap"][0][f]).sum() / per["asap"][2][f].sum()) / W.money
                r["focal_net_gain_per_h"] = float((per["mixed"][1][f] - per["asap"][1][f]).sum() / per["asap"][2][f].sum()) / W.money
                r["others_net_gain_per_h"] = float(((per["mixed"][1] - per["asap"][1])[(grp == 0) & (per["asap"][2] > 0.25)]).sum()
                                                   / per["asap"][2][(grp == 0) & (per["asap"][2] > 0.25)].sum()) / W.money
        return res
    finally:
        torch.cuda.empty_cache()


def _clipped(new, old, A, clip):
    ratio = torch.exp(new - old)
    return torch.minimum(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="drv2")
    ap.add_argument("--platform", default="mbc_pre", help="stage-6 agent (data/wm3_models/<name>.pt)")
    ap.add_argument("--env", default="wm3_exo")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--groups", type=int, nargs=2, default=[24, 16], help="groups per iteration: SF, Pune")
    ap.add_argument("--H", type=int, default=90)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--p-free", type=float, default=0.3, help="share of groups started at a free (no-menu) epoch")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--workers", type=int, nargs=2, default=[2, 2])
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-starts", type=int, default=12)
    ap.add_argument("--cities", nargs="+", default=["sf_hd", "pune_hd"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128, help="driver policy width")
    ap.add_argument("--layers", type=int, default=4, help="self-attention layers")
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()

    from world import cities
    from world.wm3 import drivers2 as D2
    TR = _tr()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    arch = {"h": args.hidden, "layers": args.layers, "heads": args.heads}
    pol = D2.DriverPolicy2(**arch)
    print(f"driver policy: {sum(p.numel() for p in pol.parameters()):,} weights, {arch}", flush=True)
    ref = copy.deepcopy(pol).eval()
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    out = cities.OUTPUTS / "wm3" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    wpath = Path("/dev/shm") / f"drv2_{args.tag}.pt"
    p0path = Path("/dev/shm") / f"drv2_{args.tag}_start.pt"
    torch.save(ref.state_dict(), p0path)
    starts = {k: sorted((cities.get(k).data_dir / "wm3_rl" / "starts").glob("train_*.pkl")) for k in args.cities}
    vstarts = {k: sorted((cities.get(k).data_dir / "wm3_rl" / "starts").glob("val_*.pkl"))[: args.eval_starts] for k in args.cities}
    ctx = mp.get_context("spawn")
    nw = dict(zip(["sf_hd", "pune_hd"], args.workers))
    pool = {k: ctx.Pool(nw[k], initializer=worker_init, initargs=(k, args.env, args.platform, str(p0path), arch)) for k in args.cities}
    print(f"== {args.tag}: platform {args.platform}, H {args.H}, groups {args.groups}, market "
          + json.dumps(json.loads((out / "market.json").read_text())), flush=True)

    def save_w():
        torch.save(pol.state_dict(), wpath)

    def evaluate(it, version):
        tasks = [(k, {"path": str(wpath), "version": version, "start": s, "seed": 7, "H": 120}) for k in args.cities for s in vstarts[k]]
        res = {k: pool[k].map_async(run_eval, [t for kk, t in tasks if kk == k]) for k in args.cities}
        df = pd.DataFrame([r for k in args.cities for rr in res[k].get() for r in rr])
        df["iter"] = it
        msg = []
        for k in args.cities:
            g = df[df.key == k].groupby("mode")[["pickups", "wait_min", "u_per_h", "net_per_h", "declined", "self_moves"]].mean()
            fg = df[(df.key == k) & (df["mode"] == "mixed")][["focal_u_gain_per_h", "focal_net_gain_per_h", "others_net_gain_per_h"]].mean()
            msg.append(f"{k}: " + "; ".join(f"{m} pickups {g.pickups[m]:.1f} wait {g.wait_min[m]:.2f} u/h {g.u_per_h[m]:.3f} net/h {g.net_per_h[m]:.3f} "
                                            f"declined {g.declined[m]:.0f} moves {g.self_moves[m]:.0f}" for m in ARMS)
                       + f"; 25% leave ASAP: their u/h {fg.focal_u_gain_per_h:+.3f} net/h {fg.focal_net_gain_per_h:+.3f}, others net/h {fg.others_net_gain_per_h:+.3f}")
        print(f"  eval it {it}: " + " | ".join(msg), flush=True)
        return df

    log, evals, t0, version = [], [], time.time(), 0
    save_w()
    evals.append(evaluate(0, version))
    pd.concat(evals).to_csv(out / f"eval_{args.tag}.csv", index=False)
    for it in range(1, args.iters + 1):
        ng = dict(zip(["sf_hd", "pune_hd"], args.groups))
        tasks = [(k, {"path": str(wpath), "version": version, "start": starts[k][rng.integers(len(starts[k]))],
                      "seed": int(rng.integers(1 << 30)), "H": args.H, "warm": args.warm, "p_free": args.p_free})
                 for k in args.cities for _ in range(ng[k])]
        res = {k: pool[k].map_async(run_group, [t for kk, t in tasks if kk == k]) for k in args.cities}
        groups = [g for k in args.cities for g in res[k].get() if g is not None]
        for k in args.cities:
            gs = [g for g in groups if g["key"] == k]
            if gs:
                raw = [g["rewards"] - g["rewards"].mean() for g in gs]
                sd = float(np.concatenate(raw).std())
                for g, a in zip(gs, raw):
                    g["adv"] = a / (sd + 0.02)
        T = lambda x, dt=torch.float32: torch.tensor(np.stack(x), dtype=dt)  # noqa: E731
        O, Mv, S_ = T([g["opt"] for g in groups]), T([g["mov"] for g in groups]), T([g["own"] for g in groups])
        OM, MM = T([g["om"] for g in groups], torch.bool), T([g["mm"] for g in groups], torch.bool)
        stats = {"surr": 0.0, "kl": 0.0}
        for ep in range(args.epochs):
            lp = torch.log_softmax(pol(O, Mv, S_, OM, MM), -1)
            with torch.no_grad():
                lr_ = torch.log_softmax(ref(O, Mv, S_, OM, MM), -1)
            surr, kl = 0.0, 0.0
            for i, g in enumerate(groups):
                a = torch.from_numpy(g["acts"]).long()
                po = torch.log(torch.from_numpy(g["p_old"]).float().clamp(min=1e-30))
                A = torch.from_numpy(g["adv"] / len(g["acts"])).float()
                surr = surr + _clipped(lp[i].gather(-1, a), po.gather(-1, a), A, args.clip)
                ok = torch.isfinite(lp[i]) & torch.isfinite(lr_[i])
                a_, r_ = lp[i].masked_fill(~ok, 0.0), lr_[i].masked_fill(~ok, 0.0)
                kl = kl + (a_.exp() * ok * (a_ - r_)).sum()
            loss = (-surr + args.beta * kl) / max(len(groups), 1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            stats["surr"] += float(surr.detach()) / max(len(groups), 1) / args.epochs
            stats["kl"] += float(kl.detach()) / max(len(groups), 1) / args.epochs
        version += 1
        save_w()
        row = {"iter": it, "min": (time.time() - t0) / 60, "groups": len(groups), **stats,
               "b_wait": float(pol.b_wait), "b_move": float(pol.b_move), "w_rank": float(pol.w_rank)}
        for k in args.cities:
            gs = [g for g in groups if g["key"] == k]
            gm = [g for g in gs if g["has_menu"]]
            gf = [g for g in gs if not g["has_menu"]]
            if gs:
                row[f"{k}/reward"] = float(np.mean([g["rewards"].mean() for g in gs]))
                row[f"{k}/best_gain"] = float(np.mean([g["rewards"].max() - g["rewards"].mean() for g in gs]))
            if gm:
                # gains against the first option (ASAP) in menu groups: waiting, the average other option, the average move
                # (averages, not the best of several noisy branches, which would look positive by chance alone)
                ix1 = lambda g: int(np.flatnonzero(g["acts"] == 1)[0])  # noqa: E731
                row[f"{k}/wait_vs_first"] = float(np.mean([g["rewards"][0] - g["rewards"][ix1(g)] for g in gm]))
                row[f"{k}/other_vs_first"] = float(np.mean([np.mean([g["rewards"][j] for j, a in enumerate(g["acts"]) if 2 <= a <= D2.K] or [np.nan])
                                                            - g["rewards"][ix1(g)] for g in gm if g["n_opt"] >= 2] or [np.nan]))
                row[f"{k}/move_vs_first"] = float(np.mean([np.mean([g["rewards"][j] for j, a in enumerate(g["acts"]) if a > D2.K] or [np.nan])
                                                           - g["rewards"][ix1(g)] for g in gm] or [np.nan]))
                row[f"{k}/p_first"] = float(np.mean([g["p_old"][1] for g in gm]))
                row[f"{k}/p_wait_menu"] = float(np.mean([g["p_old"][0] for g in gm]))
            if gf:
                row[f"{k}/move_vs_wait_free"] = float(np.mean([g["rewards"][1:].mean() - g["rewards"][0] for g in gf]))
                row[f"{k}/p_move_free"] = float(np.mean([1 - g["p_old"][0] for g in gf]))
        log.append(row)
        print(f"  it {it:4d} {row['min']:.1f} min  " + "  ".join(f"{k} {v:.3f}" for k, v in row.items() if k not in ("iter", "min")), flush=True)
        pd.DataFrame(log).to_csv(out / f"train_{args.tag}.csv", index=False)
        torch.save({"state": pol.state_dict(), "arch": arch, "platform": args.platform, "iter": it, "args": vars(args),
                    "market": json.loads((out / "market.json").read_text())}, TR.MODELS / f"{args.tag}.pt")
        if it % args.eval_every == 0 or it == args.iters:
            evals.append(evaluate(it, version))
            pd.concat(evals).to_csv(out / f"eval_{args.tag}.csv", index=False)
    for p in pool.values():
        p.close()


if __name__ == "__main__":
    main()
