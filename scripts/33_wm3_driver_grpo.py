"""World model 3, stage 7: drivers as small agents maximising their own earnings, choosing from the
menus the platform's world-model agent gives them (world/wm3/drivers.py), trained with GRPO inside
the world model.

    .venv/bin/python scripts/33_wm3_driver_grpo.py --platform mrl_pre --tag drv_pre

The platform (a stage-6 agent, frozen) scores every allowed (request, idle driver) pair each 30-s
epoch and shows each idle driver its K best requests with the fare the driver will get; it also
repositions idle drivers every 5 min. Every driver takes one option or declines (one small policy
network shared by all drivers, each seeing only its own situation and menu). Taken rides are
accepted rides; the world model imagines everything else (riders, traffic, breaks, going offline,
following repositioning orders).

One GRPO group (the 'prompt') is ONE driver's menu at one moment (a driver who is the platform's
first choice for at least one request on its menu, so that its choice changes what it gets): copies of the imagined city that
differ only in that driver's choice (decline, and each option). Everyone else chooses with the same
random numbers in every copy. The driver's reward is its OWN net earnings over the next H steps
(fares after commission minus the cost of its km; a ride in progress at the end counts its fare),
in median fares. Advantage = reward - the group mean, divided by the s.d. over the city's groups of
the iteration (+ 0.02). Update: PPO-clipped surrogate + beta * KL(policy || starting policy).

Every --eval-every iterations, 60 imagined minutes from held-out validation states under:
    single        the platform's own matching, one offer per driver, drivers answer as the world
                  model predicts the simulator's people do (stage 6)
    menu_start    menus, drivers with the starting policy (take the platform's first option)
    menu_agents   menus, drivers with the current policy
Measured: riders picked up, drivers' net earnings per online hour, the welfare of stage 5, declines.

Writes data/wm3_models/<tag>.pt (driver policy), outputs/wm3/stage7/train_<tag>.csv, eval_<tag>.csv.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
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
OUT_NAME = "stage7"
CHUNK = {"sf_hd": 8, "pune_hd": 2}
ARMS = ("single", "menu_start", "menu_agents")
_W = {}


def _tr():
    spec = importlib.util.spec_from_file_location("tr3", ROOT / "scripts" / "18_wm3_train.py")
    TR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TR)
    return TR


def worker_init(key, env_tag, platform, start_policy):
    from world import cities
    from world.routing import LiveRouter
    from world.wm3 import agent as AG
    from world.wm3 import drivers as DR
    from world.wm3 import rollout as RO
    from world.wm3 import state as S
    torch.set_num_threads(2)
    TR = _tr()
    tabs = TR.tables()
    env, _ = TR.load(env_tag, tabs)
    env.eval()
    ck = torch.load(TR.MODELS / f"{platform}.pt", map_location=TR.DEV)
    plat = AG.load_agent_state(AG.Agent(TR.make_model(tabs, **ck["arch"])).to(TR.DEV), ck["state"]).eval()
    g = S.CityGraph(cities.get(key))
    rules = RO.CityRules(g)
    p0 = DR.DriverPolicy()
    p0.load_state_dict(torch.load(start_policy, map_location="cpu"))
    _W.update(key=key, TR=TR, env=env, plat=plat, g=g, rules=rules, router=LiveRouter(g.net, g.traffic),
              pol=DR.DriverPolicy().eval(), p0=p0.eval(), version=-1)


def _policy(path, version):
    if _W["version"] != version:
        _W["pol"].load_state_dict(torch.load(path, map_location="cpu"))
        _W["version"] = version
    return _W["pol"]


def _load(start):
    from world.wm3 import agent as AG
    return AG.load_start(start, _W["env"], _W["rules"], _W["router"])


def driver_value(W, d):
    """Driver d's net earnings so far in the imagined world (since setup), in money: income minus
    the cost of its km, plus the fare (after commission) of a ride it is on or driving to."""
    from world.wm3 import rollout as RO
    v = W.income[d] - W.kappa[d] * W.km[d]
    k = int(W.d_req[d])
    if W.st[d] in (RO.PICKUP, RO.OCC, RO.OUTSIDE, RO.OFFERED) and k in W.req:
        v += W.req[k]["quote"] * (1 - W.R.market.commission)
    return float(v)


def world(W, mode, seed):
    """Set up imagined world W under one of ARMS."""
    W.setup(_W["plat"], policy_seed=seed, mode="agent", match="agent")
    W.drivers = None if mode == "single" else (_W["p0"] if mode == "menu_start" else _W["pol"])
    W.override_driver = {}
    W.n_declined, W.n_lost = 0, 0
    return W


def run_group(task):
    from world.wm3 import agent as AG
    try:
        pol = _policy(task["path"], task["version"])
        W = _load(task["start"])
        seed = task["seed"]
        rng = np.random.default_rng(seed)
        world(W, "menu_agents", seed)
        W.drivers = pol
        for _ in range(int(rng.integers(task["warm"]))):
            AG.step_many([W])
        probe = AG.clone(W)
        AG.step_many([probe])
        mr = getattr(probe, "last_menu", None)
        if mr is None or not len(mr["drv"]) or mr["e"] != W.e:
            return None
        win = np.flatnonzero(mr["winnable"])          # drivers whose choice can change what they get
        if not len(win):
            return None
        i = int(rng.choice(win))
        d = int(mr["drv"][i])
        acts = np.r_[0, 1 + np.flatnonzero(mr["mask"][i])].astype(int)
        ws, v0 = [], None
        for x in acts:
            Wx = AG.clone(W)
            Wx.override_driver = {W.e: {d: int(x)}}
            ws.append(Wx)
        v0 = driver_value(W, d)
        AG.rollout_many(ws, task["H"], CHUNK[_W["key"]])
        r = np.array([(driver_value(Wx, d) - v0) / W.money for Wx in ws])
        lg = mr["lg"][i]
        p = np.exp(lg - lg[np.isfinite(lg)].max())
        p = np.nan_to_num(p / p.sum())
        pk = [sum(1 for x in Wx.log if x[1] == "pickup") for Wx in ws]
        return {"key": _W["key"], "opt": mr["opt"][i], "own": mr["own"][i], "mask": mr["mask"][i], "p_old": p,
                "acts": acts, "rewards": r, "fare": mr["fare"][i], "a_base": int(mr["a"][i]), "pickups": np.array(pk)}
    finally:
        torch.cuda.empty_cache()


def run_eval(task):
    from world.wm3 import agent as AG
    from world.wm3 import rollout as RO
    try:
        _policy(task["path"], task["version"])
        W = _load(task["start"])
        log0, inc0 = len(W.log), W.income.copy()
        ws = []
        for m in ARMS:
            Wx = world(AG.clone(W), m, task["seed"])
            Wx.arm = m
            ws.append(Wx)
        on0 = W.st != RO.OFF
        H = task["H"]
        online = np.zeros((len(ws), W.nD))
        for _ in range(H):
            AG.step_many(ws)
            for j, Wx in enumerate(ws):
                online[j] += (Wx.st != RO.OFF) * 30 / 3600
        out = []
        for j, Wx in enumerate(ws):
            m = AG.outcome(Wx, log0, inc0)
            net = Wx.income - inc0 - Wx.kappa * Wx.km
            hrs = online[j]
            act = hrs > 0.25
            out.append({"key": _W["key"], "start": str(task["start"]), "mode": Wx.arm, **m, "reward": m["pickups"] + m["welfare"],
                        "net_per_h": float(net[act].sum() / hrs[act].sum()) / Wx.money,
                        "net_per_h_p10": float(np.percentile(net[act] / hrs[act], 10)) / Wx.money,
                        "driver_hours": float(hrs.sum()), "declined": Wx.n_declined, "lost": Wx.n_lost, "on0": int(on0.sum())})
        return out
    finally:
        torch.cuda.empty_cache()


def _clipped(new, old, A, clip):
    ratio = torch.exp(new - old)
    return torch.minimum(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--platform", required=True, help="stage-6 agent (data/wm3_models/<name>.pt)")
    ap.add_argument("--env", default="wm3_exo")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--groups", type=int, nargs=2, default=[24, 8], help="groups per iteration: SF, Pune")
    ap.add_argument("--H", type=int, default=60)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--workers", type=int, nargs=2, default=[2, 2])
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-starts", type=int, default=12)
    ap.add_argument("--cities", nargs="+", default=["sf_hd", "pune_hd"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from world import cities
    from world.wm3 import drivers as DR
    TR = _tr()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pol = DR.DriverPolicy()
    ref = copy.deepcopy(pol).eval()
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    out = cities.OUTPUTS / "wm3" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    wpath = Path("/dev/shm") / f"drv_{args.tag}.pt"
    p0path = Path("/dev/shm") / f"drv_{args.tag}_start.pt"
    torch.save(ref.state_dict(), p0path)
    starts = {k: sorted((cities.get(k).data_dir / "wm3_rl" / "starts").glob("train_*.pkl")) for k in args.cities}
    vstarts = {k: sorted((cities.get(k).data_dir / "wm3_rl" / "starts").glob("val_*.pkl"))[: args.eval_starts] for k in args.cities}
    ctx = mp.get_context("spawn")
    nw = dict(zip(["sf_hd", "pune_hd"], args.workers))
    pool = {k: ctx.Pool(nw[k], initializer=worker_init, initargs=(k, args.env, args.platform, str(p0path))) for k in args.cities}
    print(f"== {args.tag}: platform {args.platform}, H {args.H}, groups {args.groups}", flush=True)

    def save_w():
        torch.save(pol.state_dict(), wpath)

    def evaluate(it, version):
        tasks = [(k, {"path": str(wpath), "version": version, "start": s, "seed": 7, "H": 120}) for k in args.cities for s in vstarts[k]]
        res = {k: pool[k].map_async(run_eval, [t for kk, t in tasks if kk == k]) for k in args.cities}
        df = pd.DataFrame([r for k in args.cities for rr in res[k].get() for r in rr])
        df["iter"] = it
        msg = []
        for k in args.cities:
            g = df[df.key == k].groupby("mode")[["pickups", "welfare", "reward", "net_per_h", "net_per_h_p10", "declined", "lost"]].mean()
            msg.append(f"{k}: " + "; ".join(f"{m} pickups {g.pickups[m]:.1f} net/h {g.net_per_h[m]:.3f} p10 {g.net_per_h_p10[m]:.3f} "
                                            f"welfare {g.welfare[m]:.2f} declined {g.declined[m]:.0f}" for m in ARMS))
        print(f"  eval it {it}: " + " | ".join(msg), flush=True)
        return df

    log, evals, t0, version = [], [], time.time(), 0
    save_w()
    evals.append(evaluate(0, version))
    pd.concat(evals).to_csv(out / f"eval_{args.tag}.csv", index=False)
    for it in range(1, args.iters + 1):
        ng = dict(zip(["sf_hd", "pune_hd"], args.groups))
        tasks = [(k, {"path": str(wpath), "version": version, "start": starts[k][rng.integers(len(starts[k]))],
                      "seed": int(rng.integers(1 << 30)), "H": args.H, "warm": args.warm})
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
        O, S_, M = T([g["opt"] for g in groups]), T([g["own"] for g in groups]), T([g["mask"] for g in groups], torch.bool)
        stats = {"surr": 0.0, "kl": 0.0}
        for ep in range(args.epochs):
            lp = torch.log_softmax(pol(O, S_, M), -1)
            with torch.no_grad():
                lr_ = torch.log_softmax(ref(O, S_, M), -1)
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
               "b_decline": float(pol.b_decline), "w_rank": float(pol.w_rank)}
        for k in args.cities:
            gs = [g for g in groups if g["key"] == k]
            if gs:
                row[f"{k}/reward"] = float(np.mean([g["rewards"].mean() for g in gs]))
                row[f"{k}/decline_gain"] = float(np.mean([g["rewards"][0] - g["rewards"][1:].mean() for g in gs if len(g["acts"]) > 1]))
                row[f"{k}/first_gain"] = float(np.mean([g["rewards"][1] - g["rewards"].mean() for g in gs if len(g["acts"]) > 1]))
                row[f"{k}/best_gain"] = float(np.mean([g["rewards"].max() - g["rewards"].mean() for g in gs]))
                row[f"{k}/p_decline"] = float(np.mean([g["p_old"][0] for g in gs]))
                row[f"{k}/p_first"] = float(np.mean([g["p_old"][1] for g in gs]))
        log.append(row)
        print(f"  it {it:4d} {row['min']:.1f} min  " + "  ".join(f"{k} {v:.3f}" for k, v in row.items() if k not in ("iter", "min")), flush=True)
        pd.DataFrame(log).to_csv(out / f"train_{args.tag}.csv", index=False)
        torch.save({"state": pol.state_dict(), "platform": args.platform, "iter": it, "args": vars(args)}, TR.MODELS / f"{args.tag}.pt")
        if it % args.eval_every == 0 or it == args.iters:
            evals.append(evaluate(it, version))
            pd.concat(evals).to_csv(out / f"eval_{args.tag}.csv", index=False)
    for p in pool.values():
        p.close()


if __name__ == "__main__":
    main()
