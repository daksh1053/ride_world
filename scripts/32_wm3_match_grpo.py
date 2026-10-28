"""World model 3, stage 6b: GRPO on the agent's matching (which driver gets which rider), inside the
world model, starting from the imitation agent (scripts/31_wm3_match_bc.py).

    .venv/bin/python scripts/32_wm3_match_grpo.py --init-agent mbc_pre --tag mrl_pre
    .venv/bin/python scripts/32_wm3_match_grpo.py --init-agent mbc_scratch --tag mrl_scratch

The agent matches every 30-s epoch and repositions every 5 min (world/wm3/agent.py). One GRPO
group (the 'prompt') is ONE pending request at one moment: copies of the imagined city that differ
only in that request's choice -- hold, and up to G-1 different drivers drawn from the agent's
distribution for it (given the drivers the requests before it took). Every other request and
driver decides by the agent with the same random numbers in every copy; later epochs are the
current agent's. Reward after H steps = riders picked up + drivers' proportional-fairness welfare
gain (agent.welfare). Advantage = reward - the group's mean, divided by the s.d. of these over the
city's groups this iteration (+ 0.02).

Moments: a training start state (scripts/25_wm3_rl_starts.py) advanced 0..--warm-1 steps by the
current agent.

Update: PPO-clipped surrogate on the group's choices, + beta * KL(agent || starting agent) on the
request's choice, + beta_repo * KL(starting agent || agent) on stage-5 repositioning states, +
aux * the world model's own prediction loss. Every --eval-every iterations: 60 imagined minutes from
held-out validation states with (matching, repositioning) = (rule, none), (rule, rule),
(rule, agent) [the stage-5 agent's setting] and (agent, agent).

Writes data/wm3_models/<tag>.pt, outputs/wm3/stage6/train_<tag>.csv and eval_<tag>.csv.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import pickle
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
OUT_NAME = "stage6"
CHUNK = {"sf_hd": 8, "pune_hd": 2}
LBATCH = {"sf_hd": 4, "pune_hd": 2}
ARMS = {"none": ("rule", "none"), "rule": ("rule", "rule"), "repo_agent": ("rule", "agent"), "agent": ("agent", "agent")}


def _tr():
    spec = importlib.util.spec_from_file_location("tr3", ROOT / "scripts" / "18_wm3_train.py")
    TR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TR)
    return TR


_W = {}


def worker_init(key, env_tag):
    from world import cities
    from world.routing import LiveRouter
    from world.wm3 import rollout as RO
    from world.wm3 import state as S
    torch.set_num_threads(2)
    TR = _tr()
    tabs = TR.tables()
    env, _ = TR.load(env_tag, tabs)
    env.eval()
    g = S.CityGraph(cities.get(key))
    rules = RO.CityRules(g)
    _W.update(key=key, TR=TR, tabs=tabs, env=env, g=g, rules=rules, router=LiveRouter(g.net, g.traffic), agent=None, version=-1)


def _agent(path, version):
    from world.wm3 import agent as AG
    from world.wm3 import model as WM
    if _W["version"] != version:
        ck = torch.load(path, map_location=_W["TR"].DEV)
        if _W["agent"] is None:
            _W["agent"] = AG.Agent(WM.WorldModel3(_W["tabs"], **ck["arch"])).to(_W["TR"].DEV).eval()
        AG.load_agent_state(_W["agent"], ck["state"])
        _W["version"] = version
    return _W["agent"]


def _load(start):
    from world.wm3 import agent as AG
    return AG.load_start(start, _W["env"], _W["rules"], _W["router"])


def reward_of(m):
    return m["pickups"] + m["welfare"]


def run_group(task):
    """GRPO group for ONE request at one moment (see the module docstring)."""
    from world.wm3 import agent as AG
    from world.wm3 import rollout as RO
    try:
        agent = _agent(task["path"], task["version"])
        W = _load(task["start"])
        seed = task["seed"]
        rng = np.random.default_rng(seed)
        W.setup(agent, policy_seed=seed, mode="agent", match="agent")
        for _ in range(int(rng.integers(task["warm"]))):
            AG.step_many([W])
        probe = AG.clone(W)
        AG.step_many([probe])
        rec = probe.last_match
        if rec is None:
            return None
        mi, lg, a = rec["mi"], rec["lg"], rec["a"]
        m = len(mi["drv"])
        U = AG.used_before(a, m)
        row = lambda j: np.where(np.r_[False, U[j]], -np.inf, lg[j])  # noqa: E731
        ok = [j for j in range(len(a)) if np.isfinite(row(j)[1:]).any()]
        if not ok:
            return None
        j = int(rng.choice(ok))
        l = row(j)
        p = np.exp(l - l.max())
        p /= p.sum()
        q = p[1:].copy()
        nz = int((q > 0).sum())
        drv = 1 + rng.choice(m, min(task["G"] - 1, nz), replace=False, p=q / q.sum())
        acts = np.concatenate([[0], drv]).astype(int)
        k = int(mi["req"][j])
        ws = []
        for x in acts:
            Wx = AG.clone(W)
            Wx.override_match = {W.e: {k: int(x)}}
            ws.append(Wx)
        log0, inc0 = len(W.log), W.income.copy()
        AG.rollout_many(ws, task["H"], CHUNK[_W["key"]])
        mets = [AG.outcome(Wx, log0, inc0) for Wx in ws]
        r = np.array([reward_of(mm) for mm in mets], float)
        sub = {kk: (v[[j]] if kk in ("req", "rpos", "feas", "eta") else v) for kk, v in mi.items()}
        return {"key": _W["key"], "version": task["version"], "start": str(task["start"]), "snap": rec["snap"], "mi": sub,
                "U": U[[j]], "p_old": p, "acts": acts, "rewards": r, "metrics": mets, "n_req": len(a), "n_drv": m,
                "eta": np.r_[0.0, mi["eta"][j]][acts], "p_base_hold": float(p[0])}
    finally:
        torch.cuda.empty_cache()


def run_eval(task):
    from world.wm3 import agent as AG
    try:
        agent = _agent(task["path"], task["version"])
        W = _load(task["start"])
        log0, inc0 = len(W.log), W.income.copy()
        ws = []
        for name, (mt, rp) in ARMS.items():
            Wx = AG.clone(W).setup(agent if "agent" in (mt, rp) else None, policy_seed=task["seed"], mode=rp, match=mt)
            Wx.arm = name
            ws.append(Wx)
        AG.rollout_many(ws, task["H"], CHUNK[_W["key"]])
        out = []
        for Wx in ws:
            m = AG.outcome(Wx, log0, inc0)
            out.append({"key": _W["key"], "start": str(task["start"]), "mode": Wx.arm, **m, "reward": reward_of(m),
                        "held": Wx.n_held})
        return out
    finally:
        torch.cuda.empty_cache()


def _clipped(new, old, A, clip):
    ratio = torch.exp(new - old)
    return torch.minimum(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).sum()


def policy_terms(agent, ref, groups, dev, clip):
    from world.wm3 import agent as AG
    b, rpos, dpos, feas, eta = AG.collate_matches([g["snap"] for g in groups], [g["mi"] for g in groups], dev)
    lg = agent.match_logits(b, rpos, dpos, feas, eta).float()
    with torch.no_grad():
        lr = ref.match_logits(b, rpos, dpos, feas, eta).float()
    surr, kl = 0.0, 0.0
    for i, g in enumerate(groups):
        m = g["n_drv"]
        U = torch.from_numpy(g["U"]).to(dev)
        lp = AG.seq_logprobs(lg[i, :1, :1 + m], U)[0]
        lpr = AG.seq_logprobs(lr[i, :1, :1 + m], U)[0]
        a = torch.from_numpy(g["acts"]).long().to(dev)
        po = torch.log(torch.from_numpy(g["p_old"]).float().to(dev).clamp(min=1e-30))
        A = torch.from_numpy(g["adv"] / len(g["acts"])).float().to(dev)
        surr = surr + _clipped(lp.gather(-1, a), po.gather(-1, a), A, clip)
        ok = torch.isfinite(lp) & torch.isfinite(lpr)
        a_, r_ = lp.masked_fill(~ok, 0.0), lpr.masked_fill(~ok, 0.0)
        kl = kl + (a_.exp() * ok * (a_ - r_)).sum()
    return -surr / max(len(groups), 1), kl / max(len(groups), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init-agent", required=True)
    ap.add_argument("--env", default="wm3_exo")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--groups", type=int, nargs=2, default=[16, 6], help="groups per iteration: SF, Pune")
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--H", type=int, default=40, help="30-s steps after the decision")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-scalar", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--beta-repo", type=float, default=1.0)
    ap.add_argument("--aux", type=float, default=1.0)
    ap.add_argument("--workers", type=int, nargs=2, default=[2, 2])
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-starts", type=int, default=12)
    ap.add_argument("--cities", nargs="+", default=["sf_hd", "pune_hd"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from world import cities
    from world.wm3 import agent as AG
    from world.wm3 import data as WD
    from world.wm3 import model as WM
    TR = _tr()
    spec = importlib.util.spec_from_file_location("mbc", ROOT / "scripts" / "31_wm3_match_bc.py")
    MBC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MBC)
    dev = TR.DEV
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tabs = TR.tables()
    ck = torch.load(TR.MODELS / f"{args.init_agent}.pt", map_location=dev)
    arch = ck["arch"]
    agent = AG.load_agent_state(AG.Agent(TR.make_model(tabs, **arch)).to(dev), ck["state"])
    ref = copy.deepcopy(agent).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    scal = [p for n, p in agent.named_parameters() if n in ("w_eta", "b_hold")]
    head = [p for n, p in agent.named_parameters() if n.startswith(AG.MATCH_KEYS) and n not in ("w_eta", "b_hold")]
    rest = [p for n, p in agent.named_parameters() if not n.startswith(AG.MATCH_KEYS)]
    opt = torch.optim.AdamW([{"params": scal, "lr": args.lr_scalar}, {"params": head, "lr": args.lr},
                             {"params": rest, "lr": args.lr_backbone}], weight_decay=0.0)
    pools = {k: WD.DayPool("train", n_days=2, swap=40, seed=args.seed + i, keys=[k]) for i, k in enumerate(WD.CITY_KEYS)}
    val = TR.fixed_snaps("val", 24, seed=1)
    repo = {}
    for k in args.cities:
        r = pickle.loads((cities.get(k).data_dir / "wm3_rl" / "bc_train.pkl").read_bytes())
        repo[k] = [r[i] for i in rng.choice(len(r), min(800, len(r)), replace=False)]
        del r

    starts = {k: sorted((cities.get(k).data_dir / "wm3_rl" / "starts").glob("train_*.pkl")) for k in args.cities}
    vstarts = {k: sorted((cities.get(k).data_dir / "wm3_rl" / "starts").glob("val_*.pkl"))[: args.eval_starts] for k in args.cities}
    out = cities.OUTPUTS / "wm3" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    wpath = Path("/dev/shm") / f"agent_{args.tag}.pt"
    ctx = mp.get_context("spawn")
    nw = dict(zip(["sf_hd", "pune_hd"], args.workers))
    pool = {k: ctx.Pool(nw[k], initializer=worker_init, initargs=(k, args.env)) for k in args.cities}
    print(f"== {args.tag}: init {args.init_agent}, beta {args.beta}, G {args.G} H {args.H}; workers {nw}", flush=True)

    def save_w(version):
        torch.save({"state": {k: v.detach().cpu() for k, v in agent.state_dict().items()}, "arch": arch}, wpath)

    def evaluate(it, version):
        tasks = [(k, {"path": str(wpath), "version": version, "start": s, "seed": 7, "H": 120}) for k in args.cities for s in vstarts[k]]
        res = {k: pool[k].map_async(run_eval, [t for kk, t in tasks if kk == k]) for k in args.cities}
        df = pd.DataFrame([r for k in args.cities for rr in res[k].get() for r in rr])
        df["iter"] = it
        wmv = TR.evaluate(agent.wm, val)
        agent.eval()
        torch.cuda.empty_cache()
        msg = []
        for k in args.cities:
            piv = {c: df[df.key == k].pivot_table(index="start", columns="mode", values=c) for c in ("reward", "pickups", "welfare", "moved", "held")}
            d = lambda c, a: float((piv[c][a] - piv[c]["none"]).mean())  # noqa: E731
            msg.append(f"{k}: reward minus none: agent {d('reward', 'agent'):+.2f} repo_agent {d('reward', 'repo_agent'):+.2f} rule {d('reward', 'rule'):+.2f} "
                       f"(pickups {d('pickups', 'agent'):+.1f}/{d('pickups', 'repo_agent'):+.1f}/{d('pickups', 'rule'):+.1f}, welfare "
                       f"{d('welfare', 'agent'):+.2f}/{d('welfare', 'repo_agent'):+.2f}/{d('welfare', 'rule'):+.2f}, held {piv['held']['agent'].mean():.0f})")
        nll = sum(wmv.values()) / 2
        print(f"  eval it {it}: " + "; ".join(msg) + f"; world-model val NLL {nll:.3f}", flush=True)
        df["wm_val_nll"] = nll
        return df

    log, evals, t0, version = [], [], time.time(), 0
    save_w(version)
    evals.append(evaluate(0, version))
    pd.concat(evals).to_csv(out / f"eval_{args.tag}.csv", index=False)
    for it in range(1, args.iters + 1):
        ng = dict(zip(["sf_hd", "pune_hd"], args.groups))
        tasks = [(k, {"path": str(wpath), "version": version, "start": starts[k][rng.integers(len(starts[k]))],
                      "seed": int(rng.integers(1 << 30)), "G": args.G, "H": args.H, "warm": args.warm})
                 for k in args.cities for _ in range(ng[k])]
        res = {k: pool[k].map_async(run_group, [t for kk, t in tasks if kk == k]) for k in args.cities}
        groups = [g for k in args.cities for g in res[k].get() if g is not None]
        for k in args.cities:
            gs = [g for g in groups if g["key"] == k]
            if not gs:
                continue
            raw = [g["rewards"] - g["rewards"].mean() for g in gs]
            sd = float(np.concatenate(raw).std())
            for g, a in zip(gs, raw):
                g["adv"] = a / (sd + 0.02)
        stats = {"surr": 0.0, "kl": 0.0, "repo_kl": 0.0, "aux": 0.0}
        agent.train()
        nb = 0
        for ep in range(args.epochs):
            for k in args.cities:
                gs = [g for g in groups if g["key"] == k]
                for i in range(0, len(gs), LBATCH[k]):
                    surr, kl = policy_terms(agent, ref, gs[i: i + LBATCH[k]], dev, args.clip)
                    rb = [repo[k][j] for j in rng.choice(len(repo[k]), LBATCH[k], replace=False)]
                    rkl = MBC.repo_kl(agent, ref, rb, dev)
                    day, si = pools[k].sample()
                    bb = WD.collate([WD.snapshot(day, si)], dev)
                    with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
                        o = agent.wm(bb)
                    o = {kk: (v.float() if torch.is_tensor(v) else v) for kk, v in o.items()}
                    aux = sum(v for v in WM.losses(agent.wm, bb, o).values() if torch.is_tensor(v) and torch.isfinite(v))
                    loss = surr + args.beta * kl + args.beta_repo * rkl + args.aux * aux
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    opt.step()
                    nb += 1
                    stats["surr"] += float(surr.detach()); stats["kl"] += float(kl.detach())
                    stats["repo_kl"] += float(rkl.detach()); stats["aux"] += float(aux)
        torch.cuda.empty_cache()
        agent.eval()
        version += 1
        save_w(version)
        row = {"iter": it, "min": (time.time() - t0) / 60, "groups": len(groups), **{k: v / max(nb, 1) for k, v in stats.items()},
               "w_eta": float(agent.w_eta), "b_hold": float(agent.b_hold)}
        for k in args.cities:
            gs = [g for g in groups if g["key"] == k]
            if gs:
                row[f"{k}/reward"] = float(np.mean([g["rewards"].mean() for g in gs]))
                row[f"{k}/spread"] = float(np.mean([g["rewards"].std() for g in gs]))
                row[f"{k}/hold_gain"] = float(np.mean([g["rewards"][0] - g["rewards"][1:].mean() for g in gs if len(g["acts"]) > 1]))
                row[f"{k}/best_gain"] = float(np.mean([g["rewards"].max() - g["rewards"].mean() for g in gs]))
                row[f"{k}/p_hold"] = float(np.mean([g["p_base_hold"] for g in gs]))
                row[f"{k}/eta_corr"] = float(np.nanmean([np.corrcoef(g["eta"][1:], g["rewards"][1:])[0, 1] if len(g["acts"]) > 2 and np.std(g["eta"][1:]) > 0
                                                         and np.std(g["rewards"][1:]) > 0 else np.nan for g in gs]))
        log.append(row)
        print(f"  it {it:4d} {row['min']:.1f} min  " + "  ".join(f"{k} {v:.3f}" for k, v in row.items() if k not in ("iter", "min")), flush=True)
        pd.DataFrame(log).to_csv(out / f"train_{args.tag}.csv", index=False)
        torch.save({"state": agent.state_dict(), "arch": arch, "iter": it, "init": args.init_agent, "args": vars(args)}, TR.MODELS / f"{args.tag}.pt")
        if it % args.eval_every == 0 or it == args.iters:
            evals.append(evaluate(it, version))
            pd.concat(evals).to_csv(out / f"eval_{args.tag}.csv", index=False)
    for p in pool.values():
        p.close()


if __name__ == "__main__":
    main()
