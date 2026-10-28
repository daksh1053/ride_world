"""World model 3, stage 5b: train the agent (world/wm3/agent.py) with GRPO inside the world model.

    .venv/bin/python scripts/26_wm3_grpo.py --init wm3_exo --tag rl_pre        # RL on the pretrained world model
    .venv/bin/python scripts/26_wm3_grpo.py --init none --tag rl_scratch       # the same network from random weights

Environment: the frozen world model wm3_exo imagining the city (world/wm3/rollout.py), started from
real states of training days (scripts/25_wm3_rl_starts.py). Matching is the platform's rule; the
agent decides every 5 minutes which idle drivers to send where.

One sample (a group): at a start state the agent's choice distribution is computed once and every
deciding driver's choice drawn (the base world). G-1 copies each change ONE driver's choice to another
draw from the agent. Each copy then runs H steps (later decisions by the current agent, with the same
random numbers in every copy) and its reward is the number of riders
picked up plus the drivers' proportional-fairness welfare gain (agent.welfare: sum over drivers of
log(1 + today's net earnings / median fare), net of driving costs). Because all copies share the world's random numbers (common random numbers, rollout.py),
reward(copy) - reward(base) is that one driver's effect: group-relative credit per decision.
Advantage = that difference / (its s.d. over the iteration's groups of the city + 0.05).
(A first version varied 16 drivers at once in every copy, as plain GRPO; with 300-700 deciding
drivers in Pune the per-driver credit was lost in the others' noise: outputs/wm3/stage5/v0_group16/.)

Update: pairwise PPO-clipped policy gradient (the better of the two choices gains probability), plus beta * KL(agent || starting agent) on the
same choices, plus aux * the world model's own prediction loss on training snapshots (so the agent
stays a world model). Every --eval-every iterations: 60 imagined minutes from held-out validation
states with no repositioning, the platform's rule, and the agent.

Writes data/wm3_models/<tag>.pt (agent state), outputs/wm3/stage5/train_<tag>.csv and eval_<tag>.csv.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
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
OUT_NAME = "stage5"


def _tr():
    spec = importlib.util.spec_from_file_location("tr3", ROOT / "scripts" / "18_wm3_train.py")
    TR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TR)
    return TR


# ---------------------------------------------------------------------- workers (GPU, one city each)
_W = {}
CHUNK = {"sf_hd": 8, "pune_hd": 2}
LBATCH = {"sf_hd": 4, "pune_hd": 2}            # groups per learner step (GPU memory)


def worker_init(key, env_tag):
    from world import cities
    from world.wm3 import rollout as RO
    from world.wm3 import state as S
    from world.routing import LiveRouter
    torch.set_num_threads(2)
    TR = _tr()
    tabs = TR.tables()
    env, _ = TR.load(env_tag, tabs)
    env.eval()
    g = S.CityGraph(cities.get(key))
    rules = RO.CityRules(g)
    _W.update(key=key, TR=TR, tabs=tabs, env=env, g=g, rules=rules, router=LiveRouter(g.net, g.traffic),
              agent=None, version=-1)


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
    """Riders picked up plus the drivers' proportional-fairness welfare gain (agent.welfare)."""
    return m["pickups"] + m["welfare"]


def run_group(task):
    """One group from one start state: a base world (every deciding driver's choice drawn from the
    agent) and G-1 copies that each change ONE driver's choice to another draw from the agent
    (excluding the base choice). All copies run in lockstep with the same random numbers, so
    reward(copy) - reward(base) is that driver's effect."""
    from world.wm3 import agent as AG
    from world.wm3 import rollout as RO
    try:
        agent = _agent(task["path"], task["version"])
        W = _load(task["start"])
        base_seed = task["seed"]
        W.setup(agent, policy_seed=base_seed, mode="agent")
        probe = AG.clone(W)
        rec = AG.step_many([probe])[0]
        dec, p = rec["dec"], rec["p"]
        n = len(dec["drv"])
        if n == 0:
            return None
        rng = np.random.default_rng(base_seed)
        a_base = AG.choose(p, base_seed, W.e, dec["drv"])
        devs, alts, ws = [], [], []
        Wb = AG.clone(W)
        Wb.override = {W.e: a_base}
        ws.append(Wb)
        for j, i in enumerate(rng.permutation(n)):
            if len(devs) == task["G"] - 1:
                break
            q = p[i].copy()
            q[a_base[i]] = 0.0
            if q.sum() <= 1e-9:
                continue
            b = int(RO.pick(q / q.sum(), RO.keyed_uniform(base_seed, W.e + j + 1, "policy", [dec["drv"][i]]))[0])
            a = a_base.copy()
            a[i] = b
            Wj = AG.clone(W)
            Wj.override = {W.e: a}
            ws.append(Wj)
            devs.append(int(i)); alts.append(b)
        if not devs:
            return None
        log0, inc0 = len(W.log), W.income.copy()
        AG.rollout_many(ws, task["H"], CHUNK[_W["key"]])
        mets = [AG.outcome(Wj, log0, inc0) for Wj in ws]
        r = np.array([reward_of(m) for m in mets], float)
        devs = np.array(devs)
        return {"key": _W["key"], "version": task["version"], "start": str(task["start"]), "snap": rec["snap"],
                "dec": {k: v[devs] for k, v in dec.items()}, "p_old": p[devs], "a_base": a_base[devs], "a_alt": np.array(alts),
                "delta": r[1:] - r[0], "rewards": r, "metrics": mets, "n_deciding": n}
    finally:
        torch.cuda.empty_cache()


def run_group_driver(task):
    """GRPO group for ONE driver at one start state (the 'prompt'): copies that differ only in that
    driver's choice -- stay, and up to G-1 different destinations drawn from the agent's move
    distribution. Every other deciding driver takes its own draw (the same in every copy), later
    decisions are the current agent's, all copies share the random numbers. Rewards after H steps."""
    from world.wm3 import agent as AG
    try:
        agent = _agent(task["path"], task["version"])
        W = _load(task["start"])
        seed = task["seed"]
        W.setup(agent, policy_seed=seed, mode="agent")
        probe = AG.clone(W)
        rec = AG.step_many([probe])[0]
        dec, p = rec["dec"], rec["p"]
        n = len(dec["drv"])
        rng = np.random.default_rng(seed)
        ok = np.where(np.nan_to_num(p[:, 1:]).sum(1) > 1e-9)[0] if n else []
        if len(ok) == 0:
            return None
        i = int(rng.choice(ok))
        a_base = AG.choose(p, seed, W.e, dec["drv"])
        q = np.nan_to_num(p[i, 1:])
        nz = int((q > 0).sum())
        moves = 1 + rng.choice(len(q), min(task["G"] - 1, nz), replace=False, p=q / q.sum())
        acts = np.concatenate([[0], moves]).astype(int)
        ws = []
        for b in acts:
            a = a_base.copy()
            a[i] = b
            Wj = AG.clone(W)
            Wj.override = {W.e: a}
            ws.append(Wj)
        log0, inc0 = len(W.log), W.income.copy()
        AG.rollout_many(ws, task["H"], CHUNK[_W["key"]])
        mets = [AG.outcome(Wj, log0, inc0) for Wj in ws]
        r = np.array([reward_of(m) for m in mets], float)
        return {"key": _W["key"], "version": task["version"], "start": str(task["start"]), "snap": rec["snap"],
                "dec": {k: v[[i]] for k, v in dec.items()}, "p_old": p[[i]], "acts": acts, "rewards": r,
                "metrics": mets, "n_deciding": n}
    finally:
        torch.cuda.empty_cache()


def driver_advantages(gs):
    """Per group (one driver, Q = reward of stay and of each sampled destination):
    stay vs move: V = pi(stay) Q(stay) + pi(move) mean Q(moves); advantages Q(stay) - V and
        mean Q(moves) - V, weighted by pi(stay) and pi(move) (the exact gradient of the expected value);
    destination given a move: advantage Q(z) - mean Q(moves), weight 1/n_moves -- NOT scaled by
        pi(move), so the agent keeps learning where to go even when it rarely moves.
    Both normalised by the s.d. of the raw advantages over the city's groups this iteration."""
    for g in gs:
        ps = float(g["p_old"][0, 0])
        r = g["rewards"]
        qm = r[1:].mean()
        v = ps * r[0] + (1 - ps) * qm
        g["bin_adv"] = np.array([r[0] - v, qm - v])
        g["bin_w"] = np.array([ps, 1 - ps])
        g["dst_adv"] = r[1:] - qm
    allv = np.concatenate([np.concatenate([g["bin_adv"], g["dst_adv"]]) for g in gs]) if gs else np.ones(1)
    sd = float(allv.std())
    for g in gs:
        g["bin_adv"] = g["bin_adv"] / (sd + 0.02)
        g["dst_adv"] = g["dst_adv"] / (sd + 0.02)


def _clipped(new, old, A, clip):
    ratio = torch.exp(new - old)
    return torch.minimum(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).sum()


def policy_terms_driver(agent, ref, groups, dev, clip):
    from world.wm3 import agent as AG
    b, pos, cand, tt = AG.collate_decisions([g["snap"] for g in groups], [g["dec"] for g in groups], dev)
    lg = agent.logits(b, pos, cand, tt).float()
    with torch.no_grad():
        lr = ref.logits(b, pos, cand, tt).float()
    lp, lpr = torch.log_softmax(lg, -1), torch.log_softmax(lr, -1)
    surr, kl = 0.0, 0.0
    for i, g in enumerate(groups):
        l = lp[i, 0]
        po = torch.log(torch.from_numpy(g["p_old"][0]).float().to(dev).clamp(min=1e-30))
        # stay vs move
        lm_new = torch.logsumexp(l[1:], -1)
        lm_old = torch.logsumexp(po[1:], -1)
        new = torch.stack([l[0], lm_new])
        old = torch.stack([po[0], lm_old])
        A = torch.from_numpy(g["bin_adv"] * g["bin_w"]).float().to(dev)
        surr = surr + _clipped(new, old, A, clip)
        # destination given a move
        z = torch.from_numpy(g["acts"][1:]).long().to(dev)
        if len(z):
            A = torch.from_numpy(g["dst_adv"] / len(z)).float().to(dev)
            surr = surr + _clipped(l.gather(-1, z) - lm_new, po.gather(-1, z) - lm_old, A, clip)
        ok = torch.isfinite(l) & torch.isfinite(lpr[i, 0])
        a_, r_ = l.masked_fill(~ok, 0.0), lpr[i, 0].masked_fill(~ok, 0.0)
        kl = kl + (a_.exp() * ok * (a_ - r_)).sum()
    return -surr / max(len(groups), 1), kl / max(len(groups), 1)


def run_eval(task):
    """task["H"] imagined steps from a validation state with no repositioning, the platform's rule
    and the agent (the three worlds in lockstep, same random numbers)."""
    from world.wm3 import agent as AG
    try:
        agent = _agent(task["path"], task["version"])
        W = _load(task["start"])
        log0, inc0 = len(W.log), W.income.copy()
        ws = [AG.clone(W).setup(agent if m == "agent" else None, policy_seed=task["seed"], mode=m) for m in ("none", "rule", "agent")]
        AG.rollout_many(ws, task["H"], CHUNK[_W["key"]])
        out = []
        for Wm in ws:
            m = AG.outcome(Wm, log0, inc0)
            out.append({"key": _W["key"], "start": str(task["start"]), "mode": Wm.mode, **m, "reward": reward_of(m)})
        return out
    finally:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------- learner (GPU)
def policy_terms(agent, ref, groups, dev, clip, beta):
    """Pairwise clipped surrogate: for each compared driver, the choice that did better (alternative
    vs base) gains probability in proportion to the normalised reward difference g["adv"]."""
    from world.wm3 import agent as AG
    b, pos, cand, tt = AG.collate_decisions([g["snap"] for g in groups], [g["dec"] for g in groups], dev)
    lg = agent.logits(b, pos, cand, tt).float()          # float32, as the workers that sampled the choices
    with torch.no_grad():
        lr = ref.logits(b, pos, cand, tt).float()
    lp, lpr = torch.log_softmax(lg, -1), torch.log_softmax(lr, -1)
    surr, kl, n = 0.0, 0.0, 0
    for i, g in enumerate(groups):
        k = len(g["dec"]["pos"])
        adv = torch.from_numpy(g["adv"]).float().to(dev)                    # (k,)
        po = torch.from_numpy(g["p_old"]).float().to(dev)
        for acts, sign in ((g["a_alt"], 0.5), (g["a_base"], -0.5)):
            a = torch.from_numpy(acts).long().to(dev)[:, None]
            new = lp[i, :k].gather(-1, a)[:, 0]
            old = torch.log(po.gather(-1, a)[:, 0].clamp(min=1e-12))
            ratio = torch.exp(new - old)
            A = sign * adv
            surr = surr + torch.minimum(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).sum()
        ok = torch.isfinite(lp[i, :k]) & torch.isfinite(lpr[i, :k])
        a_, r_ = lp[i, :k].masked_fill(~ok, 0.0), lpr[i, :k].masked_fill(~ok, 0.0)    # no inf - inf in the gradient
        kl = kl + (a_.exp() * ok * (a_ - r_)).sum()
        n += k
    return -surr / max(n, 1), kl / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init", default="wm3_exo", help="world model to start from, or 'none' for random weights")
    ap.add_argument("--env", default="wm3_exo")
    ap.add_argument("--init-agent", default=None, help="start from this whole agent (data/wm3_models/<name>.pt), e.g. bc_pre")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--groups", type=int, nargs=2, default=[16, 6], help="groups per iteration: SF, Pune")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--credit", default="driver", choices=["driver", "pair"],
                    help="driver: GRPO group = one driver's alternative choices (default); pair: v1 one-driver-difference")
    ap.add_argument("--H", type=int, default=40, help="30-s steps after the decision")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-scalar", type=float, default=1e-2)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--aux", type=float, default=1.0)
    ap.add_argument("--workers", type=int, nargs=2, default=[2, 2], help="SF and Pune workers")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-starts", type=int, default=6)
    ap.add_argument("--cities", nargs="+", default=["sf_hd", "pune_hd"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from world import cities
    from world.wm3 import agent as AG
    from world.wm3 import data as WD
    from world.wm3 import model as WM
    TR = _tr()
    dev = TR.DEV
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tabs = TR.tables()
    ck0 = torch.load(TR.MODELS / f"{args.env}.pt", map_location=dev)
    arch = ck0["arch"]
    wm = TR.make_model(tabs, **arch)
    if args.init != "none":
        wm.load_state_dict(torch.load(TR.MODELS / f"{args.init}.pt", map_location=dev)["state"])
    agent = AG.Agent(wm).to(dev)
    if args.init_agent:                       # a whole agent, e.g. one that imitated the rule (30_wm3_rl_bc.py)
        AG.load_agent_state(agent, torch.load(TR.MODELS / f"{args.init_agent}.pt", map_location=dev)["state"])
    ref = copy.deepcopy(agent).eval()         # KL anchor: the starting agent
    for p in ref.parameters():
        p.requires_grad_(False)
    scal = [p for n, p in agent.named_parameters() if n in ("b_stay", "w_tt")]
    head = [p for n, p in agent.named_parameters() if not n.startswith("wm.") and n not in ("b_stay", "w_tt")]
    back = [p for n, p in agent.named_parameters() if n.startswith("wm.")]
    # the two scalars (stay bias, travel-time weight) set how often and how far drivers move; Adam moves
    # a parameter by ~lr per step, so they get their own larger rate
    opt = torch.optim.AdamW([{"params": scal, "lr": args.lr_scalar}, {"params": head, "lr": args.lr},
                             {"params": back, "lr": args.lr_backbone}], weight_decay=0.0)
    pools = {k: WD.DayPool("train", n_days=2, swap=40, seed=args.seed + i, keys=[k]) for i, k in enumerate(WD.CITY_KEYS)}
    val = TR.fixed_snaps("val", 24, seed=1)

    starts = {k: sorted((cities.get(k).data_dir / "wm3_rl" / "starts").glob("train_*.pkl")) for k in args.cities}
    vstarts = {k: sorted((cities.get(k).data_dir / "wm3_rl" / "starts").glob("val_*.pkl"))[: args.eval_starts] for k in args.cities}
    out = cities.OUTPUTS / "wm3" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    wpath = Path("/dev/shm") / f"agent_{args.tag}.pt"
    ctx = mp.get_context("spawn")
    nw = dict(zip(["sf_hd", "pune_hd"], args.workers))
    pool = {k: ctx.Pool(nw[k], initializer=worker_init, initargs=(k, args.env)) for k in args.cities}
    print(f"== {args.tag}: init {args.init_agent or args.init}, beta {args.beta}, env {args.env}; {sum(len(v) for v in starts.values())} training starts, "
          f"G {args.G} K {args.K} H {args.H}; workers {nw}", flush=True)

    def save_w(version):
        torch.save({"state": {k: v.detach().cpu() for k, v in agent.state_dict().items()}, "arch": arch}, wpath)

    def evaluate(it, version):
        tasks = [(k, {"path": str(wpath), "version": version, "start": s, "seed": 7, "H": 120})
                 for k in args.cities for s in vstarts[k]]
        res = {k: pool[k].map_async(run_eval, [t for kk, t in tasks if kk == k]) for k in args.cities}
        rows = [r for k in args.cities for rr in res[k].get() for r in rr]
        df = pd.DataFrame(rows)
        df["iter"] = it
        wmv = TR.evaluate(agent.wm, val)
        agent.eval()
        torch.cuda.empty_cache()
        msg = []
        for k in args.cities:
            d = df[df.key == k]
            piv = {c: d.pivot_table(index="start", columns="mode", values=c) for c in ("reward", "pickups", "welfare", "moved")}
            msg.append(f"{k}: reward agent-none {float((piv['reward'].agent - piv['reward'].none).mean()):+.2f} rule-none "
                       f"{float((piv['reward'].rule - piv['reward'].none).mean()):+.2f} (pickups {float((piv['pickups'].agent - piv['pickups'].none).mean()):+.1f}/"
                       f"{float((piv['pickups'].rule - piv['pickups'].none).mean()):+.1f}, welfare {float((piv['welfare'].agent - piv['welfare'].none).mean()):+.2f}/"
                       f"{float((piv['welfare'].rule - piv['welfare'].none).mean()):+.2f}, moved {piv['moved'].agent.mean():.0f}/{piv['moved'].rule.mean():.0f})")
        nll = sum(wmv.values()) / 2
        print(f"  eval it {it}: " + "; ".join(msg) + f"; world-model val NLL {nll:.3f}", flush=True)
        df["wm_val_nll"] = nll
        return df

    log, evals, t0, version = [], [], time.time(), 0
    save_w(version)
    evals.append(evaluate(0, version))
    for it in range(1, args.iters + 1):
        tasks = []
        ng = dict(zip(["sf_hd", "pune_hd"], args.groups))
        for k in args.cities:
            for _ in range(ng[k]):
                tasks.append((k, {"path": str(wpath), "version": version, "start": starts[k][rng.integers(len(starts[k]))],
                                  "seed": int(rng.integers(1 << 30)), "G": args.G, "K": args.K, "H": args.H}))
        fn = run_group_driver if args.credit == "driver" else run_group
        res = {k: pool[k].map_async(fn, [t for kk, t in tasks if kk == k]) for k in args.cities}
        groups = [g for k in args.cities for g in res[k].get() if g is not None]
        for k in args.cities:                       # reward differences, normalised per city and iteration
            gs = [g for g in groups if g["key"] == k]
            if args.credit == "driver":
                driver_advantages(gs)
            elif gs:
                sd = float(np.concatenate([g["delta"] for g in gs]).std())
                for g in gs:
                    g["adv"] = g["delta"] / (sd + 0.05)
        # update
        stats = {"surr": 0.0, "kl": 0.0, "aux": 0.0}
        agent.train()
        for ep in range(args.epochs):
            for k in args.cities:
                gs = [g for g in groups if g["key"] == k]
                step = LBATCH[k] if args.credit == "driver" else 3
                for i in range(0, len(gs), step):
                    if args.credit == "driver":
                        surr, kl = policy_terms_driver(agent, ref, gs[i: i + step], dev, args.clip)
                    else:
                        surr, kl = policy_terms(agent, ref, gs[i: i + 3], dev, args.clip, args.beta)
                    day, si = pools[k].sample()
                    bb = WD.collate([WD.snapshot(day, si)], dev)
                    with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
                        o = agent.wm(bb)
                    o = {kk: (v.float() if torch.is_tensor(v) else v) for kk, v in o.items()}
                    aux = sum(v for v in WM.losses(agent.wm, bb, o).values() if torch.is_tensor(v) and torch.isfinite(v))
                    loss = surr + args.beta * kl + args.aux * aux
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    opt.step()
                    stats["surr"] += float(surr.detach()); stats["kl"] += float(kl.detach()); stats["aux"] += float(aux)
        torch.cuda.empty_cache()
        agent.eval()
        version += 1
        save_w(version)
        nb = max(1, args.epochs * sum(math.ceil(sum(g["key"] == k for g in groups) / (LBATCH[k] if args.credit == "driver" else 3)) for k in args.cities))
        row = {"iter": it, "min": (time.time() - t0) / 60, "groups": len(groups), **{k: v / nb for k, v in stats.items()}}
        for k in args.cities:
            gs = [g for g in groups if g["key"] == k]
            if gs:
                row[f"{k}/reward"] = float(np.mean([g["rewards"].mean() for g in gs]))
                row[f"{k}/spread"] = float(np.mean([g["rewards"].std() for g in gs]))
                if args.credit == "driver":
                    row[f"{k}/move_gain"] = float(np.mean([g["rewards"][1:].mean() - g["rewards"][0] for g in gs]))
                    row[f"{k}/best_move_gain"] = float(np.mean([g["rewards"][1:].max() - g["rewards"][0] for g in gs]))
                else:
                    row[f"{k}/mean_delta"] = float(np.mean(np.concatenate([g["delta"] for g in gs])))
                    row[f"{k}/alt_is_move"] = float(np.mean(np.concatenate([g["a_alt"] > 0 for g in gs])))
                row[f"{k}/moved"] = float(np.mean([np.mean([m["moved"] for m in g["metrics"]]) for g in gs]))
                row[f"{k}/pickups"] = float(np.mean([np.mean([m["pickups"] for m in g["metrics"]]) for g in gs]))
                row[f"{k}/welfare"] = float(np.mean([np.mean([m["welfare"] for m in g["metrics"]]) for g in gs]))
                row[f"{k}/p_move"] = float(np.mean([1 - g["p_old"][:, 0].mean() for g in gs]))
        log.append(row)
        print(f"  it {it:4d} {row['min']:.1f} min  " + "  ".join(f"{k} {v:.3f}" for k, v in row.items() if k not in ("iter", "min")), flush=True)
        pd.DataFrame(log).to_csv(out / f"train_{args.tag}.csv", index=False)
        torch.save({"state": agent.state_dict(), "arch": arch, "iter": it, "init": args.init, "args": vars(args)},
                   TR.MODELS / f"{args.tag}.pt")
        if it % args.eval_every == 0 or it == args.iters:
            evals.append(evaluate(it, version))
            pd.concat(evals).to_csv(out / f"eval_{args.tag}.csv", index=False)
    for p in pool.values():
        p.close()


if __name__ == "__main__":
    main()
