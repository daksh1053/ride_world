"""World model 3, stage 6a: the agent also matches riders to drivers -- imitation of the platform's
matching first (as stage 5 did for repositioning).

    .venv/bin/python scripts/31_wm3_match_bc.py --collect                                  # the rule's offers in imagination
    .venv/bin/python scripts/31_wm3_match_bc.py --init-agent rl_bc_pre --tag mbc_pre        # pretrained agent of stage 5
    .venv/bin/python scripts/31_wm3_match_bc.py --init-agent rl_bc_scratch --tag mbc_scratch

The matching head (world/wm3/agent.py, Agent.match_logits): every 30-s epoch the pending requests,
oldest first, each choose one idle driver of their mode within the pickup limit (not one who
declined them, not one taken by an earlier request) or hold.

--collect: from the training start states the frozen world model imagines --H steps with the
platform's rule (HDDispatch: min-ETA assignment per mode, airport queue, rebalancing); every
--every epochs the matching inputs and the rule's offers as the agent's actions are saved to
data/<city>/wm3_rl/match_<split>.pkl (snapshots in float16). Validation starts give the held-out set.

Training: cross-entropy of each request's choice given the drivers the rule's earlier choices took,
plus beta_repo * KL(starting agent || agent) on stage-5 repositioning states (bc_train.pkl) so the
repositioning the agent learned in stage 5 is kept, plus aux * the world model's own prediction
loss. Writes data/wm3_models/<tag>.pt and outputs/wm3/stage6/31_<tag>.csv (per epoch: held-out
NLL, exact-driver accuracy, hold rate vs the rule's, mean pickup ETA of the chosen vs the rule's).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CHUNK = {"sf_hd": 8, "pune_hd": 2}
LB = {"sf_hd": 4, "pune_hd": 2}                 # snapshots per learner batch (GPU memory)


def _tr():
    spec = importlib.util.spec_from_file_location("tr3", ROOT / "scripts" / "18_wm3_train.py")
    TR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TR)
    return TR


def _half(snap):
    return {k: (v.astype(np.float16) if isinstance(v, np.ndarray) and v.dtype == np.float32 else v) for k, v in snap.items()}


def collect(args, TR):
    from world import cities
    from world.routing import LiveRouter
    from world.wm3 import agent as AG
    from world.wm3 import rollout as RO
    from world.wm3 import state as S
    tabs = TR.tables()
    env, _ = TR.load(args.env, tabs)
    env.eval()
    for key in args.cities:
        g = S.CityGraph(cities.get(key))
        rules = RO.CityRules(g)
        router = LiveRouter(g.net, g.traffic)
        for split in ("train", "val"):
            out = cities.get(key).data_dir / "wm3_rl" / f"match_{split}.pkl"
            if out.exists():
                print(f"  {out} exists", flush=True)
                continue
            starts = sorted((cities.get(key).data_dir / "wm3_rl" / "starts").glob(f"{split}_*.pkl"))
            recs, t0, miss, n_rule = [], time.time(), 0, 0
            for c in range(0, len(starts), CHUNK[key]):
                ws = []
                for s in starts[c: c + CHUNK[key]]:
                    W = AG.load_start(s, env, rules, router).setup(None, mode="rule")
                    W.record_match = args.every
                    ws.append(W)
                for _ in range(args.H):
                    AG.step_many(ws)
                    for W in ws:
                        r = W.last_match
                        if r is not None:
                            miss += r["miss"]
                            n_rule += r["n_rule"]
                            recs.append({"key": key, "snap": _half(r["snap"]), "mi": r["mi"], "a": r["a"], "e": r["e"]})
                torch.cuda.empty_cache()
                print(f"  {key} {split}: {c + len(ws)}/{len(starts)} starts, {len(recs)} epochs, {time.time() - t0:.0f} s", flush=True)
            n = sum(len(r["a"]) for r in recs)
            off = sum(int((r["a"] > 0).sum()) for r in recs)
            print(f"  {key} {split}: {n} request choices in {len(recs)} epochs, rule offers {off} ({off / max(n, 1):.3f} of choices), "
                  f"rule offers outside the agent's choice set {miss} of {n_rule}; mean candidates "
                  f"{np.mean([len(r['mi']['drv']) for r in recs]):.1f} drivers, {np.mean([len(r['a']) for r in recs]):.1f} requests", flush=True)
            out.write_bytes(pickle.dumps(recs, protocol=pickle.HIGHEST_PROTOCOL))


def match_terms(agent, recs, dev):
    """Sequential cross-entropy on the rule's choices; stats."""
    from world.wm3 import agent as AG
    b, rpos, dpos, feas, eta = AG.collate_matches([r["snap"] for r in recs], [r["mi"] for r in recs], dev)
    lg = agent.match_logits(b, rpos, dpos, feas, eta).float()
    B, n, m1 = lg.shape
    A = np.zeros((B, n), np.int64)
    U = np.zeros((B, n, m1 - 1), bool)
    M = np.zeros((B, n), bool)
    for i, r in enumerate(recs):
        k = len(r["a"])
        A[i, :k] = r["a"]
        U[i, :k, :len(r["mi"]["drv"])] = AG.used_before(r["a"], len(r["mi"]["drv"]))
        M[i, :k] = True
    A, U, M = (torch.from_numpy(x).to(dev) for x in (A, U, M))
    lp = AG.seq_logprobs(lg, U)
    la = lp.gather(-1, A[..., None])[..., 0]
    nll = -(la * M).sum() / M.sum()
    with torch.no_grad():
        p = lp.exp()
        etam = torch.cat([torch.zeros_like(eta[..., :1]), eta], -1)
        e_agent = (p * etam.masked_fill(~torch.isfinite(lp), 0)).sum(-1)       # expected pickup minutes (0 if held)
        e_rule = etam.gather(-1, A[..., None])[..., 0]
        off = (A > 0) & M
        stats = {"nll": float(nll), "acc": float(((lp.argmax(-1) == A) & M).sum() / M.sum()),
                 "p_hold": float((p[..., 0] * M).sum() / M.sum()), "rule_hold": float(((A == 0) & M).sum() / M.sum()),
                 "eta_agent": float((e_agent * off).sum() / off.sum().clamp(min=1)), "eta_rule": float((e_rule * off).sum() / off.sum().clamp(min=1)),
                 "n": int(M.sum())}
    return nll, stats


def repo_kl(agent, ref, recs, dev):
    """KL(starting agent || agent) of the repositioning choice on stage-5 decision states."""
    from world.wm3 import agent as AG
    b, pos, cand, tt = AG.collate_decisions([r["snap"] for r in recs], [r["dec"] for r in recs], dev)
    lp = torch.log_softmax(agent.logits(b, pos, cand, tt).float(), -1)
    with torch.no_grad():
        lr = torch.log_softmax(ref.logits(b, pos, cand, tt).float(), -1)
    ok = torch.isfinite(lp) & torch.isfinite(lr) & (pos >= 0)[..., None]
    a_, r_ = lp.masked_fill(~ok, 0.0), lr.masked_fill(~ok, 0.0)
    return (r_.exp() * ok * (r_ - a_)).sum() / (pos >= 0).sum().clamp(min=1)


def train(args, TR):
    from world import cities
    from world.wm3 import agent as AG
    from world.wm3 import data as WD
    from world.wm3 import model as WM
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
    data = {s: {k: pickle.loads((cities.get(k).data_dir / "wm3_rl" / f"match_{s}.pkl").read_bytes()) for k in args.cities}
            for s in ("train", "val")}
    repo = {}
    for k in args.cities:                    # stage-5 repositioning states for the KL anchor (a sample)
        r = pickle.loads((cities.get(k).data_dir / "wm3_rl" / "bc_train.pkl").read_bytes())
        repo[k] = [r[i] for i in rng.choice(len(r), min(args.n_repo, len(r)), replace=False)]
        del r
    match_keys = AG.MATCH_KEYS
    scal = [p for n, p in agent.named_parameters() if n in ("w_eta", "b_hold")]
    head = [p for n, p in agent.named_parameters() if n.startswith(match_keys) and n not in ("w_eta", "b_hold")]
    rest = [p for n, p in agent.named_parameters() if not n.startswith("wm.") and not n.startswith(match_keys)]
    back = [p for n, p in agent.named_parameters() if n.startswith("wm.")]
    opt = torch.optim.AdamW([{"params": scal, "lr": args.lr_scalar}, {"params": head, "lr": args.lr},
                             {"params": rest, "lr": args.lr_backbone}, {"params": back, "lr": args.lr_backbone}], weight_decay=0.0)
    pools = {k: WD.DayPool("train", n_days=2, swap=40, seed=args.seed + i, keys=[k]) for i, k in enumerate(WD.CITY_KEYS)}
    val = TR.fixed_snaps("val", 24, seed=1)

    def evaluate():
        agent.eval()
        out = {}
        with torch.no_grad():
            for k in args.cities:
                st = [match_terms(agent, data["val"][k][i: i + LB[k]], dev)[1] for i in range(0, len(data["val"][k]), LB[k])]
                w = np.array([s["n"] for s in st], float)
                out.update({f"{k}/{q}": float(np.average([s[q] for s in st], weights=w))
                            for q in ("nll", "acc", "p_hold", "rule_hold", "eta_agent", "eta_rule")})
                kl = [float(repo_kl(agent, ref, data_r, dev)) for data_r in
                      [repo[k][i: i + LB[k]] for i in range(0, min(len(repo[k]), 40), LB[k])]]
                out[f"{k}/repo_kl"] = float(np.mean(kl))
        wmv = TR.evaluate(agent.wm, val)
        out["wm_val_nll"] = sum(wmv.values()) / 2
        torch.cuda.empty_cache()
        return out

    out = cities.OUTPUTS / "wm3" / "stage6"
    out.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    row = {"epoch": 0, **evaluate()}
    log.append(row)
    print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    best = np.inf
    for ep in range(1, args.epochs + 1):
        agent.train()
        tr = []
        order = [(k, i) for k in args.cities for i in range(0, len(data["train"][k]), LB[k])]
        perm = {k: rng.permutation(len(data["train"][k])) for k in args.cities}
        order = [order[j] for j in rng.permutation(len(order))]
        for step, (k, i) in enumerate(order):
            batch = [data["train"][k][j] for j in perm[k][i: i + LB[k]]]
            nll, st = match_terms(agent, batch, dev)
            loss = nll
            if args.beta_repo > 0:
                rb = [repo[k][j] for j in rng.choice(len(repo[k]), LB[k], replace=False)]
                loss = loss + args.beta_repo * repo_kl(agent, ref, rb, dev)
            if args.aux > 0:
                day, si = pools[k].sample()
                bb = WD.collate([WD.snapshot(day, si)], dev)
                with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
                    o = agent.wm(bb)
                o = {kk: (v.float() if torch.is_tensor(v) else v) for kk, v in o.items()}
                loss = loss + args.aux * sum(v for v in WM.losses(agent.wm, bb, o).values() if torch.is_tensor(v) and torch.isfinite(v))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
            tr.append(st["nll"])
        torch.cuda.empty_cache()
        row = {"epoch": ep, "train_nll": float(np.mean(tr)), "min": (time.time() - t0) / 60, **evaluate()}
        log.append(row)
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        pd.DataFrame(log).to_csv(out / f"31_{args.tag}.csv", index=False)
        score = np.mean([row[f"{k}/nll"] for k in args.cities])
        if score < best:
            best = score
            torch.save({"state": agent.state_dict(), "arch": arch, "epoch": ep, "init": args.init_agent, "args": vars(args)},
                       TR.MODELS / f"{args.tag}.pt")
    print(f"  best held-out matching NLL {best:.4f} -> {TR.MODELS / (args.tag + '.pt')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--init-agent", default="rl_bc_pre", help="stage-5 agent to start from (data/wm3_models/<name>.pt)")
    ap.add_argument("--env", default="wm3_exo")
    ap.add_argument("--H", type=int, default=40, help="imagined steps per start state when collecting")
    ap.add_argument("--every", type=int, default=2, help="record the matching every this many epochs")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-scalar", type=float, default=1e-2)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--beta-repo", type=float, default=1.0)
    ap.add_argument("--n-repo", type=int, default=1500)
    ap.add_argument("--aux", type=float, default=1.0)
    ap.add_argument("--cities", nargs="+", default=["sf_hd", "pune_hd"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    TR = _tr()
    if args.collect:
        collect(args, TR)
    else:
        train(args, TR)


if __name__ == "__main__":
    main()
