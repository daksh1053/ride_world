"""World model 3, stage 5e: teach the agent the platform's rule by imitation (supervised fine-tuning)
before GRPO, as language models are fine-tuned on demonstrations before RL.

    .venv/bin/python scripts/30_wm3_rl_bc.py --collect                      # rule decisions in imagination
    .venv/bin/python scripts/30_wm3_rl_bc.py --init wm3_exo --tag bc_pre    # imitate, pretrained world model
    .venv/bin/python scripts/30_wm3_rl_bc.py --init none --tag bc_scratch   # imitate, random weights

Why: GRPO alone (scripts/26_wm3_grpo.py, outputs/wm3/stage5/v0-v4) learns to stop repositioning.
The rule's gain (about +0.1 pickup per move) only shows after about an hour. One driver's choice
is too noisy to judge over an hour (outputs/wm3/stage5/probe/), and over 20 minutes the average
move looks like a small loss. Starting from the rule, RL only has to find where to deviate from it.

--collect: from the training start states (scripts/25_wm3_rl_starts.py) the frozen world model
imagines --H steps with the platform's rule repositioning; at every decision epoch the deciding
drivers (agent.decision_inputs) and the rule's choice for each (stay, or the zone it sent them to)
are saved to data/<city>/wm3_rl/bc_<split>.pkl. Validation starts give the held-out set.

Training: cross-entropy of the agent's choice distribution on the rule's choices, plus aux * the world
model's own prediction loss on training snapshots (so the agent stays a world model). Writes
data/wm3_models/<tag>.pt (full agent state, loadable by 26_wm3_grpo.py --init-agent) and
outputs/wm3/stage5/30_<tag>.csv (per epoch: held-out NLL, accuracy, P(move) vs the rule's move rate).
"""

from __future__ import annotations

import argparse
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


def _tr():
    spec = importlib.util.spec_from_file_location("tr3", ROOT / "scripts" / "18_wm3_train.py")
    TR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TR)
    return TR


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
            out = cities.get(key).data_dir / "wm3_rl" / f"bc_{split}.pkl"
            if out.exists():
                print(f"  {out} exists", flush=True)
                continue
            starts = sorted((cities.get(key).data_dir / "wm3_rl" / "starts").glob(f"{split}_*.pkl"))
            recs, t0 = [], time.time()
            for c in range(0, len(starts), CHUNK[key]):
                ws = []
                for s in starts[c: c + CHUNK[key]]:
                    W = AG.load_start(s, env, rules, router).setup(None, mode="rule")
                    W.record = True
                    ws.append(W)
                for _ in range(args.H):
                    for W, r in zip(ws, AG.step_many(ws)):
                        if r is not None and len(r["dec"]["drv"]):
                            recs.append({"key": key, **r})
                torch.cuda.empty_cache()
                print(f"  {key} {split}: {c + len(ws)}/{len(starts)} starts, {len(recs)} decision epochs, "
                      f"{time.time() - t0:.0f} s", flush=True)
            n = sum(len(r["a"]) for r in recs)
            mv = sum(int((r["a"] > 0).sum()) for r in recs)
            nr = sum(r["n_rule"] for r in recs)
            print(f"  {key} {split}: {n} decisions, rule moves {mv} ({mv / max(n, 1):.3f}); "
                  f"rule orders {nr}, of them expressible as agent actions {mv / max(nr, 1):.3f}", flush=True)
            out.write_bytes(pickle.dumps(recs, protocol=pickle.HIGHEST_PROTOCOL))


def batch_terms(agent, recs, dev):
    from world.wm3 import agent as AG
    b, pos, cand, tt = AG.collate_decisions([r["snap"] for r in recs], [r["dec"] for r in recs], dev)
    lg = agent.logits(b, pos, cand, tt).float()
    lp = torch.log_softmax(lg, -1)
    n = max(len(r["a"]) for r in recs)
    a = np.zeros((len(recs), n), np.int64)
    m = np.zeros((len(recs), n), bool)
    for i, r in enumerate(recs):
        a[i, :len(r["a"])] = r["a"]
        m[i, :len(r["a"])] = True
    a, m = torch.from_numpy(a).to(dev), torch.from_numpy(m).to(dev)
    la = lp.gather(-1, a[..., None])[..., 0]
    nll = -(la * m).sum() / m.sum()
    pm = 1 - lp[..., 0].exp()
    stats = {"nll": float(nll.detach()), "acc": float(((lp.argmax(-1) == a) & m).sum() / m.sum()),
             "p_move": float((pm * m).sum() / m.sum()), "rule_move": float(((a > 0) & m).sum() / m.sum()),
             "n": int(m.sum())}
    return nll, stats


def train(args, TR):
    from world import cities
    from world.wm3 import agent as AG
    from world.wm3 import data as WD
    from world.wm3 import model as WM
    dev = TR.DEV
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tabs = TR.tables()
    arch = torch.load(TR.MODELS / f"{args.env}.pt", map_location="cpu")["arch"]
    wm = TR.make_model(tabs, **arch)
    if args.init != "none":
        wm.load_state_dict(torch.load(TR.MODELS / f"{args.init}.pt", map_location=dev)["state"])
    agent = AG.Agent(wm).to(dev)
    data = {s: {k: pickle.loads((cities.get(k).data_dir / "wm3_rl" / f"bc_{s}.pkl").read_bytes()) for k in args.cities}
            for s in ("train", "val")}
    head = [p for n, p in agent.named_parameters() if not n.startswith("wm.")]
    back = [p for n, p in agent.named_parameters() if n.startswith("wm.")]
    opt = torch.optim.AdamW([{"params": head, "lr": args.lr}, {"params": back, "lr": args.lr_backbone}], weight_decay=0.0)
    pools = {k: WD.DayPool("train", n_days=2, swap=40, seed=args.seed + i, keys=[k]) for i, k in enumerate(WD.CITY_KEYS)}
    val = TR.fixed_snaps("val", 24, seed=1)

    def evaluate():
        agent.eval()
        out = {}
        with torch.no_grad():
            for k in args.cities:
                st = [batch_terms(agent, data["val"][k][i: i + LB[k]], dev)[1] for i in range(0, len(data["val"][k]), LB[k])]
                w = np.array([s["n"] for s in st], float)
                out.update({f"{k}/{q}": float(np.average([s[q] for s in st], weights=w)) for q in ("nll", "acc", "p_move", "rule_move")})
        wmv = TR.evaluate(agent.wm, val)
        out["wm_val_nll"] = sum(wmv.values()) / 2
        torch.cuda.empty_cache()
        return out

    LB = {"sf_hd": 4, "pune_hd": 2}
    log, t0 = [], time.time()
    row = {"epoch": 0, **evaluate()}
    log.append(row)
    print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    out = cities.OUTPUTS / "wm3" / "stage5"
    best = np.inf
    for ep in range(1, args.epochs + 1):
        agent.train()
        order = [(k, i) for k in args.cities for i in rng.permutation(len(data["train"][k]))]
        order = [order[j] for j in rng.permutation(len(order))]
        tr = []
        for j in range(0, len(order), 4):
            for k in args.cities:
                idx = [i for kk, i in order[j: j + 4] if kk == k][: LB[k]]
                if not idx:
                    continue
                nll, st = batch_terms(agent, [data["train"][k][i] for i in idx], dev)
                day, si = pools[k].sample()
                bb = WD.collate([WD.snapshot(day, si)], dev)
                with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
                    o = agent.wm(bb)
                o = {kk: (v.float() if torch.is_tensor(v) else v) for kk, v in o.items()}
                aux = sum(v for v in WM.losses(agent.wm, bb, o).values() if torch.is_tensor(v) and torch.isfinite(v))
                loss = nll + args.aux * aux
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
                tr.append(st["nll"])
        torch.cuda.empty_cache()
        row = {"epoch": ep, "train_nll": float(np.mean(tr)), "min": (time.time() - t0) / 60, **evaluate()}
        log.append(row)
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        pd.DataFrame(log).to_csv(out / f"30_{args.tag}.csv", index=False)
        score = np.mean([row[f"{k}/nll"] for k in args.cities])
        if score < best:
            best = score
            torch.save({"state": agent.state_dict(), "arch": arch, "epoch": ep, "init": args.init, "args": vars(args)},
                       TR.MODELS / f"{args.tag}.pt")
    print(f"  best held-out imitation NLL {best:.4f} -> {TR.MODELS / (args.tag + '.pt')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--init", default="wm3_exo")
    ap.add_argument("--env", default="wm3_exo")
    ap.add_argument("--H", type=int, default=60, help="imagined steps per start state when collecting")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
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
