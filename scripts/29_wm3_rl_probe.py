"""World model 3, stage 5 probe: how much does ONE driver's move change the imagined reward, and how
noisy is that difference as the horizon grows? (Why stage-5 training credits choices over 40 steps.)

    .venv/bin/python scripts/29_wm3_rl_probe.py sf_hd 4 15
    .venv/bin/python scripts/29_wm3_rl_probe.py pune_hd 3 11

From random training start states: a base world where every deciding driver stays, and copies where
one driver moves to a destination drawn from the untrained agent; same random numbers. Prints the
mean and s.d. of the difference in pickups, welfare and reward after 10/20/40/60/120 steps.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("tr3", ROOT / "scripts/18_wm3_train.py"); TR = importlib.util.module_from_spec(spec); spec.loader.exec_module(TR)
from world import cities
from world.wm3 import agent as AG, rollout as RO, state as S, model as WM
from world.routing import LiveRouter
key = sys.argv[1]; nstart = int(sys.argv[2]); M = int(sys.argv[3])
tabs = TR.tables(); env, _ = TR.load("wm3_exo", tabs); env.eval()
g = S.CityGraph(cities.get(key)); rules = RO.CityRules(g); router = LiveRouter(g.net, g.traffic)
agent = AG.Agent(env).to(TR.DEV).eval()
starts = sorted((cities.get(key).data_dir / "wm3_rl/starts").glob("train_*.pkl"))
rng = np.random.default_rng(0)
CK = [10, 20, 40, 60, 120]; chunk = 8 if key == "sf_hd" else 2
rows = []
for s in rng.choice(starts, nstart, replace=False):
    W = AG.load_start(s, env, rules, router); W.setup(agent, policy_seed=3, mode="agent")
    probe = AG.clone(W); rec = AG.step_many([probe])[0]; dec, p = rec["dec"], rec["p"]; n = len(dec["drv"])
    a0 = np.zeros(n, int)
    ws = [AG.clone(W)]; ws[0].override = {W.e: a0}
    idx = rng.choice(n, min(M, n), replace=False)
    for i in idx:
        a = a0.copy(); q = p[i, 1:].copy()
        if not np.isfinite(q).any() or q.sum() <= 0: continue
        a[i] = 1 + int(rng.choice(len(q), p=q / q.sum()))
        Wj = AG.clone(W); Wj.override = {W.e: a}; ws.append(Wj)
    log0, inc0 = len(W.log), W.income.copy(); done = 0
    for h in CK:
        AG.rollout_many(ws, h - done, chunk); done = h
        m = [AG.outcome(x, log0, inc0) for x in ws]
        for j in range(1, len(ws)):
            rows.append({"start": s.name, "H": h, "dp": m[j]["pickups"] - m[0]["pickups"], "dw": m[j]["welfare"] - m[0]["welfare"]})
    print(s.name, n, "deciding", flush=True)
d = pd.DataFrame(rows); d["dr"] = d.dp + d.dw
print(key); print(d.groupby("H")[["dp", "dw", "dr"]].agg(["mean", "std"]).round(3).to_string())
(ROOT / "outputs/wm3/stage5/probe").mkdir(parents=True, exist_ok=True)
d.to_csv(ROOT / f"outputs/wm3/stage5/probe/probe_{key}.csv", index=False)
