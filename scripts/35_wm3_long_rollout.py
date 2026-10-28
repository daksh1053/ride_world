"""World model 3, stage 8a: long imagined rollouts and whether their spread is right.

    .venv/bin/python scripts/35_wm3_long_rollout.py --city sf_hd --seeds 8 --draws 8
    .venv/bin/python scripts/35_wm3_long_rollout.py --report

Stage 3 checked one hour. A world model should stay sensible over many hours of its own samples
and its draws should spread like the real world's luck. Starts: the stage-4 twin seasons' base days
(fresh drivers and riders, never in training), 06:30, 8 hours (960 steps of 30 s) on the model's
own samples under the simulator's own platform policy (HDDispatch, as the simulator ran it).

Compared every 5 min with what the simulator did:
  city-wide  drivers online / idle / with a rider, riders waiting, cumulative new requests,
             pickups, completed trips, cancellations, log-ons, log-offs
  drivers    online at the end, trips and income over the 8 h (per driver)
Baselines: persistence (counts stay, events at the last 30 min's rate) and the historical average
(the same clock time on the other seasons' days, as a forecaster with history but no state would).
Spread: the truth's rank among the draws (uniform if the draws are calibrated) and how often the
truth lies within the draws' range (ideal (n-1)/(n+1)).

Writes data/<city>/wm3_long/seed<k>.npz and outputs/wm3/stage8/35_*.png / 35_long.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from world import cities
from world.wm3 import agent as AG
from world.wm3 import features as FT
from world.wm3 import rollout as RO
from world.wm3 import state as S

_spec = importlib.util.spec_from_file_location("tr3", Path(__file__).with_name("18_wm3_train.py"))
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)
OUT = cities.OUTPUTS / "wm3" / "stage8"
START_H = 6.5
HOURS = 8.0
EVERY = 10                                                     # record every 5 min
STATE_KPI = ["online", "idle", "with_rider", "waiting_riders"]
EVENT_KPI = {"new requests": ["request_created", "quote_declined"], "pickups": ["pickup"], "completed trips": ["dropoff"],
             "cancellations": ["customer_cancelled"], "log-ons": ["driver_online"], "log-offs": ["driver_offline"]}
KPIS = STATE_KPI + list(EVENT_KPI)
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
INK, INK2, GRID, BLUE, ORANGE, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#9a9893"


def twin_days(key):
    root = cities.get(key).data_dir / "wm3_twins"
    recs = [json.loads(p.read_text()) for p in sorted(root.glob("rec_seed*.json"))]
    return [(r["seed"], root / f"seed{r['seed']}" / r["date"]) for r in recs]


def season_prior(day_dir):
    pr = FT.Prior()
    for d in sorted(p for p in day_dir.parent.iterdir() if p.is_dir() and p.name < day_dir.name):
        pr.update_light(d, 5000)
    return pr


def truth_series(L, e0, epochs):
    rows = []
    ev = L.ev
    for e in epochs:
        st = L.status[:, e]
        R = L.open_requests(e)
        t = e * S.DT
        row = {"online": int((st != 0).sum()), "idle": int((st == 1).sum()), "with_rider": int(((st == 4) | (st == 6)).sum()),
               "waiting_riders": int((R.status <= 1).sum())}
        w = ev[(ev.t_event > e0 * S.DT) & (ev.t_event <= t)]
        for k, types in EVENT_KPI.items():
            row[k] = int(((L.req.t_request > e0 * S.DT) & (L.req.t_request <= t)).sum()) if k == "new requests" else int(w.type.isin(types).sum())
        rows.append([row[k] for k in KPIS])
    return np.array(rows, float)


def persistence(L, e0, truth, epochs):
    p = truth.copy()
    t0 = e0 * S.DT
    for j, k in enumerate(KPIS):
        if k in STATE_KPI:
            p[:, j] = truth[0, j]
        else:
            ev = L.ev[(L.ev.t_event > t0 - 1800) & (L.ev.t_event <= t0)]
            n = int(((L.req.t_request > t0 - 1800) & (L.req.t_request <= t0)).sum()) if k == "new requests" else int(ev.type.isin(EVENT_KPI[k]).sum())
            p[:, j] = n / 1800 * (np.asarray(epochs) - e0) * S.DT
    return p


def run_city(key, args):
    city = cities.get(key)
    g = S.CityGraph(city)
    rules = RO.CityRules(g)
    model = TR.load("wm3_exo")[0].eval()
    out = city.data_dir / "wm3_long"
    out.mkdir(exist_ok=True)
    e0 = int(START_H * 3600 / S.DT)
    H = int(HOURS * 3600 / S.DT)
    epochs = list(range(e0, e0 + H + 1, EVERY))
    chunk = {"sf_hd": 8, "pune_hd": 4}[key]
    for seed, day_dir in twin_days(key)[: args.seeds]:
        f = out / f"seed{seed}.npz"
        if f.exists():
            continue
        t0 = time.time()
        L = S.DayLog(g, day_dir)
        prior = season_prior(day_dir)
        truth = truth_series(L, e0, epochs)
        pers = persistence(L, e0, truth, epochs)
        ids = np.flatnonzero(L.status[:, e0] != 0)
        Ws = []
        for s in range(args.draws):
            W = AG.AgentWorld(model, rules, L, prior, e0, None, seed=100 + s)
            W.setup(None, s, "rule")
            W.match = RO.policy_for(L, seed=s)                    # the simulator's own settings (explore 5%)
            W.match._n = e0                                        # its epoch counter as in the simulator
            Ws.append(W)
        inc0 = [W.income.copy() for W in Ws]
        trips0 = [W.count["trips"].copy() for W in Ws]
        ptr = [len(W.log) for W in Ws]
        cnt = [Counter() for _ in Ws]
        sims = np.zeros((args.draws, len(epochs), len(KPIS)))

        def record(j):
            for i, W in enumerate(Ws):
                for x in W.log[ptr[i]:]:
                    cnt[i][x[1]] += 1
                ptr[i] = len(W.log)
                k = W.kpis()
                sims[i, j] = [k[n] if n in STATE_KPI else sum(cnt[i][t] for t in EVENT_KPI[n]) for n in KPIS]

        record(0)
        for j in range(1, len(epochs)):
            AG.rollout_many(Ws, EVERY, chunk)
            record(j)
        e1 = e0 + H
        drv = {"ids": ids, "true_online": L.status[ids, e1] != 0, "true_trips": L.count["trips"][ids, e1] - L.count["trips"][ids, e0],
               "true_income": L.income[ids, e1] - L.income[ids, e0],
               "sim_online": np.stack([W.st[ids] != 0 for W in Ws]),
               "sim_trips": np.stack([W.count["trips"][ids] - t0_[ids] for W, t0_ in zip(Ws, trips0)]),
               "sim_income": np.stack([W.income[ids] - i0[ids] for W, i0 in zip(Ws, inc0)])}
        np.savez_compressed(f, epochs=np.array(epochs), truth=truth, pers=pers, sims=sims, date=str(L.date.date()), **drv)
        print(f"  {key} seed {seed} {L.date.date()}: {(time.time() - t0) / 60:.1f} min; at +8 h pickups "
              f"{sims[:, -1, KPIS.index('pickups')].mean():.0f} (true {truth[-1, KPIS.index('pickups')]:.0f}), online "
              f"{sims[:, -1, 0].mean():.0f} (true {truth[-1, 0]:.0f})", flush=True)
        del Ws
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- report
def load(key):
    fs = sorted((cities.get(key).data_dir / "wm3_long").glob("seed*.npz"))
    return [dict(np.load(f, allow_pickle=True)) for f in fs]


def auc(y, x):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def gini(x):
    x = np.sort(np.maximum(np.asarray(x, float), 0))
    return float((2 * np.arange(1, len(x) + 1) - len(x) - 1) @ x / (len(x) * max(x.sum(), 1e-9))) if len(x) > 1 else 0.0


def summarise(key):
    D = load(key)
    if not D:
        return None
    T = np.stack([d["truth"] for d in D])                         # (seeds, times, kpi)
    M = np.stack([d["sims"] for d in D])                          # (seeds, draws, times, kpi)
    P = np.stack([d["pers"] for d in D])
    n = len(D)
    # historical average: the same clock time on the other seasons' days
    C = np.stack([np.delete(T, i, 0).mean(0) for i in range(n)])
    mean = M.mean(1)
    hz = (D[0]["epochs"] - D[0]["epochs"][0]) * S.DT / 3600
    res = {"n_seeds": n, "n_draws": int(M.shape[1]), "hours": hz.tolist(), "kpi": {}}
    for j, k in enumerate(KPIS):
        scale = np.maximum(np.abs(T[:, :, j]).mean(0), 1.0)
        r = {"model": (np.abs(mean[:, :, j] - T[:, :, j]).mean(0) / scale).tolist(),
             "persistence": (np.abs(P[:, :, j] - T[:, :, j]).mean(0) / scale).tolist(),
             "history": (np.abs(C[:, :, j] - T[:, :, j]).mean(0) / scale).tolist(),
             "truth_mean": T[:, :, j].mean(0).tolist(), "model_mean": mean[:, :, j].mean(0).tolist()}
        # rank of the truth among the draws (ties split at random), at every whole hour
        rk, inside = [], []
        g = np.random.default_rng(0)
        for h in range(1, int(HOURS) + 1):
            ti = int(np.argmin(np.abs(hz - h)))
            for s in range(n):
                x, y = M[s, :, ti, j], T[s, ti, j]
                lo, eq = int((x < y).sum()), int((x == y).sum())
                rk.append(lo + int(g.integers(0, eq + 1)))
                inside.append(bool(x.min() <= y <= x.max()))
        r["rank_hist"] = np.bincount(rk, minlength=M.shape[1] + 1).tolist()
        r["inside_range"] = float(np.mean(inside))
        res["kpi"][k] = r
    # drivers over the whole 8 h
    on_auc = [auc(d["true_online"], d["sim_online"].mean(0)) for d in D]
    inc_r = [np.corrcoef(d["true_income"], d["sim_income"].mean(0))[0, 1] for d in D]
    trip_r = [np.corrcoef(d["true_trips"], d["sim_trips"].mean(0))[0, 1] for d in D]
    g_true = [gini(d["true_income"]) for d in D]
    g_sim = [np.mean([gini(x) for x in d["sim_income"]]) for d in D]
    res["drivers"] = {"online_end_auc": float(np.nanmean(on_auc)), "income_r": float(np.mean(inc_r)), "trips_r": float(np.mean(trip_r)),
                      "gini_true": float(np.mean(g_true)), "gini_model": float(np.mean(g_sim)),
                      "gini_abs_err": float(np.mean(np.abs(np.array(g_true) - np.array(g_sim))))}
    res["ideal_inside"] = (M.shape[1] - 1) / (M.shape[1] + 1)
    return res, (hz, T, M, P, C)


def _ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    allres, data = {}, {}
    for key in ("sf_hd", "pune_hd"):
        s = summarise(key)
        if s:
            allres[key], data[key] = s
    (OUT / "35_long.json").write_text(json.dumps(allres, indent=1))
    keys = list(allres)
    # error vs horizon
    fig, axes = plt.subplots(len(keys), len(KPIS), figsize=(2.2 * len(KPIS), 2.5 * len(keys)), squeeze=False)
    for a, key in enumerate(keys):
        r = allres[key]
        hz = np.array(r["hours"])
        for j, k in enumerate(KPIS):
            ax = axes[a, j]
            _ax(ax)
            kk = r["kpi"][k]
            ax.plot(hz[1:], kk["persistence"][1:], color=GREY, lw=1.5, label="persistence")
            ax.plot(hz[1:], kk["history"][1:], color=ORANGE, lw=1.5, label="historical average")
            ax.plot(hz[1:], kk["model"][1:], color=BLUE, lw=2, label="world model (mean of draws)")
            ax.set_ylim(0, min(1.0, max(max(kk["model"][1:]), max(kk["history"][1:])) * 1.2 + 0.02))
            if a == 0:
                ax.set_title(k, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{CITY_NAME[key]}\nabs. error / true value", fontsize=8, color=INK2)
            if a == len(keys) - 1:
                ax.set_xlabel("hours imagined", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"8 imagined hours from 06:30, fresh seasons ({', '.join(f'{CITY_NAME[k]} {allres[k]['n_seeds']} days' for k in keys)}): "
                 "error of the world model vs two baselines", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "35_long_errors.png", dpi=130)
    plt.close(fig)
    # fan chart for the first seed of each city
    fig, axes = plt.subplots(len(keys), len(KPIS), figsize=(2.2 * len(KPIS), 2.5 * len(keys)), squeeze=False)
    for a, key in enumerate(keys):
        hz, T, M, P, C = data[key]
        for j, k in enumerate(KPIS):
            ax = axes[a, j]
            _ax(ax)
            for m in M[0, :, :, j]:
                ax.plot(START_H + hz, m, color=BLUE, lw=0.7, alpha=0.45)
            ax.plot(START_H + hz, C[0, :, j], color=ORANGE, lw=1.2, ls="--")
            ax.plot(START_H + hz, T[0, :, j], color=INK, lw=1.8)
            if a == 0:
                ax.set_title(k, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(CITY_NAME[key], fontsize=8, color=INK2)
            if a == len(keys) - 1:
                ax.set_xlabel("hour of day", fontsize=8, color=INK2)
    fig.suptitle("One day per city: simulator (black), the world model's 8 draws (blue), historical average (orange, dashed)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "35_long_fan.png", dpi=130)
    plt.close(fig)
    # rank histograms
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 3.2), squeeze=False)
    for a, key in enumerate(keys):
        ax = axes[0, a]
        _ax(ax)
        r = allres[key]
        nb = r["n_draws"] + 1
        tot = np.zeros(nb)
        for k in KPIS:
            tot += np.array(r["kpi"][k]["rank_hist"])
        ax.bar(np.arange(nb), tot / tot.sum(), color=BLUE, width=0.8)
        ax.axhline(1 / nb, color=INK, lw=1, ls="--")
        ins = np.mean([r["kpi"][k]["inside_range"] for k in KPIS])
        ax.set_title(f"{CITY_NAME[key]}: truth inside the draws' range {ins:.0%} (ideal {r['ideal_inside']:.0%})", fontsize=9, color=INK)
        ax.set_xlabel("rank of the simulator's value among the model's draws (all KPIs, hours 1-8)", fontsize=8, color=INK2)
        ax.set_ylabel("share", fontsize=8, color=INK2)
    fig.suptitle("Is the spread of the imagined futures right? Flat = calibrated; U = too narrow; hump = too wide",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "35_long_rank.png", dpi=130)
    plt.close(fig)
    for key in keys:
        r = allres[key]
        print(key, "drivers", {k: round(v, 3) for k, v in r["drivers"].items()})
        for k in KPIS:
            kk = r["kpi"][k]
            print(f"  {k:16s} err@1h/4h/8h model {kk['model'][12]:.3f}/{kk['model'][48]:.3f}/{kk['model'][-1]:.3f}  "
                  f"pers {kk['persistence'][12]:.3f}/{kk['persistence'][48]:.3f}/{kk['persistence'][-1]:.3f}  "
                  f"hist {kk['history'][12]:.3f}/{kk['history'][48]:.3f}/{kk['history'][-1]:.3f}  inside {kk['inside_range']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["sf_hd", "pune_hd"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    run_city(args.city, args)


if __name__ == "__main__":
    main()
