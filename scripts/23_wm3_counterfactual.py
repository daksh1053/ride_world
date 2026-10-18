"""World model 3, stage 4b: does the model respond to a platform action the way the world does?

    .venv/bin/python scripts/23_wm3_counterfactual.py --tag wm3_v2

Uses the twin runs of scripts/22_wm3_twins.py: a base morning and copies with an extra order of
n idle drivers to one zone, identical until the order. The true effect of the order is the
difference between a copy and the base, step by step for 90 minutes. The model's effect: from
the base state just before the order it imagines the next 90 minutes twice, with the platform's
usual actions and with the same order added, sharing random numbers (common random numbers),
several draws; the effect is the mean difference. Measured, cf minus base:
  idle drivers in the target zone, idle drivers in the source zones, drivers repositioning to
  the target, pickups starting in the target zone and in the source zones, all pickups,
  completed trips, cancellations, riders waiting; for the ordered drivers: how many followed,
  their trips and net income in the 90 minutes.
Baseline: no effect (what a model without actions predicts), and for the followers the
population-average compliance.
Writes outputs/wm3/stage4/23_effects.png, 23_scatter.png, 23_followers.png, 23_counterfactual.json,
23_pairs.csv.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from world import cities
from world.policies import Action
from world.wm3 import features as FT
from world.wm3 import rollout as RO
from world.wm3 import state as S

_spec = importlib.util.spec_from_file_location("tr3", Path(__file__).with_name("18_wm3_train.py"))
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)
OUT = cities.OUTPUTS / "wm3" / "stage4"
FOLDER = "wm3_twins"
H = 180                                                    # 90 minutes of 30-s steps
KPI = ["tgt_idle", "src_idle", "to_tgt", "tgt_pickups", "src_pickups", "pickups", "dropoffs", "cancels", "waiting"]
LABEL = {"tgt_idle": "idle drivers in the target zone", "src_idle": "idle drivers in the source zones",
         "to_tgt": "drivers repositioning to the target", "tgt_pickups": "pickups in the target zone (cum.)",
         "src_pickups": "pickups in the source zones (cum.)", "pickups": "all pickups (cum.)",
         "dropoffs": "completed trips (cum.)", "cancels": "cancellations (cum.)", "waiting": "riders waiting"}
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
INK, INK2, GRID, BLUE, ORANGE, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#898781"


def truth_kpis(L, e0, tgt, src):
    g = L.g
    rows = []
    ev = L.ev
    pk = ev[ev.type == "pickup"]
    pz = L.req.set_index("request_id").origin_zone.reindex(pk.request_id).to_numpy()
    for h in range(H + 1):
        e = e0 + h
        t = e * S.DT
        st = L.status[:, e]
        z = g.zone[L.stop[:, e]]
        leg = L.leg[:, e]
        rep = (leg >= 0) & np.isin(L.leg_kind[np.maximum(leg, 0)], [3, 4])
        rep_tgt = rep & (g.zone[L.leg_dest[np.maximum(leg, 0)]] == tgt)
        win = (pk.t_event > e0 * S.DT).to_numpy() & (pk.t_event <= t).to_numpy()
        w = ev[(ev.t_event > e0 * S.DT) & (ev.t_event <= t)]
        R = L.open_requests(e)
        rows.append({"tgt_idle": int(((st == 1) & (z == tgt)).sum()), "src_idle": int(((st == 1) & np.isin(z, src)).sum()),
                     "to_tgt": int(rep_tgt.sum()), "tgt_pickups": int((win & (pz == tgt)).sum()),
                     "src_pickups": int((win & np.isin(pz, src)).sum()), "pickups": int(win.sum()),
                     "dropoffs": int((w.type == "dropoff").sum()), "cancels": int((w.type == "customer_cancelled").sum()),
                     "waiting": int((R.status <= 1).sum())})
    return pd.DataFrame(rows)


def model_row(W, log0, tgt, src):
    g = W.g
    z = g.zone[W.node]
    lg = W.log[log0:]
    pk = [(x[3]) for x in lg if x[1] == "pickup"]
    pz = np.array([g.zone[W.req_origin.get(k, 0)] for k in pk]) if pk else np.zeros(0, int)
    types = pd.Series([x[1] for x in lg], dtype=object)
    return {"tgt_idle": int(((W.st == RO.IDLE) & (z == tgt)).sum()), "src_idle": int(((W.st == RO.IDLE) & np.isin(z, src)).sum()),
            "to_tgt": int(((W.st == RO.REPO) & (W.repo_zone == tgt) & ~W.going_home).sum()),
            "tgt_pickups": int((pz == tgt).sum()), "src_pickups": int(np.isin(pz, src).sum()), "pickups": len(pk),
            "dropoffs": int((types == "dropoff").sum()), "cancels": int((types == "customer_cancelled").sum()),
            "waiting": int(sum(q["status"] in ("pending", "offered") for q in W.req.values()))}


class Tracked(RO.ImaginedWorld):
    """ImaginedWorld that remembers each request's origin (requests leave W.req when served)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.req_origin = {k: q["origin"] for k, q in self.req.items()}

    def _on_request(self, t, z, m, origin, dz, dnode, ext, hZ, k, push, t1):
        super()._on_request(t, z, m, origin, dz, dnode, ext, hZ, k, push, t1)
        self.req_origin[k] = origin


def imagine(model, rules, L, prior, e0, order, tgt, src, movers, seed):
    """One draw: (kpis, follower count, movers' trips, movers' income) for this arm."""
    W = Tracked(model, rules, L, prior, e0, RO.policy_for(L, seed=seed), seed=seed)
    inc0, trips0 = W.income.copy(), W.count["trips"].copy()
    log0 = len(W.log)
    rows = [model_row(W, log0, tgt, src)]
    for h in range(H):
        act = None
        if h == 0 and order:
            act = W.policy.act(W.observe())
            busy = {d for d, _ in act.offers} | {d for d, _ in act.dispatches}
            act = Action(offers=act.offers, dispatches=act.dispatches + [(int(d), int(tgt)) for d in movers if d not in busy],
                         info=act.info)
        W.step(act)
        rows.append(model_row(W, log0, tgt, src))
    lg = pd.DataFrame(W.log[log0:], columns=["t", "type", "d", "k", "z", "v"])
    mv = np.asarray(movers, int)
    fset = set(lg[(lg.type == "dispatch_accepted") & lg.d.isin(mv) & (lg.z == tgt)].d.astype(int))
    imagine.followed = fset
    imagine.order_p = dict(W.order_p)
    fol = len(fset)
    return pd.DataFrame(rows), fol, float((W.count["trips"][mv] - trips0[mv]).sum()), float((W.income[mv] - inc0[mv]).sum())


def pair_inputs(key, rec, arm):
    city = cities.get(key)
    root = city.data_dir / FOLDER
    g = S.CityGraph(city) if key not in pair_inputs.g else pair_inputs.g[key]
    pair_inputs.g[key] = g
    Lb = S.DayLog(g, root / f"seed{rec['seed']}" / rec["date"])
    Lc = S.DayLog(g, root / f"seed{rec['seed']}_{arm['arm']}" / rec["date"])
    t = rec["t_order"]
    sb = Lb.ev[(Lb.ev.type == "dispatch_sent") & (Lb.ev.t_event == t)].driver_id
    sc = Lc.ev[(Lc.ev.type == "dispatch_sent") & (Lc.ev.t_event == t)]
    movers = sorted(set(sc[sc.value == arm["dst_zone"]].driver_id) - set(sb))
    return g, Lb, Lc, movers


pair_inputs.g = {}


def run(args):
    model, ck = TR.load(args.tag)
    model.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curves, drivers = [], [], []
    for key in args.cities:
        city = cities.get(key)
        rules = None
        for rf in sorted((city.data_dir / FOLDER).glob("rec_seed*.json")):
            rec = json.loads(rf.read_text())
            for arm in rec["arms"]:
                if not arm.get("n_moved"):
                    continue
                t0 = time.time()
                g, Lb, Lc, movers = pair_inputs(key, rec, arm)
                rules = rules or RO.CityRules(g)
                prior = FT.Prior()
                base_dir = city.data_dir / FOLDER / f"seed{rec['seed']}"
                for d in sorted(p for p in base_dir.iterdir() if p.is_dir() and p.name < rec["date"]):
                    prior.update_light(d, Lb.nD)
                e0 = int(rec["t_order"] // S.DT)
                tgt, src = int(arm["dst_zone"]), list(arm["src_zones"])
                tb, tc = truth_kpis(Lb, e0, tgt, src), truth_kpis(Lc, e0, tgt, src)
                # truth for the movers
                evc = Lc.ev[(Lc.ev.t_event > rec["t_order"]) & (Lc.ev.t_event <= rec["t_order"] + 60)]
                t_fol = int(((evc.type == "dispatch_accepted") & evc.driver_id.isin(movers)).sum())
                e1 = e0 + H
                t_trips = [float((L.count["trips"][movers, e1] - L.count["trips"][movers, e0]).sum()) for L in (Lb, Lc)]
                t_inc = [float((L.income[movers, e1] - L.income[movers, e0]).sum()) for L in (Lb, Lc)]
                comp = Lb.latent.set_index("driver_id").reposition_compliance.reindex(movers).to_numpy()
                diffs, fols, mtr, minc = [], [], [], []
                f_cnt = {d: 0 for d in movers}
                for m in range(args.draws):
                    kb, _, trb, inb = imagine(model, rules, Lb, prior, e0, False, tgt, src, movers, 100 + m)
                    kc, fol, trc, inc = imagine(model, rules, Lb, prior, e0, True, tgt, src, movers, 100 + m)
                    for d in imagine.followed:
                        f_cnt[d] += 1
                    p_follow = imagine.order_p
                    diffs.append(kc - kb); fols.append(fol); mtr.append(trc - trb); minc.append(inc - inb)
                md = sum(diffs) / len(diffs)
                t_set = set(evc[(evc.type == "dispatch_accepted") & evc.driver_id.isin(movers)].driver_id.astype(int))
                hist = Lb.count["orders"][movers, e0].astype(float)
                obs = (Lb.count["orders"][movers, e0] - Lb.count["orders_declined"][movers, e0] + prior_follow(prior, movers)) / (hist + 1.0)
                for j, d in enumerate(movers):
                    drivers.append({"key": key, "seed": rec["seed"], "arm": arm["arm"], "driver": int(d), "followed": int(d in t_set),
                                    "p_model": p_follow.get(int(d), np.nan), "model_follow_share": f_cnt[d] / args.draws, "trait": float(comp[j]), "observed_rate": float(obs[j])})
                td = tc - tb
                curves.append({"key": key, "seed": rec["seed"], "arm": arm["arm"], "n": len(movers), "truth": td, "model": md,
                               "draws": diffs})
                row = {"key": key, "seed": rec["seed"], "date": rec["date"], "arm": arm["arm"], "n_moved": len(movers),
                       "t_order_h": rec["t_order"] / 3600, "travel_min": arm["travel_s"] / 60, "tgt": tgt,
                       "followers_true": t_fol, "followers_model": float(np.mean(fols)),
                       "followers_trait": float(np.nansum(comp)), "followers_avg_rate": len(movers) * 0.735,
                       "mover_trips_true": t_trips[1] - t_trips[0], "mover_trips_model": float(np.mean(mtr)),
                       "mover_income_true": t_inc[1] - t_inc[0], "mover_income_model": float(np.mean(minc))}
                for k in KPI:
                    for h in (60, 120, 180):
                        row[f"{k}@{h // 2}_true"] = float(td[k].iloc[h])
                        row[f"{k}@{h // 2}_model"] = float(md[k].iloc[h])
                        row[f"{k}@{h // 2}_model_sd"] = float(np.std([d[k].iloc[h] for d in diffs]))
                rows.append(row)
                print(f"  {key} seed {rec['seed']} {arm['arm']} ({len(movers)} moved, {arm['travel_s'] / 60:.0f} min): "
                      f"followers {t_fol} vs model {np.mean(fols):.1f}; target idle @30min {td.tgt_idle.iloc[60]:+d} vs "
                      f"{md.tgt_idle.iloc[60]:+.1f}; src idle {td.src_idle.iloc[60]:+d} vs {md.src_idle.iloc[60]:+.1f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "23_pairs.csv", index=False)
    dr = pd.DataFrame(drivers)
    dr.to_csv(OUT / "23_drivers.csv", index=False)
    return df, curves, dr


def prior_follow(prior, ids):
    """Follow share from earlier days (the Prior), as a pseudo-count for today's rate."""
    try:
        a = prior.a
        o, f = a["orders"][ids], a["followed"][ids]
        return np.where(o > 0, f / np.maximum(o, 1), 0.735)
    except Exception:
        return np.full(len(ids), 0.735)


def auc(y, x):
    y, x = np.asarray(y), np.asarray(x, float)
    if y.min() == y.max():
        return None
    r = pd.Series(x).rank().to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def summarise(df, dr=None):
    res = {}
    for key, d in df.groupby("key"):
        r = {"pairs": len(d)}
        for k in KPI + ["followers", "mover_trips", "mover_income"]:
            for h in ([30, 60, 90] if k in KPI else [None]):
                col = f"{k}@{h}" if h else k
                y, x = d[f"{col}_true"].to_numpy(), d[f"{col}_model"].to_numpy()
                r[col] = {"r": float(np.corrcoef(x, y)[0, 1]) if len(d) > 2 and x.std() > 0 and y.std() > 0 else None,
                          "mae_model": float(np.abs(x - y).mean()), "mae_zero": float(np.abs(y).mean()),
                          "sign_agree": float((np.sign(x) == np.sign(y)).mean()), "mean_true": float(y.mean()),
                          "mean_model": float(x.mean())}
        r["followers_trait_mae"] = float(np.abs(d.followers_trait - d.followers_true).mean())
        r["followers_avg_rate_mae"] = float(np.abs(d.followers_avg_rate - d.followers_true).mean())
        if dr is not None and len(dr):
            q = dr[dr.key == key]
            q = q[q.p_model.notna()]
            r["per_driver_follow_auc"] = {"model": auc(q.followed, q.p_model), "hidden_trait": auc(q.followed, q.trait),
                                          "observed_rate": auc(q.followed, q.observed_rate), "drivers": len(q)}
        res[key] = r
    return res


def _ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_effects(curves, path):
    """Effect over time for the largest order of each seed: truth vs model (mean and draws)."""
    big = {}
    for c in curves:
        k = (c["key"], c["seed"])
        if k not in big or c["n"] > big[k]["n"]:
            big[k] = c
    sel = sorted(big.values(), key=lambda c: (c["key"], c["seed"]))[:8]
    ks = ["tgt_idle", "src_idle", "to_tgt", "tgt_pickups", "src_pickups", "pickups"]
    fig, axs = plt.subplots(len(sel), len(ks), figsize=(3.1 * len(ks), 2.1 * len(sel)), squeeze=False)
    x = np.arange(H + 1) / 2
    for i, c in enumerate(sel):
        for j, k in enumerate(ks):
            ax = axs[i, j]
            for d in c["draws"]:
                ax.plot(x, d[k], color=BLUE, lw=0.5, alpha=0.3)
            ax.plot(x, c["model"][k], color=BLUE, lw=2, label="model (mean of draws)")
            ax.plot(x, c["truth"][k], color=INK, lw=1.6, label="simulator twins")
            ax.axhline(0, color=GREY, lw=0.8, ls="--")
            _ax(ax)
            if i == 0:
                ax.set_title(LABEL[k], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{CITY_NAME[c['key']]} seed {c['seed']}\n{c['n']} moved", fontsize=8, color=INK2)
            if i == len(sel) - 1:
                ax.set_xlabel("minutes after the order", fontsize=8, color=INK2)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Effect of one repositioning order (with order minus without), simulator twins vs world model 3", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_scatter(df, path):
    ks = ["tgt_idle", "src_idle", "to_tgt", "tgt_pickups", "src_pickups", "pickups", "cancels", "mover_income"]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4))
    for ax, k in zip(axs.flat, ks):
        col = k if k == "mover_income" else f"{k}@60" if k not in ("tgt_idle", "src_idle", "to_tgt") else f"{k}@30"
        for key, colr in (("sf_hd", BLUE), ("pune_hd", ORANGE)):
            d = df[df.key == key]
            ax.scatter(d[f"{col}_true"], d[f"{col}_model"], s=28, color=colr, label=CITY_NAME[key], edgecolor="white", lw=0.6)
        lo = min(df[f"{col}_true"].min(), df[f"{col}_model"].min())
        hi = max(df[f"{col}_true"].max(), df[f"{col}_model"].max())
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8)
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        _ax(ax)
        r = np.corrcoef(df[f"{col}_true"], df[f"{col}_model"])[0, 1]
        name = "ordered drivers' net income in 90 min" if k == "mover_income" else LABEL[k]
        ax.set_title(f"{name}\n{'at +' + col.split('@')[1] + ' min, ' if '@' in col else ''}r = {r:.2f}", fontsize=9, color=INK)
        ax.set_xlabel("true effect (twins)", fontsize=8, color=INK2)
        ax.set_ylabel("model effect", fontsize=8, color=INK2)
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Every twin pair: effect of the order in the simulator vs imagined by the model (diagonal = exact)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_followers(df, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (col, name) in zip(axs, [("followers_model", "world model 3 (reads each driver's history)"),
                                      ("followers_avg_rate", "population-average compliance (73.5%)")]):
        for key, colr in (("sf_hd", BLUE), ("pune_hd", ORANGE)):
            d = df[df.key == key]
            ax.scatter(d.followers_true, d[col], s=30, color=colr, label=CITY_NAME[key], edgecolor="white", lw=0.6)
        hi = max(df.followers_true.max(), df[col].max()) * 1.05
        ax.plot([0, hi], [0, hi], color=GREY, lw=0.8)
        _ax(ax)
        mae = np.abs(df[col] - df.followers_true).mean()
        ax.set_title(f"{name}\nmean abs. error {mae:.1f} drivers", fontsize=10, color=INK)
        ax.set_xlabel("ordered drivers who followed (simulator)", fontsize=9, color=INK2)
        ax.set_ylabel("predicted followers", fontsize=9, color=INK2)
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("How many of the ordered drivers follow the order", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wm3_v2")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--cities", nargs="+", default=["sf_hd", "pune_hd"])
    args = ap.parse_args()
    df, curves, dr = run(args)
    res = summarise(df, dr)
    (OUT / "23_counterfactual.json").write_text(json.dumps(res, indent=1))
    plot_effects(curves, OUT / "23_effects.png")
    plot_scatter(df, OUT / "23_scatter.png")
    plot_followers(df, OUT / "23_followers.png")
    for key, r in res.items():
        print(key, {k: (round(v["r"], 2) if v["r"] is not None else None, round(v["mae_model"], 1), round(v["mae_zero"], 1))
                    for k, v in r.items() if isinstance(v, dict) and "r" in v and ("@30" in k or "@60" in k or "@" not in k)})
        print(key, "per-driver follow AUC", r.get("per_driver_follow_auc"))
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
