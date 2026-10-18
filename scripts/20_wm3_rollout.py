"""World model 3, stage 3: does the imagined world unfold like the real one?

    .venv/bin/python scripts/20_wm3_rollout.py --tag wm3_base

From real states on test days (several days per city, starts at 07:30, 12:00 and 17:30) the
model imagines the next hour (world/wm3/rollout.py) under the same dispatch policy the simulator
ran, several times with different random draws. Compared with what the simulator actually did:

  city-wide   drivers by status, riders waiting, cumulative new requests, declined quotes,
              pickups, completed trips, cancellations, offers and answers, every 30 s
  drivers     per driver online at the start: still online after 60 min, trips and income in the
              hour (model mean over draws vs truth)
  fairness    spread of driver earnings in the hour (variance, Gini, lowest decile) and rider
              waiting times
Baseline: persistence (state counts stay as at the start; event counts grow at their rate in
the 30 minutes before the start).
Writes outputs/wm3/stage3/20_rollout.json, 20_kpis_<city>.png, 20_errors.png, 20_drivers.png,
20_fairness.png and <city>_imagined_vs_real.html (side-by-side map replay).
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
import torch

from world import cities
from world.wm3 import data as WD
from world.wm3 import features as FT
from world.wm3 import rollout as RO
from world.wm3 import state as S

_spec = importlib.util.spec_from_file_location("tr3", Path(__file__).with_name("18_wm3_train.py"))
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)
OUT = cities.OUTPUTS / "wm3" / "stage3"
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
STARTS = [7.5, 12.0, 17.5]
H = 120                                                     # 60 minutes of 30-s steps
STATE_KPI = ["online", "idle", "to_pickup", "with_rider", "repositioning", "break", "waiting_riders"]
EVENT_KPI = {"new requests": ["request_created", "quote_declined"], "declined quotes": ["quote_declined"],
             "pickups": ["pickup"], "completed trips": ["dropoff"], "cancellations": ["customer_cancelled"],
             "offers": ["offer_sent"], "offers rejected": ["offer_rejected"], "drivers going online": ["driver_online"],
             "drivers going offline": ["driver_offline"]}
INK, INK2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def season_prior(day_dir, cache={}):
    """The drivers' records from the earlier days of the season (as the dataset builder)."""
    days = sorted(p for p in day_dir.parent.iterdir() if p.is_dir() and p.name.startswith("20") and p.name < day_dir.name)
    key = (str(day_dir.parent), day_dir.name)
    if key not in cache:
        pr = FT.Prior()
        for d in days:
            pr.update_light(d, 5000)
        cache[key] = pr
    return cache[key]


def truth_kpis(L, e0):
    """State counts every step and cumulative event counts, from the logs."""
    rows = []
    ev = L.ev
    for h in range(H + 1):
        e = e0 + h
        st = L.status[:, e]
        R = L.open_requests(e)
        t = e * S.DT
        row = {"t": t, "online": int((st != 0).sum()), "idle": int((st == 1).sum()), "to_pickup": int(((st == 2) | (st == 3)).sum()),
               "with_rider": int(((st == 4) | (st == 6)).sum()), "repositioning": int((st == 5).sum()), "break": int((st == 7).sum()),
               "waiting_riders": int((R.status <= 1).sum())}
        w = ev[(ev.t_event > e0 * S.DT) & (ev.t_event <= t)]
        for k, types in EVENT_KPI.items():
            row[k] = int(w.type.isin(types).sum()) if k != "new requests" else int(((L.req.t_request > e0 * S.DT) & (L.req.t_request <= t)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def model_kpis(W, log0):
    row = W.kpis()
    lg = W.log[log0:]
    types = pd.Series([x[1] for x in lg], dtype=object)
    for k, tt in EVENT_KPI.items():
        row[k] = int(types.isin(tt).sum())
    return row


def persistence(L, e0, truth):
    """State counts stay; event counts grow at the rate of the previous 30 minutes."""
    p = truth.copy()
    for k in STATE_KPI:
        p[k] = truth[k].iloc[0]
    t0 = e0 * S.DT
    ev = L.ev[(L.ev.t_event > t0 - 1800) & (L.ev.t_event <= t0)]
    for k, types in EVENT_KPI.items():
        n = int(ev.type.isin(types).sum()) if k != "new requests" else int(((L.req.t_request > t0 - 1800) & (L.req.t_request <= t0)).sum())
        p[k] = n / 1800 * (p.t - t0)
    return p


def driver_outcomes(L, e0, ids):
    e1 = e0 + H
    return pd.DataFrame({"online_end": L.status[ids, e1] != 0, "trips": L.count["trips"][ids, e1] - L.count["trips"][ids, e0],
                         "income": L.income[ids, e1] - L.income[ids, e0]}, index=ids)


def gini(x):
    x = np.sort(np.maximum(np.asarray(x, float), 0))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def run(args):
    model, ck = TR.load(args.tag)
    model.eval()
    res, all_traj, drv_rows, fair_rows, duo = {}, {}, [], [], {}
    for key in WD.CITY_KEYS:
        city = cities.get(key)
        g = S.CityGraph(city)
        rules = RO.CityRules(g)
        meta = json.loads((city.data_dir / WD.SUBDIR / "meta.json").read_text())
        sp = meta["split"]
        test = [d for d in meta["days"] if sp["test_from"] <= d <= sp["test_until"]]
        rng = np.random.default_rng(3)
        days = sorted(rng.choice(test, size=min(args.days, len(test)), replace=False))
        traj = []
        for date in days:
            day_dir = next((city.data_dir / "hd_runs").glob(f"wm2_seed*/{date}"))
            L = S.DayLog(g, day_dir)
            prior = season_prior(day_dir)
            for hs in STARTS:
                e0 = int(hs * 3600 / S.DT)
                truth = truth_kpis(L, e0)
                pers = persistence(L, e0, truth)
                ids = np.flatnonzero(L.status[:, e0] != 0)
                true_d = driver_outcomes(L, e0, ids)
                sims = []
                for m in range(args.samples):
                    t0 = time.time()
                    W = RO.ImaginedWorld(model, rules, L, prior, e0, RO.policy_for(L, seed=m), seed=100 + m)
                    inc0, trips0 = W.income.copy(), W.count["trips"].copy()
                    log0 = len(W.log)
                    rows = [model_kpis(W, log0)]
                    keep = args.duo and date == days[0] and hs == STARTS[0] and m == 0
                    tracks = []
                    for h in range(H):
                        W.step()
                        rows.append(model_kpis(W, log0))
                        if keep:
                            pos = W.positions(W.t)[0]
                            tracks.append((pos.copy(), W.st.copy(), [(q["origin"], q["status"]) for q in W.req.values()]))
                    sims.append(pd.DataFrame(rows))
                    drv_rows.append(pd.DataFrame({"key": key, "date": date, "start": hs, "sample": m, "driver": ids,
                                                  "online_end": W.st[ids] != 0, "trips": W.count["trips"][ids] - trips0[ids],
                                                  "income": W.income[ids] - inc0[ids]}))
                    if keep:
                        duo[key] = {"L": L, "e0": e0, "tracks": tracks, "g": g}
                    print(f"  {key} {date} {hs:4.1f}h draw {m}: {time.time() - t0:.0f}s  "
                          f"pickups {rows[-1]['pickups']} (true {truth['pickups'].iloc[-1]}), online {rows[-1]['online']} "
                          f"(true {truth['online'].iloc[-1]}), waiting {rows[-1]['waiting_riders']} (true {truth['waiting_riders'].iloc[-1]})", flush=True)
                traj.append({"date": date, "start": hs, "truth": truth, "pers": pers, "sims": sims})
                td = true_d.assign(key=key, date=date, start=hs, sample=-1, driver=ids)
                drv_rows.append(td.reset_index(drop=True))
                # fairness over drivers online at the start
                for m, sdf in enumerate(sims):
                    pass
        all_traj[key] = traj
        res[key] = summarize(traj)
    drv = pd.concat(drv_rows, ignore_index=True)
    return res, all_traj, drv, duo


def summarize(traj):
    """Mean absolute error at 15/30/60 min per KPI: model (mean of draws) vs persistence, and
    whether the truth lies within the range of the draws."""
    out = {}
    for k in STATE_KPI + list(EVENT_KPI):
        for hz in (30, 60, 120):
            tv = np.array([tr["truth"][k].iloc[hz] for tr in traj], float)
            mv = np.array([np.mean([s[k].iloc[hz] for s in tr["sims"]]) for tr in traj])
            pv = np.array([tr["pers"][k].iloc[hz] for tr in traj], float)
            lo = np.array([min(s[k].iloc[hz] for s in tr["sims"]) for tr in traj])
            hi = np.array([max(s[k].iloc[hz] for s in tr["sims"]) for tr in traj])
            scale = max(np.mean(np.abs(tv)), 1)
            out[f"{k}@{hz // 2}min"] = {"model_rel_err": float(np.mean(np.abs(mv - tv)) / scale),
                                        "persistence_rel_err": float(np.mean(np.abs(pv - tv)) / scale),
                                        "truth_in_range": float(np.mean((tv >= lo) & (tv <= hi))), "truth_mean": float(np.mean(tv))}
    return out


# ---------------------------------------------------------------------- plots
def plot_kpis(key, traj, path):
    ks = STATE_KPI + ["new requests", "pickups", "completed trips", "cancellations", "drivers going online", "drivers going offline"]
    tr = traj[0]
    fig, axs = plt.subplots(len(STARTS), len(ks), figsize=(2.3 * len(ks), 2.5 * len(STARTS)), constrained_layout=True)
    for r, tr in enumerate([t for t in traj if t["date"] == traj[0]["date"]]):
        m = np.arange(H + 1) / 2
        for c, k in enumerate(ks):
            ax = axs[r, c]
            for s in tr["sims"]:
                ax.plot(m, s[k], color=BLUE, lw=.8, alpha=.55)
            ax.plot(m, tr["pers"][k], color=INK2, lw=1, ls="--")
            ax.plot(m, tr["truth"][k], color=INK, lw=1.8)
            ax.set_title(f"{k} ({tr['start']:.1f} h)" if c == 0 else k, fontsize=7.5)
            ax.tick_params(labelsize=6.5); ax.grid(color=GRID, lw=.4); ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].plot([], [], color=INK, lw=1.8, label="simulator (truth)")
    axs[0, 0].plot([], [], color=BLUE, lw=.8, label="imagined by the model (each draw)")
    axs[0, 0].plot([], [], color=INK2, lw=1, ls="--", label="persistence")
    axs[0, 0].legend(fontsize=6.5, loc="lower left", frameon=False)
    fig.suptitle(f"{CITY_NAME[key]}, {traj[0]['date']}: the next hour imagined by world model 3 vs the simulator "
                 "(state counts and cumulative event counts; x = minutes after the start)", fontsize=10)
    fig.savefig(path, dpi=110); plt.close(fig)


def plot_errors(res, path):
    ks = STATE_KPI + list(EVENT_KPI)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4), constrained_layout=True)
    for ax, key in zip(axs, WD.CITY_KEYS):
        r = res[key]
        y = np.arange(len(ks))
        for j, (hz, col) in enumerate(((30, "#9ec5f0"), (60, BLUE))):
            mv = [100 * r[f"{k}@{hz}min"]["model_rel_err"] for k in ks]
            pv = [100 * r[f"{k}@{hz}min"]["persistence_rel_err"] for k in ks]
            ax.barh(y + (j - .5) * .38, mv, height=.36, color=col, label=f"model, {hz} min")
            ax.scatter(pv, y + (j - .5) * .38, marker="|", s=90, color=INK, zorder=4, label=f"persistence, {hz} min" if j else None)
        ax.set_yticks(y); ax.set_yticklabels(ks, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("mean absolute error, % of the true value", fontsize=8.5)
        ax.set_title(f"{CITY_NAME[key]}: error of the imagined hour (bars) vs persistence (black ticks)", fontsize=9.5)
        ax.legend(fontsize=7.5, frameon=False); ax.grid(axis="x", color=GRID, lw=.5)
        ax.spines[["top", "right"]].set_visible(False); ax.set_xlim(0, 100)
    fig.suptitle("World model 3 rollouts on test days: error after 30 and 60 minutes, all starts and days", fontsize=11)
    fig.savefig(path, dpi=115); plt.close(fig)


def plot_drivers(drv, path):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    out = {}
    for i, key in enumerate(WD.CITY_KEYS):
        d = drv[drv.key == key]
        tr = d[d["sample"] < 0].set_index(["date", "start", "driver"])
        md = d[d["sample"] >= 0].groupby(["date", "start", "driver"])[["online_end", "trips", "income"]].mean()
        j = tr.join(md, rsuffix="_m", how="inner")
        res = {}
        for c, (col, name) in enumerate((("online_end", "still online after 60 min"), ("trips", "trips in the hour"), ("income", "income in the hour"))):
            ax = axs[i, c]
            x, y = j[f"{col}_m"].astype(float), j[col].astype(float)
            if col == "online_end":
                bins = np.linspace(0, 1, 11)
                b = np.clip(np.digitize(x, bins) - 1, 0, 9)
                cal = [(x[b == k].mean(), y[b == k].mean(), (b == k).sum()) for k in range(10) if (b == k).any()]
                cal = np.array(cal)
                ax.plot([0, 1], [0, 1], color=GRID); ax.plot(cal[:, 0], cal[:, 1], "o-", color=BLUE, ms=4)
                ax.set_xlabel("predicted share of draws still online", fontsize=8); ax.set_ylabel("truly still online", fontsize=8)
                from sklearn.metrics import roc_auc_score
                res["online_auc"] = float(roc_auc_score(y, x)) if 0 < y.mean() < 1 else None
                ax.set_title(f"{CITY_NAME[key]}: {name} (AUC {res['online_auc']:.2f}, n = {len(x):,})", fontsize=9)
            else:
                ax.scatter(x + np.random.default_rng(0).normal(0, .03, len(x)), y, s=4, alpha=.25, color=BLUE)
                r = float(np.corrcoef(x, y)[0, 1])
                res[f"{col}_r"] = r
                lim = np.percentile(np.r_[x, y], 99.5)
                ax.plot([0, lim], [0, lim], color=GRID)
                ax.set_xlim(0, lim); ax.set_ylim(0, lim)
                ax.set_xlabel("predicted (mean over draws)", fontsize=8); ax.set_ylabel("simulator", fontsize=8)
                ax.set_title(f"{CITY_NAME[key]}: {name} per driver (r = {r:.2f})", fontsize=9)
            ax.tick_params(labelsize=7.5); ax.grid(color=GRID, lw=.4); ax.spines[["top", "right"]].set_visible(False)
        out[key] = res
    fig.suptitle("Each driver online at the start: what the imagined hour predicts for that driver vs what the simulator did", fontsize=11)
    fig.savefig(path, dpi=115); plt.close(fig)
    return out


def plot_fairness(drv, path):
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.3), constrained_layout=True)
    out = {}
    for i, key in enumerate(WD.CITY_KEYS):
        d = drv[drv.key == key]
        rows = []
        for (date, st, smp), grp in d.groupby(["date", "start", "sample"]):
            inc = grp.income.to_numpy()
            rows.append({"date": date, "start": st, "sample": smp, "gini": gini(inc), "sd": float(inc.std()),
                         "p10": float(np.percentile(inc, 10)), "mean": float(inc.mean()), "zero_share": float((inc <= 0).mean())})
        f = pd.DataFrame(rows)
        tr = f[f["sample"] < 0].set_index(["date", "start"])
        md = f[f["sample"] >= 0].groupby(["date", "start"]).mean(numeric_only=True)
        j = tr.join(md, rsuffix="_m")
        for c, (col, name) in enumerate((("gini", "Gini of hourly income"), ("zero_share", "share of drivers earning nothing"))):
            ax = axs[2 * i + c]
            ax.scatter(j[f"{col}_m"], j[col], s=30, color=BLUE)
            lim = (min(j[col].min(), j[f"{col}_m"].min()) * .9, max(j[col].max(), j[f"{col}_m"].max()) * 1.1)
            ax.plot(lim, lim, color=GRID); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("imagined (mean over draws)", fontsize=8); ax.set_ylabel("simulator", fontsize=8)
            err = float(np.mean(np.abs(j[f"{col}_m"] - j[col])))
            ax.set_title(f"{CITY_NAME[key]}: {name}\nper start (mean abs. error {err:.3f})", fontsize=9)
            ax.tick_params(labelsize=7.5); ax.grid(color=GRID, lw=.4); ax.spines[["top", "right"]].set_visible(False)
            out[f"{key}/{col}_mae"] = err
            out[f"{key}/{col}_truth_mean"] = float(j[col].mean())
    fig.suptitle("Driver fairness in the imagined hour vs the simulator (each dot = one start on a test day)", fontsize=11)
    fig.savefig(path, dpi=115); plt.close(fig)
    return out


# ---------------------------------------------------------------- duo viewer
def write_duo(key, d, path):
    import base64
    L, e0, tracks, g = d["L"], d["e0"], d["tracks"], d["g"]
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()  # noqa: E731
    x0, y0 = g.x.min(), g.y.min()
    q = lambda v: np.clip(np.round(v / 2.0), 0, 65534).astype(np.uint16)  # noqa: E731
    ids = np.flatnonzero((L.status[:, e0:e0 + H + 1] != 0).any(1) | np.array([(st != 0) for _, st, _ in tracks]).any(0))
    n, ne = len(ids), H + 1
    tx, ty = np.full((n, ne), 65535, np.uint16), np.full((n, ne), 65535, np.uint16)
    mx, my = tx.copy(), ty.copy()
    ts, ms = np.zeros((n, ne), np.int8), np.zeros((n, ne), np.int8)
    W0 = RO.ImaginedWorld  # noqa: F841
    for k in range(ne):
        e = e0 + k
        xy = L.xy(e)[ids]
        st = L.status[ids, e]
        m = st != 0
        tx[m, k], ty[m, k] = q(xy[m, 0] - x0), q(xy[m, 1] - y0)
        ts[:, k] = st
        if k:
            pos, mst, _ = tracks[k - 1]
            mm = mst[ids] != 0
            mx[mm, k], my[mm, k] = q(pos[ids][mm, 0] - x0), q(pos[ids][mm, 1] - y0)
            ms[:, k] = mst[ids]
    mx[:, 0], my[:, 0], ms[:, 0] = tx[:, 0], ty[:, 0], ts[:, 0]
    # waiting riders
    def riders_truth(k):
        R = L.open_requests(e0 + k)
        o = R.origin[R.status <= 1].to_numpy()
        return o
    rt = [riders_truth(k) for k in range(ne)]
    rm = [rt[0]] + [np.array([o for o, s in tr[2] if s in ("pending", "offered")], int) for tr in tracks]
    pack = lambda lst: {"off": b64(np.r_[0, np.cumsum([len(a) for a in lst])].astype(np.int32)),  # noqa: E731
                        "node": b64(np.concatenate(lst).astype(np.int32) if sum(len(a) for a in lst) else np.zeros(0, np.int32))}
    cls = pd.Series(g.road_class).map({"motorway": 3, "trunk": 3, "primary": 2, "secondary": 1}).fillna(0).to_numpy(np.uint8)
    data = {"city": CITY_NAME[key], "date": str(L.date.date()), "w0": e0 * S.DT, "dt": S.DT, "ne": ne, "unit": 2.0,
            "nodes": {"x": b64(q(g.x - x0)), "y": b64(q(g.y - y0))}, "edges": {"u": b64(g.eu.astype(np.uint32)), "v": b64(g.ev.astype(np.uint32)), "cls": b64(cls)},
            "truth": {"x": b64(tx), "y": b64(ty), "st": b64(ts), "riders": pack(rt)},
            "model": {"x": b64(mx), "y": b64(my), "st": b64(ms), "riders": pack(rm)}, "n": int(n)}
    tpl = (Path(__file__).resolve().parents[1] / "world" / "wm3" / "duo.html").read_text()
    path.write_text(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wm3_base")
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--duo", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res, traj, drv, duo = run(args)
    for key in WD.CITY_KEYS:
        plot_kpis(key, traj[key], OUT / f"20_kpis_{key}.png")
    plot_errors(res, OUT / "20_errors.png")
    res["drivers"] = plot_drivers(drv, OUT / "20_drivers.png")
    res["fairness"] = plot_fairness(drv, OUT / "20_fairness.png")
    drv.to_csv(OUT / "20_drivers.csv", index=False)
    for key, d in duo.items():
        write_duo(key, d, OUT / f"{key}_imagined_vs_real.html")
    (OUT / "20_rollout.json").write_text(json.dumps(res, indent=1, default=float))
    for key in WD.CITY_KEYS:
        print(key, {k: (round(v["model_rel_err"], 3), round(v["persistence_rel_err"], 3)) for k, v in res[key].items() if k.endswith("@60min")})
    print(json.dumps({k: res[k] for k in ("drivers", "fairness")}, indent=1))


if __name__ == "__main__":
    main()
