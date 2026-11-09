"""Stage 7b, step 2: tune driver pay and availability so that "take the first ride at once" is
no longer a driver's best choice.

    .venv/bin/python scripts/41_wm3_market_tune.py

Reads the branched choices of scripts/40_wm3_market_probe.py (data/<city>/wm3_rl/market_probe/
probe_<menu>_s<supply>.pkl): for one driver at one moment, what it earned in the 45 imagined
minutes after declining and after taking each option of its menu, everyone else taking their
first option. Every candidate pay scheme (world/wm3/pay.py), made budget-neutral (drivers paid in
total what the simulator's 'fare minus commission' pays for the same rides), prices the same rides,
so the schemes are compared on identical imagined futures.

For each (menu design, supply, scheme) and city, the gain over ASAP (option 1, the platform's best
for the driver) of a driver who chooses instead by
    rate      the option with the most expected net pay per engaged minute (pickup + trip),
              or declines if even that pays under theta median fares per 30 engaged minutes
    net       the option with the most expected net pay (pay - km cost)
    learned   a value model fitted on other choices (5-fold cross-fitting, so the gain is not
              read off its own outcomes): ridge regression of the branch's value on the option's
              and the driver's features; the driver picks the best predicted action (incl. decline)
    oracle    the best branch in hindsight (an upper bound; it also profits from noise)
Values in median fares; +- is one standard error over choices.
Writes outputs/wm3/stage7b/41_tune.csv, 41_tune.md and 41_tune.png.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from world import cities
from world.wm3 import pay as PY

OUT = cities.OUTPUTS / "wm3" / "stage7b"
CITY = {"sf_hd": "San Francisco", "pune_hd": "Pune"}

# candidate schemes before calibration (scale is set so the total paid equals the baseline's)
SCHEMES = {
    "fare_share": lambda m: PY.baseline(m),
    "surge_x2": lambda m: PY.PayScheme("surge_x2", share=1.0, surge_mult=2.0),
    "surge_flat": lambda m: PY.PayScheme("surge_flat", share=1.0, surge_mult=0.0, surge_flat=2.0),
    "per_trip": lambda m: PY.PayScheme("per_trip", per_trip=1.0, per_km=0.4, per_min=0.4),
    "time_heavy": lambda m: PY.PayScheme("time_heavy", per_km=0.3, per_min=1.5),
    "km_heavy": lambda m: PY.PayScheme("km_heavy", per_km=1.5, per_min=0.0),
    "pickup_paid": lambda m: PY.PayScheme("pickup_paid", share=1.0, pickup_per_min=1.0, pickup_free_min=4.0),
    "pickup_surge": lambda m: PY.PayScheme("pickup_surge", share=1.0, surge_mult=2.0, pickup_per_min=1.0, pickup_free_min=4.0),
}


def load(key):
    d = cities.get(key).data_dir / "wm3_rl" / "market_probe"
    out = {}
    for f in sorted(d.glob("probe_*_s*.pkl")):
        menu, s = f.stem.split("_")[1], float(f.stem.split("_s")[-1])
        out[(menu, s)] = pickle.loads(f.read_bytes())
    return out


def all_trips(recs):
    rows = [t for r in recs for b in r["branches"] for t in b["trips"]]
    return pd.DataFrame(rows)


def option_table(r, scheme, key, market):
    """Expected pay, km cost and engaged minutes of each option on the menu."""
    o = pd.DataFrame(r["options"])
    if not len(o):
        return o
    o["pay"] = PY.pay(scheme, key, market, o["mode"], o.km_in, o.outside_km, o.minutes, o.surge, o.pick_min + 1.0)
    km_all = r["opt"][o.j.to_numpy(), 5] * r["money"] / max(r["kappa"], 1e-9)
    o["cost"] = r["kappa"] * (km_all + o.outside_km)
    o["eng"] = np.maximum(o.pick_min + 1.0 + o.minutes, 1.0)
    o["net"] = o.pay - o.cost
    o["rate"] = o.net / o.eng
    return o


def feats(r, o, a):
    """Features of action a (0 = decline, 1 + j = option j) for the value model."""
    own = list(r["own"])
    if a == 0:
        return [1.0] + [0.0] * 11 + own
    x = o[o.j == a - 1].iloc[0]
    m = r["money"]
    return [0.0, 1.0, x.pay / m, x.cost / m, x.net / m, x.eng / 30, x.rate / m * 30, x.pick_min / 10, a - 1] + \
        list(r["opt"][a - 1, 8:11]) + own


def evaluate(recs, scheme, key, market, thetas=(0.0, 0.5, 1.0, 1.5)):
    rows = []
    for r in recs:
        if len(r["acts"]) < 2:
            continue
        v = np.array([PY.focal_value(b, scheme, key, market, r["kappa"]) for b in r["branches"]]) / r["money"]
        o = option_table(r, scheme, key, market)
        acts = list(r["acts"])
        ix = {a: i for i, a in enumerate(acts)}
        best_rate = int(o.rate.idxmax())
        rec_rate = o.rate.max() / r["money"] * 30          # median fares per 30 engaged min
        row = {"start": r["start"], "seed": r["seed"], "n_opt": len(acts) - 1, "asap": v[ix[1]], "decline": v[ix[0]],
               "oracle": v.max(), "net": v[ix[1 + int(o.loc[o.net.idxmax(), "j"])]],
               "first_best": float(v[ix[1]] >= v.max() - 0.05), "pickups_asap": r["pickups"][ix[1]]}
        for th in thetas:            # decline when even the best option pays under theta median fares per 30 engaged min
            a = 1 + int(o.loc[best_rate, "j"])
            row[f"rate_{th}"] = v[ix[a]] if rec_rate >= th else v[ix[0]]
        row["_X"] = [feats(r, o, a) for a in acts]
        row["_v"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) >= 10:
        df["learned"] = cross_fit(df)
    return df


def cross_fit(df, folds=5, lam=1.0):
    """Ridge value model, 5-fold over choices; returns the value of the action it picks."""
    n = len(df)
    fold = np.arange(n) % folds
    out = np.zeros(n)
    width = max(len(x) for X in df._X for x in X)
    for f in range(folds):
        tr = np.flatnonzero(fold != f)
        X = np.array([x + [0] * (width - len(x)) for i in tr for x in df._X.iloc[i]], float)
        y = np.concatenate([df._v.iloc[i] - df._v.iloc[i].mean() for i in tr])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        w = np.linalg.solve(Z.T @ Z + lam * len(Z) ** 0.5 * np.eye(Z.shape[1]), Z.T @ y)
        for i in np.flatnonzero(fold == f):
            Xi = (np.array([x + [0] * (width - len(x)) for x in df._X.iloc[i]], float) - mu) / sd
            out[i] = df._v.iloc[i][int(np.argmax(Xi @ w))]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in CITY:
        market = cities.get(key).market
        for (menu, s), recs in load(key).items():
            trips = all_trips(recs)
            for name, mk in SCHEMES.items():
                sch = mk(market)
                if name != "fare_share":
                    sch = PY.calibrate(sch, key, market, trips)
                df = evaluate(recs, sch, key, market)
                if not len(df):
                    continue
                base = df.asap
                row = {"city": key, "menu": menu, "supply_cut": s, "scheme": name, "scale": sch.scale, "choices": len(df),
                       "mean_options": df.n_opt.mean(), "asap": base.mean(), "first_is_best": df.first_best.mean()}
                for p in ["decline", "net", "rate_0.0", "rate_1.0", "learned", "oracle"]:
                    if p in df:
                        g = df[p] - base
                        row[f"gain_{p}"], row[f"se_{p}"] = g.mean(), g.std() / np.sqrt(len(g))
                rows.append(row)
                print(f"{key} {menu:9s} cut {s:.1f} {name:12s} n {len(df):3d} opts {df.n_opt.mean():.1f} asap {base.mean():.2f} "
                      f"first best {df.first_best.mean():.2f} | " + " ".join(
                          f"{p} {row.get('gain_' + p, np.nan):+.2f}+-{row.get('se_' + p, np.nan):.2f}"
                          for p in ["decline", "net", "rate_1.0", "learned", "oracle"]), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "41_tune.csv", index=False)
    return res


if __name__ == "__main__":
    main()
