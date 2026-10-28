"""World model 3, stage 8b: shocks the model never saw -- does it predict what the world does?

    .venv/bin/python scripts/36_wm3_shock.py --city sf_hd --seed0 700 --n-seeds 20
    .venv/bin/python scripts/36_wm3_shock.py --report

A world model should answer "what if" for interventions outside its training data, not only for
actions the platform usually takes. Three shocks, each at the start of a 2-h window (8, 12 or 17 h)
on the stage-5 test seasons (fresh drivers and riders, test days; data/<city>/wm3_rl/exec):
  supply30 / supply60   30% / 60% of the drivers who are idle or on a break at that moment log off
                        and stay off (an app outage, a strike); no training day had anything similar
  price15               every zone's surge is at least 1.5 for the window (a price rise everywhere;
                        the training days had surge > 1 only in busy zones)
Real: the simulator runs the day again from the same warm season with the shock; until the shock
it is identical to the stage-5 'rule' run (checked), which is the base. Imagined: from the base
run's state at the window start the world model imagines the window with and without the same shock
(4 draws, the same random numbers per draw in every arm). Effect = shock minus base, compared.

Writes data/<city>/wm3_rl/exec/shock_seed<k>.json and outputs/wm3/stage8/36_*.png / 36_shock.json.
"""

from __future__ import annotations

import argparse
import copy
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
from world.hd import sim as HS
from world.wm3 import agent as AG
from world.wm3 import features as FT
from world.wm3 import rollout as RO
from world.wm3 import state as S

_spec = importlib.util.spec_from_file_location("ex27", Path(__file__).with_name("27_wm3_rl_execute.py"))
EX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EX)
_quiet = EX._quiet                                   # the pickled warm seasons refer to __main__._quiet
TR = EX.TR
OUT = cities.OUTPUTS / "wm3" / "stage8"
FOLDER = "wm3_rl/exec"
ARMS = {"supply30": ("supply", 0.3), "supply60": ("supply", 0.6), "price15": ("price", 1.5)}
WINDOW_H = 2.0
DRAWS = 4
EVERY = 20                                            # trajectory every 10 min
TRAJ = ["online", "idle", "waiting", "pickups", "cancels", "declined"]
METRICS = ["pickups", "cancels", "expired", "declined", "wait_min", "online_mean", "waiting_mean", "income", "net_per_h"]
LABEL = {"pickups": "pickups", "cancels": "rider cancellations", "expired": "requests expired", "declined": "quotes declined",
         "wait_min": "rider wait (min)", "online_mean": "drivers online (mean)", "waiting_mean": "riders waiting (mean)",
         "income": "drivers' income", "net_per_h": "net per online hour"}
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
INK, INK2, GRID, BLUE, ORANGE, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#9a9893"


def removed_set(seed, frac, cand):
    g = np.random.default_rng(seed * 101 + int(frac * 100))
    return sorted(int(d) for d in cand if g.random() < frac)


class ShockPolicy(AG.LiveAgentPolicy):
    """The platform's rule (as the stage-5 'rule' arm) plus a shock at t_start."""

    def __init__(self, *a, shock=("supply", 0.0), **kw):
        super().__init__(None, *a, mode="rule", match="rule", **kw)
        self.shock, self.done, self.banned, self.record = shock, False, set(), {}

    def act(self, obs):
        if not self.done and obs.t >= self.t_start:
            self.done = True
            w = self.world
            kind, x = self.shock
            if kind == "supply":
                cand = np.flatnonzero((w.d_status == HS.IDLE) | (w.d_status == HS.BREAK))
                out = removed_set(self.seed, x, cand)
                self.banned.update(out)
                for d in out:
                    w.on_shift_end(obs.t, d)                       # idle / break -> offline now
                obs = copy.copy(obs)
                obs.idle = obs.idle[~obs.idle.driver_id.isin(out)]
                self.record = {"t": obs.t, "removed": out, "candidates": int(len(cand)), "online": int((w.d_status != HS.OFF).sum() + len(out))}
            else:
                w.surge = np.maximum(w.surge, x)
                obs.fleet["surge"] = w.surge
                self.record = {"t": obs.t, "floor": x}
        return super().act(obs)


def run_real(state, date, pol, t1, day_dir):
    """EX.run_day_with, with the shock's hooks installed before the run (the event loop reads its
    handlers once): banned drivers never come back online; the surge floor holds after t_start."""
    import world.wm3.agent as _ag  # noqa: F401
    orig_init = HS.HDWorld.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        on_online, on_surge = self.on_online, self.on_surge

        def online(t, i, shift_end):
            if i in pol.banned:
                return
            on_online(t, i, shift_end)

        def surge(t):
            on_surge(t)
            if pol.shock[0] == "price" and t >= pol.t_start:
                self.surge = np.maximum(self.surge, pol.shock[1])
                self.surge_log[-1] = (t, self.surge.copy())
        self.on_online, self.on_surge = online, surge
    HS.HDWorld.__init__ = init
    EX.HDWorld = HS.HDWorld
    try:
        return EX.run_day_with(state, date, pol, t1 / 3600 + 0.25, day_dir)
    finally:
        HS.HDWorld.__init__ = orig_init


def real_row(L, t0, t1, cost):
    m = EX.real_metrics(L, t0, t1, cost)
    e0, e1 = int(t0 // S.DT), int(t1 // S.DT)
    m["declined"] = int(((L.req.t_request > t0) & (L.req.t_request <= t1) & (L.req.status == "declined")).sum())
    m["online_mean"] = float((L.status[:, e0:e1] != 0).sum(0).mean())
    traj = []
    ev = L.ev
    for e in range(e0, e1 + 1, EVERY):
        st = L.status[:, e]
        R = L.open_requests(e)
        w = ev[(ev.t_event > t0) & (ev.t_event <= e * S.DT)]
        dec = int(((L.req.t_request > t0) & (L.req.t_request <= e * S.DT) & (L.req.status == "declined")).sum())
        traj.append([int((st != 0).sum()), int((st == 1).sum()), int((R.status <= 1).sum()), int((w.type == "pickup").sum()),
                     int((w.type == "customer_cancelled").sum()), dec])
    m["waiting_mean"] = float(np.mean([r[2] for r in traj]))
    m["traj"] = traj
    return m


def imagine(key, Lb, prior, rules, e0, H, rec, model):
    """All arms x draws, batched, with common random numbers per draw."""
    arms = ["base"] + list(ARMS)
    Ws, tags = [], []
    for arm in arms:
        for s in range(DRAWS):
            W = AG.AgentWorld(model, rules, Lb, prior, e0, None, seed=11 + s)
            W.setup(None, 11 + s, "rule")
            if arm != "base":
                kind, x = ARMS[arm]
                if kind == "supply":
                    out = rec["arms"][arm]["shock"]["removed"]
                    ban = set(out)
                    for d in out:
                        if W.st[d] in (RO.IDLE, RO.BREAK):
                            W._offline(d, W.t)
                    on_online = W._on_online
                    W._on_online = lambda t, d, push, t1, f=on_online, ban=ban: None if d in ban else f(t, d, push, t1)
                else:
                    W.surge = np.maximum(W.surge, x)
                    sg = W._surge

                    def surge(W=W, sg=sg, x=x):
                        sg()
                        W.surge = np.maximum(W.surge, x)
                    W._surge = surge
            Ws.append(W)
            tags.append((arm, s))
    ptr = [len(W.log) for W in Ws]
    log0 = list(ptr)
    inc0 = [W.income.copy() for W in Ws]
    cnt = [Counter() for _ in Ws]
    trajs = [[] for _ in Ws]
    on_sum = np.zeros(len(Ws))
    on_hours = [np.zeros(W.nD) for W in Ws]

    def record():
        for i, W in enumerate(Ws):
            for x in W.log[ptr[i]:]:
                cnt[i][x[1]] += 1
            ptr[i] = len(W.log)
            k = W.kpis()
            trajs[i].append([k["online"], k["idle"], k["waiting_riders"], cnt[i]["pickup"], cnt[i]["customer_cancelled"], cnt[i]["quote_declined"]])

    record()
    chunk = {"sf_hd": 8, "pune_hd": 4}[key]
    for h in range(H):
        AG.rollout_many(Ws, 1, chunk)
        for i, W in enumerate(Ws):
            on = W.st != 0
            on_sum[i] += on.sum()
            on_hours[i] += on * S.DT / 3600
        if (h + 1) % EVERY == 0:
            record()
    out = {a: [] for a in arms}
    for i, (arm, s) in enumerate(tags):
        W = Ws[i]
        m = AG.outcome(W, log0[i], inc0[i])
        typ = [x[1] for x in W.log[log0[i]:]]
        net = (W.income - inc0[i]) - W.kappa * W.km
        act = on_hours[i] > 0.25
        m.update({"declined": typ.count("quote_declined"), "online_mean": float(on_sum[i] / H),
                  "waiting_mean": float(np.mean([r[2] for r in trajs[i]])), "traj": trajs[i], "draw": s,
                  "net_per_h": float(net[act].sum() / on_hours[i][act].sum() / W.money) if act.any() else np.nan})
        out[arm].append(m)
    return out


def run_seed(key, seed, args):
    city = cities.get(key)
    root = city.data_dir / FOLDER
    rf = root / f"shock_seed{seed}.json"
    base_rec = json.loads((root / f"rec_seed{seed}.json").read_text())
    date = pd.Timestamp(base_rec["date"])
    t0, t1 = float(base_rec["t0"]), float(base_rec["t0"]) + WINDOW_H * 3600
    rec = json.loads(rf.read_text()) if rf.exists() else {"key": key, "seed": seed, "date": str(date.date()), "t0": t0, "t1": t1, "arms": {}}
    gr = S.CityGraph(city)
    rules = RO.CityRules(gr)
    base_dir = root / f"seed{seed}_rule" / str(date.date())
    Lb = S.DayLog(gr, base_dir)
    cost = Lb.drivers.cost_per_km.to_numpy()
    if "base" not in rec["arms"]:
        rec["arms"]["base"] = real_row(Lb, t0, t1, cost)
    state = None
    for arm, shock in ARMS.items():
        if arm in rec["arms"]:
            continue
        ta = time.time()
        state = state or EX.warm_state(key, seed, root, 3)
        prior = FT.Prior()
        pol = ShockPolicy(gr, rules, prior, t_start=t0, seed=seed, shock=shock)
        day_dir = root / f"seed{seed}_{arm}" / str(date.date())
        run_real(copy.deepcopy(state), date, pol, t1, day_dir)
        L = S.DayLog(gr, day_dir)
        n_before = int((L.ev.t_event < t0).sum())
        same = n_before == int((Lb.ev.t_event < t0).sum())
        rec["arms"][arm] = {**real_row(L, t0, t1, cost), "shock": pol.record, "same_before": bool(same), "minutes": (time.time() - ta) / 60}
        r, b = rec["arms"][arm], rec["arms"]["base"]
        print(f"  {key} seed {seed} {arm}: identical before {same}; pickups {r['pickups']} (base {b['pickups']}), cancels "
              f"{r['cancels']} ({b['cancels']}), declined {r['declined']} ({b['declined']}), wait {r['wait_min']:.2f} ({b['wait_min']:.2f}) "
              f"[{r['minutes']:.1f} min]", flush=True)
        rf.write_text(json.dumps(rec, indent=1, default=float))
    if "imagined" not in rec and not args.no_imagine:
        ta = time.time()
        prior = FT.Prior()
        for d in sorted(p for p in (root / f"seed{seed}").iterdir() if p.is_dir()):
            prior.update_light(d, 5000)
        model = TR.load("wm3_exo")[0].eval()
        rec["imagined"] = imagine(key, Lb, prior, rules, int(t0 // S.DT), int(WINDOW_H * 3600 // S.DT), rec, model)
        im = {a: np.mean([x["pickups"] for x in v]) for a, v in rec["imagined"].items()}
        print(f"  {key} seed {seed} imagined pickups " + ", ".join(f"{a} {v:.0f}" for a, v in im.items()) +
              f" [{(time.time() - ta) / 60:.1f} min]", flush=True)
        rf.write_text(json.dumps(rec, indent=1, default=float))
        torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------------------- report
def effects(key):
    rows = []
    for f in sorted((cities.get(key).data_dir / FOLDER).glob("shock_seed*.json")):
        rec = json.loads(f.read_text())
        if "imagined" not in rec:
            continue
        b, ib = rec["arms"]["base"], rec["imagined"]["base"]
        for arm in ARMS:
            if arm not in rec["arms"]:
                continue
            r, im = rec["arms"][arm], rec["imagined"][arm]
            row = {"key": key, "seed": rec["seed"], "arm": arm, "same_before": r["same_before"]}
            if "removed" in r["shock"]:
                row["removed"] = len(r["shock"]["removed"])
                row["removed_share_online"] = len(r["shock"]["removed"]) / max(r["shock"]["online"], 1)
            for m in METRICS:
                row[f"real_{m}"] = r[m] - b[m]
                row[f"model_{m}"] = float(np.mean([x[m] - y[m] for x, y in zip(im, ib)]))
                row[f"base_{m}"] = b[m]
                row[f"model_base_{m}"] = float(np.mean([y[m] for y in ib]))
            row["real_traj"] = (np.array(r["traj"]) - np.array(b["traj"])).tolist()
            row["model_traj"] = (np.mean([np.array(x["traj"]) - np.array(y["traj"]) for x, y in zip(im, ib)], 0)).tolist()
            rows.append(row)
    return pd.DataFrame(rows)


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
    df = pd.concat([effects(k) for k in ("sf_hd", "pune_hd")], ignore_index=True)
    if df.empty:
        print("no finished seeds")
        return
    df.drop(columns=["real_traj", "model_traj"]).to_csv(OUT / "36_shock_pairs.csv", index=False)
    summ = {}
    for (key, arm), d in df.groupby(["key", "arm"]):
        s = {"n": len(d), "identical_before": int(d.same_before.sum())}
        if "removed" in d:
            s["removed_mean"] = float(d.removed.mean())
            s["removed_share_online"] = float(d.removed_share_online.mean())
        for m in METRICS:
            x, y = d[f"model_{m}"].to_numpy(float), d[f"real_{m}"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan  # noqa: E731
            s[m] = {"real": float(y.mean()), "real_se": se(y), "model": float(x.mean()), "model_se": se(x),
                    "r": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else np.nan,
                    "mae": float(np.abs(x - y).mean()), "mae_no_effect": float(np.abs(y).mean()),
                    "sign_agree": float(np.mean(np.sign(x) == np.sign(y))),
                    "base_real": float(d[f"base_{m}"].mean()), "base_model": float(d[f"model_base_{m}"].mean())}
        summ[f"{key}/{arm}"] = s
    (OUT / "36_shock.json").write_text(json.dumps(summ, indent=1, default=float))
    keys = [k for k in ("sf_hd", "pune_hd") if k in set(df.key)]
    # bars: mean effect real vs model, per arm and metric
    show = ["pickups", "cancels", "declined", "wait_min", "waiting_mean", "online_mean", "net_per_h"]
    fig, axes = plt.subplots(len(keys), len(show), figsize=(2.4 * len(show), 2.9 * len(keys)), squeeze=False)
    for a, key in enumerate(keys):
        for j, m in enumerate(show):
            ax = axes[a, j]
            _ax(ax)
            xs = np.arange(len(ARMS))
            rv = [summ[f"{key}/{arm}"][m]["real"] for arm in ARMS]
            re_ = [summ[f"{key}/{arm}"][m]["real_se"] for arm in ARMS]
            mv = [summ[f"{key}/{arm}"][m]["model"] for arm in ARMS]
            me = [summ[f"{key}/{arm}"][m]["model_se"] for arm in ARMS]
            ax.bar(xs - 0.2, rv, 0.38, yerr=re_, color=INK2, label="simulator", error_kw={"lw": 0.8})
            ax.bar(xs + 0.2, mv, 0.38, yerr=me, color=BLUE, label="world model", error_kw={"lw": 0.8})
            ax.axhline(0, color=INK, lw=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(["-30%\ndrivers", "-60%\ndrivers", "price\nx1.5"], fontsize=7)
            if a == 0:
                ax.set_title(LABEL[m], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{CITY_NAME[key]}\neffect over 2 h", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Shocks never seen in training: effect in the simulator vs predicted by the world model (mean +- s.e. over seeds)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "36_shock_effects.png", dpi=130)
    plt.close(fig)
    # scatter per seed
    show2 = ["pickups", "cancels", "declined", "wait_min"]
    fig, axes = plt.subplots(1, len(show2), figsize=(3.6 * len(show2), 3.6))
    marks = {"supply30": "o", "supply60": "s", "price15": "^"}
    for j, m in enumerate(show2):
        ax = axes[j]
        _ax(ax)
        for key, col in zip(keys, [BLUE, ORANGE]):
            for arm, mk in marks.items():
                d = df[(df.key == key) & (df.arm == arm)]
                ax.scatter(d[f"real_{m}"], d[f"model_{m}"], s=18, marker=mk, color=col, alpha=0.75,
                           label=f"{CITY_NAME[key]} {arm}" if j == 0 else None)
        lo = min(df[f"real_{m}"].min(), df[f"model_{m}"].min())
        hi = max(df[f"real_{m}"].max(), df[f"model_{m}"].max())
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
        x, y = df[f"real_{m}"].to_numpy(float), df[f"model_{m}"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.set_title(f"{LABEL[m]}  (r {np.corrcoef(x[ok], y[ok])[0, 1]:.2f})", fontsize=9, color=INK)
        ax.set_xlabel("simulator effect", fontsize=8, color=INK2)
        ax.set_ylabel("world-model effect", fontsize=8, color=INK2)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Every seed and shock: predicted vs true effect (dashed = perfect)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "36_shock_scatter.png", dpi=130)
    plt.close(fig)
    # trajectories: waiting riders and cumulative pickups, mean over seeds
    fig, axes = plt.subplots(len(keys), 3, figsize=(11, 2.9 * len(keys)), squeeze=False)
    for a, key in enumerate(keys):
        for j, (name, ci) in enumerate([("riders waiting", 2), ("pickups (cum.)", 3), ("quotes declined (cum.)", 5)]):
            ax = axes[a, j]
            _ax(ax)
            for arm, ls in zip(ARMS, ["-", "--", ":"]):
                d = df[(df.key == key) & (df.arm == arm)]
                R = np.mean([np.array(x)[:, ci] for x in d.real_traj], 0)
                M = np.mean([np.array(x)[:, ci] for x in d.model_traj], 0)
                tt = np.arange(len(R)) * EVERY * S.DT / 60
                ax.plot(tt, R, color=INK2, ls=ls, lw=1.6, label=f"simulator {arm}")
                ax.plot(tt, M, color=BLUE, ls=ls, lw=1.6, label=f"model {arm}")
            ax.axhline(0, color=INK, lw=0.5)
            if a == 0:
                ax.set_title(f"effect on {name}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(CITY_NAME[key], fontsize=8, color=INK2)
            if a == len(keys) - 1:
                ax.set_xlabel("minutes after the shock", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "36_shock_traj.png", dpi=130)
    plt.close(fig)
    for k, s in summ.items():
        print(k, f"n {s['n']} identical {s['identical_before']}", *(f"\n   {m:13s} real {s[m]['real']:+8.2f} +- {s[m]['real_se']:.2f}  model "
                                                                 f"{s[m]['model']:+8.2f} +- {s[m]['model_se']:.2f}  r {s[m]['r']:.2f}  "
                                                                 f"mae {s[m]['mae']:.2f} (no-effect {s[m]['mae_no_effect']:.2f})" for m in METRICS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["sf_hd", "pune_hd"])
    ap.add_argument("--seed0", type=int, default=700)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--no-imagine", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    for seed in range(args.seed0, args.seed0 + args.n_seeds):
        run_seed(args.city, seed, args)


if __name__ == "__main__":
    main()
