"""Stage 14 (step 5): can the world model choose a good platform action?

    .venv/bin/python scripts/14_wm_plan.py --city sf_hd --seeds 4          # simulate the candidates
    .venv/bin/python scripts/14_wm_plan.py --score st_ro st4_blocks_ro      # rank them with the models

At 08:00 the platform can send N idle drivers from the crowded source area (as in stage 13)
to one of several under-supplied zones, or do nothing. Each candidate is simulated from the
identical morning (same seed), and its true outcome over the next hour is measured:
    pickups   riders picked up city-wide 08:00-09:00
    lost      unmatched riders plus cancellations city-wide (the demand the platform loses)
Each model is given the identical history and each candidate's action sequence, rolls out 60
minutes on its own predictions, and predicts the same two outcomes. A good world model ranks
the candidates the way the simulator does, so the action it would choose (the one with the
highest predicted pickups) is close to the truly best one.

Simulations go to data/<city>/wm2_plan2/seed<k>_c<j>.npz (+ _dirs.npy) and cands.json (the
first version, ordered exactly at 08:00 on a snapshot boundary, is kept in wm2_plan/).
Scores go to outputs/wm2/<outdir>/14_plan.json and 14_plan.png.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from world import cities
from world.hd import raster as R
from world.hd.run import Season
from world.hd.sim import HDDispatch
from world.wm2 import data as WD


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CF = _load("cf", "13_cf_truth.py")
TR = _load("tr", "11_wm2_train.py")
DEV = TR.DEV
CH = list(R.CHANNELS)
H = 12
N_DST = 6
OUTCOMES = {"pickups": [CH.index("pickups")], "lost": [CH.index("unmatched"), CH.index("cancellations")]}
DATES = {"sf_hd": "2025-04-23", "pune_hd": "2025-08-04"}       # test-period weekdays, not the stage-13 ones
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
T_ORDER = 8 * 3600 + 150.0     # 08:02:30, mid-frame like the orders in the training data (08:00 sits on a snapshot)
FOLDER = "wm2_plan2"


class Planner(HDDispatch):
    """HDDispatch plus, at 08:00, either nothing (dst=None, but the candidates are recorded)
    or one extra order sending N idle drivers from the source area to zone `dst`."""

    def __init__(self, n, dst=None, take=None, cands=None, **kw):
        """`take`, `cands`: source zones and candidate targets fixed from the first seed, so
        that every seed simulates the same candidates and their outcomes can be averaged."""
        super().__init__(**kw)
        self.n, self.dst, self.take, self.cands, self.done, self.record = n, dst, take, cands, False, {}

    def act(self, obs):
        act = super().act(obs)
        if self.done or obs.t < T_ORDER:
            return act
        self.done = True
        used = {d for d, _ in act.offers} | {d for d, _ in act.dispatches}
        free = obs.idle[~obs.idle.driver_id.isin(used)]
        Z = len(obs.recent_demand)
        cnt = np.bincount(free.zone, minlength=Z)
        src = int(cnt.argmax())
        near = np.argsort(obs.zone_tt[src])
        take = near[: int(np.searchsorted(np.cumsum(cnt[near]), self.n)) + 1]
        need = obs.recent_demand / obs.recent_window_s * 600 + np.bincount(obs.pending.origin_zone, minlength=Z)
        deficit = need - (np.bincount(obs.idle.zone, minlength=Z) + obs.repositioning_to)
        reach = np.flatnonzero((obs.zone_tt[src] <= 1200) & ~np.isin(np.arange(Z), take))
        cands = reach[np.argsort(-deficit[reach])][:N_DST]
        if self.take is not None:
            take, cands = np.array(self.take), np.array(self.cands)
        self.record = {"src_zone": src, "src_zones": [int(z) for z in take], "cands": [int(z) for z in cands],
                       "deficit": [float(deficit[z]) for z in cands], "travel_s": [float(obs.zone_tt[src, z]) for z in cands]}
        if self.dst is not None:
            movers = free[free.zone.isin(take)].sort_values("idle_since").driver_id.to_numpy()[: self.n]
            act.dispatches = act.dispatches + [(int(d), int(self.dst)) for d in movers]
            self.record["n_moved"] = int(len(movers))
        return act


def simulate(key, n_seeds, date):
    city = cities.get(key)
    out = city.data_dir / FOLDER
    tmp = out / "runs"
    out.mkdir(exist_ok=True)
    cr = R.CityRaster(city)
    f = out / "cands.json"
    recs = json.loads(f.read_text()) if f.exists() else []
    done = {r["seed"] for r in recs}
    n = CF.N_MOVE[key]

    first = recs[0] if recs else None

    def run(seed, dst, j):
        fix = dict(take=first["src_zones"], cands=first["cands"]) if first else {}
        pol = Planner(n=n, dst=dst, seed=seed, **fix, **CF.BASE)
        season = Season(key, seed=seed, policy_factory=lambda: pol, log=lambda *a: None)
        season.run_day(date, out_dir=tmp / f"c{j}", horizon_h=9.5)
        X, C, Y, meta = cr.day(tmp / f"c{j}" / date, seed=seed)
        R.save_day(out / f"seed{seed}_c{j}.npz", X, C, Y, {**meta, "dst": dst})
        np.save(out / f"seed{seed}_c{j}_dirs.npy", cr.dispatch_dirs(tmp / f"c{j}" / date))
        return pol.record

    for seed in range(200, 200 + n_seeds):
        if seed in done:
            continue
        t0 = time.time()
        rec = {"seed": seed, "date": date, "n": n, **run(seed, None, 0)}   # candidate 0: no extra order
        rec["n_moved"] = []
        for j, z in enumerate(rec["cands"], 1):
            rec["n_moved"].append(run(seed, z, j)["n_moved"])
        shutil.rmtree(tmp)
        recs.append(rec)
        first = recs[0]
        f.write_text(json.dumps(recs, indent=1))
        print(f"  {key} seed {seed}: source zone {rec['src_zone']}, candidates {rec['cands']}  "
              f"({time.time() - t0:.0f} s)", flush=True)


def load(key):
    city = cities.get(key)
    d = city.data_dir / FOLDER
    recs = json.loads((d / "cands.json").read_text())
    for r in recs:
        r["arms"] = []
        for j in range(len(r["cands"]) + 1):
            X, C, _, _ = R.load_day(d / f"seed{r['seed']}_c{j}.npz")
            X[:, WD.SURGE] *= 100
            r["arms"].append((np.clip(np.round(X), 0, 255), C, np.load(d / f"seed{r['seed']}_c{j}_dirs.npy").astype(np.float32)))
        X0, X1 = r["arms"][0][0], r["arms"][1][0]
        diff = np.abs(X0[:, WD.ACT] - X1[:, WD.ACT]).reshape(len(X0), -1).sum(1)
        r["a"] = int(np.argmax(diff > 0))
    return recs, np.load(city.data_dir / "wm2" / "static.npz")["S"].astype(np.float32)


def outcome(frames):
    """frames (..., H, C, G, G) -> {name: (...)} summed over the hour and the city."""
    return {k: frames[..., :, ch, :, :].sum(axis=(-1, -2, -3, -4)) for k, ch in OUTCOMES.items()}


@torch.no_grad()
def rollouts(model, K, dirs, recs, S):
    """(seeds, candidates, H, C, G, G) expected frames from the model's own rollouts."""
    J = len(recs[0]["arms"])
    res = np.zeros((len(recs), J, H, len(CH), R.G, R.G), np.float32)
    jobs = [(i, j) for i in range(len(recs)) for j in range(J)]
    St = torch.tensor(S, device=DEV)
    for b0 in range(0, len(jobs), 8):
        batch = jobs[b0: b0 + 8]
        xs, cs = [], []
        for i, j in batch:
            X, C, Dd = recs[i]["arms"][j]
            lo, hi = recs[i]["a"] - K, recs[i]["a"] + H
            xs.append(np.concatenate([X[lo:hi], Dd[lo:hi]], 1) if dirs else X[lo:hi])
            cs.append(C[lo:hi])
        buf = torch.tensor(np.stack(xs), device=DEV)
        c = torch.tensor(np.stack(cs), device=DEV)
        Sb = St[None].expand(len(batch), *St.shape)
        for h in range(H):
            inp, cn, _ = WD.make_io(buf[:, h: h + K + 1], c[:, h: h + K + 1], Sb)
            with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
                o = model(inp, cn)[:, -1].float()
            pred = TR.to_frame(o)
            buf[:, h + K, WD.OBS] = pred[:, WD.OBS]
            for k, (i, j) in enumerate(batch):
                res[i, j, h] = pred[k].cpu().numpy()
    return res


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) and np.std(rb) else float("nan")


def score(tags, outdir):
    out_dir = cities.OUTPUTS / "wm2" / outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for key in TR.CITIES:
        recs, S = load(key)
        truth = np.stack([np.stack([X[r["a"]: r["a"] + H] for X, _, _ in r["arms"]]) for r in recs])
        tv = outcome(truth)                                            # name -> (seeds, J)
        # the effect of each candidate is measured against "no extra order" on the same seed;
        # the truth is averaged over seeds, since one seed's runs also differ by chance
        teff = {k: (v - v[:, :1]).mean(0) for k, v in tv.items()}
        tse = {k: (v - v[:, :1]).std(0, ddof=1) / np.sqrt(len(recs)) for k, v in tv.items()}
        r = {"seeds": [x["seed"] for x in recs], "date": recs[0]["date"], "n_moved": recs[0]["n_moved"],
             "cands": recs[0]["cands"], "same_cands_all_seeds": all(x["cands"] == recs[0]["cands"] for x in recs),
             "truth": {k: {"effect": teff[k].tolist(), "se": tse[k].tolist()} for k in OUTCOMES}}
        best = int(np.argmax(teff["pickups"]))
        for tag in tags:
            model, ck = TR.load_ckpt(tag)
            model.eval()
            pv = outcome(rollouts(model, ck["K"], ck.get("dirs", False), recs, S))
            peff = {k: (v - v[:, :1]).mean(0) for k, v in pv.items()}
            pick = int(np.argmax(peff["pickups"]))
            r[tag] = {"effect": {k: v.tolist() for k, v in peff.items()},
                      "spearman_pickups": spearman(peff["pickups"], teff["pickups"]),
                      "spearman_lost": spearman(peff["lost"], teff["lost"]),
                      "choice": pick, "regret_pickups": float(teff["pickups"][best] - teff["pickups"][pick])}
            print(f"  {key} {tag}: rank correlation pickups {r[tag]['spearman_pickups']:.2f}, lost {r[tag]['spearman_lost']:.2f}; "
                  f"chooses candidate {pick} (best {best}), regret {r[tag]['regret_pickups']:.1f} pickups/h", flush=True)
        r["best"] = best
        res[key] = r
        print(f"  {key} truth: pickups effect {np.round(teff['pickups'], 1).tolist()} (se {np.round(tse['pickups'], 1).tolist()})")
    (out_dir / "14_plan.json").write_text(json.dumps(res, indent=1))
    plot(res, tags, out_dir / "14_plan.png")
    print(f"  wrote {out_dir / '14_plan.png'}")


def plot(res, tags, path):
    pal = ["#2a78d6", "#eb6834", "#1baa7a", "#8a5cd6", "#c9a227"]
    fig, axs = plt.subplots(len(res), 2, figsize=(13, 4.2 * len(res)), constrained_layout=True, squeeze=False)
    for ci, (key, r) in enumerate(res.items()):
        J = len(r["truth"]["pickups"]["effect"])
        x = np.arange(J)
        labels = ["no order"] + [f"to zone {z}" for z in r["cands"]]
        w = 0.8 / (1 + len(tags))
        for k, name in enumerate(OUTCOMES):
            ax = axs[ci, k]
            t = r["truth"][name]
            ax.bar(x - 0.4 + w / 2, t["effect"], w, yerr=2 * np.array(t["se"]), color="#0b0b0b", alpha=.75, label="simulator (truth)")
            for j, tag in enumerate(tags):
                lab = f"{tag} (rank r = {r[tag]['spearman_' + name]:.2f})"
                ax.bar(x - 0.4 + w * (j + 1.5), r[tag]["effect"][name], w, color=pal[j % len(pal)], label=lab)
            ax.axhline(0, color="#898781", lw=.7)
            ax.set_xticks(x, labels, fontsize=8)
            ax.set_title(f"{CITY_NAME[key]}: change in {'pickups' if name == 'pickups' else 'unmatched + cancelled riders'} "
                         f"08:00-09:00 vs no order", fontsize=9.5)
            ax.legend(fontsize=7)
        axs[ci, 0].annotate("best (truth)", (r["best"] - 0.4 + w / 2, r["truth"]["pickups"]["effect"][r["best"]]),
                            fontsize=7, ha="center", va="bottom")
    fig.suptitle(f"Planning with the world model: send {r['n_moved'][0]} drivers at 08:00 to which zone? "
                 f"(mean over {len(r['seeds'])} seeds; error bars = ±2 standard errors)", fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["sf_hd", "pune_hd"])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--score", nargs="+", default=None, help="model tags to score")
    ap.add_argument("--outdir", default="step5")
    args = ap.parse_args()
    if args.score:
        score(args.score, args.outdir)
    else:
        simulate(args.city, args.seeds, DATES[args.city])


if __name__ == "__main__":
    main()
