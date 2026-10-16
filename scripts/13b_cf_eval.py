"""Stage 13b: does the world model predict what a platform action really does?

    .venv/bin/python scripts/13b_cf_eval.py --tags st_small st_ro st4_k6_dirs --outdir step4

Uses the simulator counterfactuals of 13_cf_truth.py: the same morning with and without one
extra order at 08:00 (N idle drivers sent from a crowded area to an under-supplied zone).
The true effect is the cf run minus the base run, averaged over seeds. Each model gets the
identical history up to the order and then each run's own sequence of platform actions, and
rolls out 60 minutes on its own predictions; its predicted effect is cf minus base of its
rollouts, averaged the same way.

Writes outputs/wm2/<outdir>/13_cf_eval.json and 13_cf_eval.png.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from world import cities
from world.hd import raster as R
from world.wm2 import data as WD

_spec = importlib.util.spec_from_file_location("tr", Path(__file__).with_name("11_wm2_train.py"))
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)
DEV = TR.DEV
CH = list(R.CHANNELS)
H = 12
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
IDLE = [CH.index(f"idle_{m}") for m in R.MODES3]
QUANT = {   # name -> (channels, region)
    "idle drivers, source area": (IDLE, "src"),
    "idle drivers, target zone": (IDLE, "dst"),
    "repositioning drivers, city": ([CH.index("repositioning")], "city"),
    "busy drivers (to pickup + with rider), target zone": ([CH.index("to_pickup"), CH.index("with_rider")], "dst"),
    "unmatched riders, target zone": ([CH.index("unmatched")], "dst"),
    "pickups, target zone": ([CH.index("pickups")], "dst"),
}


def load_pairs(city, folder="wm2_cf"):
    d = city.data_dir / folder
    pairs = [p for p in json.loads((d / "pairs.json").read_text()) if p["travel_s"] <= 1800]   # targets within 30 min
    st = np.load(city.data_dir / "wm2" / "static.npz")
    out = []
    for p in pairs:
        arms = {}
        for arm in ("base", "cf"):
            X, C, _, _ = R.load_day(d / f"seed{p['seed']}_{arm}.npz")
            X[:, WD.SURGE] *= 100                                   # same units as the store
            arms[arm] = (np.clip(np.round(X), 0, 255), C, np.load(d / f"seed{p['seed']}_{arm}_dirs.npy").astype(np.float32))
        act = WD.ACT
        diff = np.abs(arms["base"][0][:, act] - arms["cf"][0][:, act]).reshape(len(X), -1).sum(1)
        a = int(np.argmax(diff > 0))
        cz = st["cell_zone"]
        masks = {"src": np.isin(cz, p["src_zones"]), "dst": cz == p["dst_zone"], "city": np.ones_like(cz, bool)}
        out.append({**p, "arms": arms, "a": a, "masks": masks})
    return out, st["S"].astype(np.float32)


@torch.no_grad()
def model_rollouts(model, K, dirs, pairs, S, feed="mean"):
    """(n_pairs, 2 arms, H, C, G, G) expected frames from the model's own rollouts.
    feed: what goes back into the history, "mean" (expected counts, clamped to the uint8
    range of the data) or "sample" (Poisson-sampled whole counts, like the frames the model
    was trained on)."""
    res = np.zeros((len(pairs), 2, H, len(CH), R.G, R.G), np.float32)
    jobs = [(i, j) for i in range(len(pairs)) for j in range(2)]
    St = torch.tensor(S, device=DEV)
    for b0 in range(0, len(jobs), 8):
        batch = jobs[b0: b0 + 8]
        xs, cs = [], []
        for i, j in batch:
            p = pairs[i]
            X, C, Dd = p["arms"][("base", "cf")[j]]
            lo, hi = p["a"] - K, p["a"] + H
            x = X[lo:hi]
            if dirs:
                x = np.concatenate([x, Dd[lo:hi]], 1)
            xs.append(x); cs.append(C[lo:hi])
        buf = torch.tensor(np.stack(xs), device=DEV)
        c = torch.tensor(np.stack(cs), device=DEV)
        Sb = St[None].expand(len(batch), *St.shape)
        for h in range(H):
            inp, cn, _ = WD.make_io(buf[:, h: h + K + 1], c[:, h: h + K + 1], Sb)
            with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
                o = model(inp, cn)[:, -1].float()
            pred = TR.to_frame(o)
            fb = pred.clone()
            if feed == "sample":
                fb[:, WD.COUNT] = torch.poisson(pred[:, WD.COUNT])
            buf[:, h + K, WD.OBS] = fb[:, WD.OBS].clamp(0, 255)
            for k, (i, j) in enumerate(batch):
                res[i, j, h] = pred[k].cpu().numpy()
    return res


def effects(frames_base, frames_cf, pairs):
    """Per pair and horizon: cf - base of each quantity (frames: (P, H, C, G, G))."""
    out = {q: np.zeros((len(pairs), H)) for q in QUANT}
    for i, p in enumerate(pairs):
        for q, (chs, reg) in QUANT.items():
            m = p["masks"][reg]
            out[q][i] = (frames_cf[i][:, chs].sum(1)[:, m] - frames_base[i][:, chs].sum(1)[:, m]).sum(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--outdir", default="step4")
    ap.add_argument("--cf", default="wm2_cf", help="counterfactual folder under data/<city>/ (step 4d: wm2_cf2)")
    ap.add_argument("--feed", default="mean", choices=["mean", "sample"], help="step 4d: what a rollout feeds back")
    ap.add_argument("--samples", type=int, default=4, help="rollouts averaged per run with --feed sample")
    args = ap.parse_args()
    out_dir = cities.OUTPUTS / "wm2" / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    results, maps = {}, {}
    for key in TR.CITIES:
        city = cities.get(key)
        pairs, S = load_pairs(city, args.cf)
        P = len(pairs)
        truth = effects(np.stack([p["arms"]["base"][0][p["a"]: p["a"] + H] for p in pairs]),
                        np.stack([p["arms"]["cf"][0][p["a"]: p["a"] + H] for p in pairs]), pairs)
        r = {"n_pairs": P, "n_moved": [p["n_moved"] for p in pairs],
             "truth": {q: {"mean": v.mean(0).tolist(), "se": (v.std(0, ddof=1) / np.sqrt(P)).tolist()} for q, v in truth.items()}}
        # effect maps (idle drivers, 10-30 min after the order) over the seeds that share the most
        # common source and target, so the outlines drawn are right for every seed averaged
        sites = [(p["src_zone"], p["dst_zone"]) for p in pairs]
        common = max(set(sites), key=sites.count)
        grp = [i for i, st_ in enumerate(sites) if st_ == common]
        tmap = np.mean([(pairs[i]["arms"]["cf"][0][pairs[i]["a"] + 2: pairs[i]["a"] + 6, IDLE].sum((0, 1)) -
                         pairs[i]["arms"]["base"][0][pairs[i]["a"] + 2: pairs[i]["a"] + 6, IDLE].sum((0, 1))) / 4 for i in grp], 0)
        maps[key] = {"truth": tmap, "pairs": [pairs[i] for i in grp]}
        r["map_seeds"] = [pairs[i]["seed"] for i in grp]
        for tag in args.tags:
            model, ck = TR.load_ckpt(tag)
            model.eval()
            if args.feed == "sample":
                torch.manual_seed(0)
                fr = np.mean([model_rollouts(model, ck["K"], ck.get("dirs", False), pairs, S, "sample")
                              for _ in range(args.samples)], 0)
            else:
                fr = model_rollouts(model, ck["K"], ck.get("dirs", False), pairs, S)
            eff = effects(fr[:, 0], fr[:, 1], pairs)
            err, tot = 0.0, 0.0
            for q in QUANT:
                err += np.abs(eff[q].mean(0) - truth[q].mean(0)).sum()
                tot += np.abs(truth[q].mean(0)).sum()
            mmap = (fr[grp, 1, 2:6][:, :, IDLE].sum((1, 2)) - fr[grp, 0, 2:6][:, :, IDLE].sum((1, 2))).mean(0) / 4
            blk = lambda m: m.reshape(16, 4, 16, 4).sum((1, 3)).ravel()  # noqa: E731
            corr = float(np.corrcoef(blk(mmap), blk(tmap))[0, 1])
            r[tag] = {"effects": {q: v.mean(0).tolist() for q, v in eff.items()},
                      "relative_error": float(err / max(tot, 1e-9)), "map_corr_4x4": corr}
            maps[key][tag] = mmap
            print(f"  {key} {tag}: relative error of the effect {err / tot:.3f}, map correlation {corr:.3f}  " +
                  "  ".join(f"{q.split(',')[0][:12]}@{q.split(', ')[1][:6]} {eff[q].mean(0)[[1, 5, 11]].round(1).tolist()}"
                            f" (true {truth[q].mean(0)[[1, 5, 11]].round(1).tolist()})" for q in QUANT), flush=True)
        results[key] = r
    (out_dir / "13_cf_eval.json").write_text(json.dumps(results, indent=1))
    plot(results, maps, args.tags, out_dir / "13_cf_eval.png")
    print(f"  wrote {out_dir / '13_cf_eval.png'}")


def plot(results, maps, tags, path):
    nq = len(QUANT)
    fig = plt.figure(figsize=(3.3 * nq, 3.2 * 2 * 2), constrained_layout=True)
    sub = fig.subfigures(2 * len(results), 1)
    pal = ["#2a78d6", "#eb6834", "#1baa7a", "#8a5cd6", "#c9a227", "#d64f7a"]
    t = np.arange(1, H + 1) * 5
    for ci, (key, r) in enumerate(results.items()):
        axs = sub[2 * ci].subplots(1, nq)
        for ax, q in zip(axs, QUANT):
            m, se = np.array(r["truth"][q]["mean"]), np.array(r["truth"][q]["se"])
            ax.fill_between(t, m - 2 * se, m + 2 * se, color="#0b0b0b", alpha=.12, lw=0)
            ax.plot(t, m, "k-", lw=2, label="simulator (truth)")
            for j, tag in enumerate(tags):
                ax.plot(t, r[tag]["effects"][q], "o-", ms=2.5, lw=1.3, color=pal[j % len(pal)], label=tag)
            ax.axhline(0, color="#898781", lw=.7)
            lim = max(np.abs(m).max() + 2 * se.max(), 1.0) * 2.5             # models that blow up go off the chart
            ax.set_ylim(-lim, lim)
            ax.set_title(q, fontsize=8.5); ax.set_xlabel("min after the order", fontsize=8); ax.tick_params(labelsize=7)
        axs[0].legend(fontsize=7)
        sub[2 * ci].suptitle(f"{CITY_NAME[key]}: effect of sending {int(np.mean(r['n_moved']))} drivers at 08:00 "
                             f"(cf minus base, mean of {r['n_pairs']} seeds; band = ±2 standard errors; lines leaving the chart = rollout blew up)", fontsize=10)
        axs = sub[2 * ci + 1].subplots(1, 1 + len(tags))
        mm = maps[key]
        blk = lambda m: np.kron(m.reshape(16, 4, 16, 4).sum((1, 3)), np.ones((4, 4)))  # noqa: E731  4 x 4 block sums
        v = max(abs(blk(mm["truth"])).max(), 1e-3) * 0.8
        p0 = mm["pairs"][0]
        for ax, (name, arr) in zip(axs, [("simulator (truth)", mm["truth"])] + [(tg, mm[tg]) for tg in tags]):
            ax.imshow(blk(arr), cmap="RdBu_r", vmin=-v, vmax=v)
            ax.contour(p0["masks"]["dst"].astype(float), levels=[.5], colors="#eb6834", linewidths=1)
            ax.contour(p0["masks"]["src"].astype(float), levels=[.5], colors="#2a78d6", linewidths=1)
            title = name if name.startswith("sim") else f"{name}  (r = {results[key][name]['map_corr_4x4']:.2f})"
            ax.set_title(title, fontsize=9); ax.axis("off")
        sub[2 * ci + 1].suptitle(f"change in idle drivers 10-30 min after the order for the {len(mm['pairs'])} seeds with the "
                                 "same source (blue outline) and target zone (orange), summed over 4 x 4 blocks; r = correlation with the truth over 4 x 4 blocks", fontsize=9)
    fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
