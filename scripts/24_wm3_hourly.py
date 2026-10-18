"""World model 3: do the rare per-driver and per-rider events add up hour by hour?

    .venv/bin/python scripts/24_wm3_hourly.py --tags wm3_v2 wm3_tf

A rollout sums each head's probability over every driver or rider and every 30-s step, so a
head can be well calibrated per instance and still put the city's total in the wrong hour
(e.g. 1e-4 instead of 1e-5 for each of 1,200 offline drivers is 14 extra log-ons an hour).
For every snapshot of a few test days, sums the predicted probability of each rare event per
hour of the day and compares it with the number that happened:
  offline drivers coming online, online drivers going offline, taking a break, heading home,
  riders cancelling.
Writes outputs/wm3/stage3/24_hourly.png and 24_hourly.json.
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
from world.wm3 import data as WD
from world.wm3 import features as FT

_spec = importlib.util.spec_from_file_location("tr3", Path(__file__).with_name("18_wm3_train.py"))
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)
DEV = TR.DEV
OUT = cities.OUTPUTS / "wm3" / "stage3"
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
EVS = {"coming online": None, "going offline": FT.EV.index("offline"), "taking a break": FT.EV.index("break"),
       "heading home": FT.EV.index("heads_home"), "rider cancels": None}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"]


def hour_of(G):
    return int(np.floor((np.arctan2(G[0], G[1]) / (2 * np.pi) * 24) % 24 + 1e-6))


def run_tag(tag, days, stride):
    model, _ = TR.load(tag)
    model.eval()
    res = {}
    for key, files in days.items():
        pred = {k: np.zeros(24) for k in EVS}
        true = {k: np.zeros(24) for k in EVS}
        for f in files:
            z = np.load(f)
            day = {"key": key, "date": f.stem, **{n: z[n] for n in z.files}}
            for i in range(0, len(day["epoch"]), stride):
                s = WD.snapshot(day, i)
                h = int(day["epoch"][i] * 30 // 3600) % 24
                w = stride * int(day["epoch"][1] - day["epoch"][0])
                b = WD.collate([s], DEV)
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
                    out = model(b)
                po = torch.sigmoid(out["online"][0].float()).cpu().numpy()[:len(s["F"])]
                pred["coming online"][h] += po.sum() * w
                true["coming online"][h] += s["F_on"].sum() * w
                pe = torch.softmax(out["ev"][0].float(), -1).cpu().numpy()[:len(s["D"])]
                for k, j in EVS.items():
                    if j is not None:
                        pred[k][h] += pe[:, j].sum() * w
                        true[k][h] += (s["D_ev"] == j).sum() * w
                pc = torch.sigmoid(out["cancel"][0].float()).cpu().numpy()[:len(s["R"])]
                pred["rider cancels"][h] += pc.sum() * w
                true["rider cancels"][h] += s["R_cancel"].sum() * w
        n = len(files)
        res[key] = {"pred": {k: (v / n).tolist() for k, v in pred.items()}, "true": {k: (v / n).tolist() for k, v in true.items()}}
        print(f"  {tag} {key}: " + "  ".join(
            f"{k} {sum(res[key]['pred'][k]):.0f}/{sum(res[key]['true'][k]):.0f} (abs. hourly err "
            f"{np.abs(np.array(res[key]['pred'][k]) - res[key]['true'][k]).sum() / max(sum(res[key]['true'][k]), 1):.2f})"
            for k in EVS), flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["wm3_v2"])
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    days = {}
    for key in WD.CITY_KEYS:
        pool = WD.DayPool("test", n_days=1, swap=10 ** 9, seed=0, keys=[key])
        days[key] = [f for _, f in pool.files[:: max(1, len(pool.files) // args.days)][: args.days]]
    res = {tag: run_tag(tag, days, args.stride) for tag in args.tags}
    summary = {}
    for tag, r in res.items():
        for key, v in r.items():
            for k in EVS:
                p, t = np.array(v["pred"][k]), np.array(v["true"][k])
                summary[f"{tag}/{key}/{k}"] = {"pred_day": float(p.sum()), "true_day": float(t.sum()),
                                               "hourly_abs_err_share": float(np.abs(p - t).sum() / max(t.sum(), 1))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "24_hourly.json").write_text(json.dumps({"days": {k: [f.stem for f in v] for k, v in days.items()},
                                                     "summary": summary, "curves": res}, indent=1))
    fig, axs = plt.subplots(2, len(EVS), figsize=(3.6 * len(EVS), 6.4))
    x = np.arange(24) + 0.5
    for i, key in enumerate(WD.CITY_KEYS):
        for j, k in enumerate(EVS):
            ax = axs[i, j]
            first = res[args.tags[0]][key]
            ax.step(x, first["true"][k], where="mid", color=INK, lw=1.8, label="happened")
            for c, tag in zip(COLORS, args.tags):
                ax.plot(x, res[tag][key]["pred"][k], color=c, lw=1.6, label=f"{tag} (sum of probabilities)")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(color=GRID, lw=0.6)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_title(f"{CITY_NAME[key]}: {k}", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel("events per hour (mean of test days)", fontsize=8, color=INK2)
            if i == 1:
                ax.set_xlabel("hour of day", fontsize=8, color=INK2)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Rare events per hour on test days: what happened vs the model's summed probabilities "
                 f"({args.days} days per city, every {args.stride * 3} min)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "24_hourly.png", dpi=115)
    plt.close(fig)
    print(f"  wrote {OUT / '24_hourly.png'}")


if __name__ == "__main__":
    main()
