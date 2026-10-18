"""World model 3, stage 2c: one-step evaluation of the event model on test days.

    .venv/bin/python scripts/19_wm3_eval.py --tag wm3_base

For every head, the test negative log-likelihood per instance against a frequency baseline
built from the training days (what a table of historical rates would predict):
    driver decision      class frequencies per status and hour
    offer / order answer frequencies per vehicle mode
    own reposition zone  zone frequencies
    comes online         rate per hour and day type
    rider cancels        rate per request status and minutes waited (5-min bins)
    rider boards         rate per minutes waited at the pickup
    new requests         mean per zone, mode and hour (Poisson)
    origin / destination node and zone frequencies (smoothed)
    quote declined       rate per mode and surge
Plus calibration curves, and whether the model has inferred the drivers' hidden traits
(predicted answer probabilities vs the hidden accept bias / compliance of the simulator).
Writes outputs/wm3/stage2/19_eval.json, 19_heads.png, 19_calibration.png, 19_hidden.png.
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
import pandas as pd
import torch
import torch.nn.functional as Fn

from world import cities
from world.wm3 import data as WD
from world.wm3 import features as FT
from world.wm3 import model as WM

_spec = importlib.util.spec_from_file_location("tr3", Path(__file__).with_name("18_wm3_train.py"))
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)
DEV = TR.DEV
OUT = TR.OUT
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
NEW = WM.NEW
DC = {c: i for i, c in enumerate(FT.D_COLS)}
FC = {c: i for i, c in enumerate(FT.F_COLS)}
INK, INK2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def hour_of(G):
    return int(np.round(np.arctan2(G[0], G[1]) / (2 * np.pi) * 24)) % 24


# ---------------------------------------------------------------- baseline tables
class Baseline:
    """Historical-rate tables from training snapshots of one city."""

    def __init__(self, key, n_days=40, seed=0):
        pool = WD.DayPool("train", n_days=1, swap=10 ** 9, seed=seed, keys=[key])
        pick = np.random.default_rng(seed).choice(len(pool.files), min(n_days, len(pool.files)), replace=False)
        t = WD.CityTables(key, "cpu")
        self.N, self.Z, self.n_ext = t.N, t.Z, t.n_ext
        self.ev = np.ones((8, 24, len(FT.EV)))
        self.offer = np.ones((3, 3)); self.order = np.ones((3, 2)); self.repo = np.ones(t.Z)
        self.online = np.ones((2, 24, 2)); self.cancel = np.ones((4, 13, 2)); self.board = np.ones((13, 2))
        self.rate_n = np.zeros((t.Z, 3, 24)); self.rate_d = np.zeros(24)
        self.onode = np.full(t.N, 0.5); self.dnode = np.full(t.N, 0.5)
        self.od = np.ones((t.Z, t.Z + t.n_ext)) * 0.2
        self.decl = np.ones((3, 6, 2))
        for fi in pick:                                   # one day in memory at a time
            day = pool._load(fi)
            for i in range(0, len(day["epoch"]), 2):
                s = WD.snapshot(day, i)
                h = hour_of(s["G"])
                wk = int(s["G"][4 + 5] + s["G"][4 + 6] > 0)
                D = s["D"]
                st = D[:, :8].argmax(1)
                np.add.at(self.ev, (st, h, s["D_ev"].astype(int)), 1)
                mode = D[:, [DC["mode_bike"], DC["mode_auto"], DC["mode_cab"]]].argmax(1)
                o = s["D_offer"] >= 0
                np.add.at(self.offer, (mode[o], s["D_offer"][o].astype(int)), 1)
                o = s["D_order"] >= 0
                np.add.at(self.order, (mode[o], s["D_order"][o].astype(int)), 1)
                o = s["D_repo"] >= 0
                np.add.at(self.repo, s["D_repo"][o].astype(int), 1)
                np.add.at(self.online, (wk, h, s["F_on"].astype(int)), 1)
                R = s["R"]
                rs = R[:, :4].argmax(1)
                wb = np.clip(R[:, FT.R_COLS.index("waited_min")] // 5, 0, 12).astype(int)
                np.add.at(self.cancel, (rs, wb, s["R_cancel"].astype(int)), 1)
                w = rs == 3
                pb = np.clip(R[w, FT.R_COLS.index("at_pickup_min")] // 0.5, 0, 12).astype(int)
                np.add.at(self.board, (pb, s["R_board"][w].astype(int)), 1)
                self.rate_n[:, :, h] += s["Z_new"]
                self.rate_d[h] += 1
                nw = s["NEW"]
                if len(nw):
                    np.add.at(self.onode, nw[:, NEW["origin"]].astype(int), 1)
                    ins = nw[:, NEW["external"]] < 0
                    np.add.at(self.dnode, nw[ins, NEW["dest_node"]].astype(int), 1)
                    dt = np.where(nw[:, NEW["external"]] >= 0, t.Z + nw[:, NEW["external"]], nw[:, NEW["dest_zone"]]).astype(int)
                    np.add.at(self.od, (nw[:, NEW["zone"]].astype(int), dt), 1)
                    sb = np.clip(np.round((nw[:, NEW["surge"]] - 1) * 10), 0, 5).astype(int)
                    np.add.at(self.decl, (nw[:, NEW["mode"]].astype(int), sb, nw[:, NEW["declined"]].astype(int)), 1)
        self.node_zone = t.node_zone.numpy()
        norm = lambda a: a / a.sum(-1, keepdims=True)  # noqa: E731
        self.ev, self.offer, self.order, self.repo = norm(self.ev), norm(self.offer), norm(self.order), norm(self.repo)
        self.online, self.cancel, self.board, self.od, self.decl = norm(self.online), norm(self.cancel), norm(self.board), norm(self.od), norm(self.decl)
        self.rate = (self.rate_n + 0.01) / np.maximum(self.rate_d, 1)          # mean new requests per 30 s
        zs_o = np.bincount(self.node_zone, weights=self.onode, minlength=t.Z)
        zs_d = np.bincount(self.node_zone, weights=self.dnode, minlength=t.Z)
        self.onode_p = self.onode / zs_o[self.node_zone]
        self.dnode_p = self.dnode / zs_d[self.node_zone]

    def nll(self, s):
        """Per-head (sum, count) for one snapshot."""
        h = hour_of(s["G"])
        wk = int(s["G"][4 + 5] + s["G"][4 + 6] > 0)
        D = s["D"]
        st = D[:, :8].argmax(1)
        mode = D[:, [DC["mode_bike"], DC["mode_auto"], DC["mode_cab"]]].argmax(1)
        r = {}
        r["driver_event"] = (-np.log(self.ev[st, h, s["D_ev"].astype(int)]).sum(), len(st))
        o = s["D_offer"] >= 0
        r["offer_answer"] = (-np.log(self.offer[mode[o], s["D_offer"][o].astype(int)]).sum(), o.sum())
        o = s["D_order"] >= 0
        r["order_answer"] = (-np.log(self.order[mode[o], s["D_order"][o].astype(int)]).sum(), o.sum())
        o = s["D_repo"] >= 0
        r["own_reposition_zone"] = (-np.log(self.repo[s["D_repo"][o].astype(int)]).sum(), o.sum())
        r["comes_online"] = (-np.log(self.online[wk, h, s["F_on"].astype(int)]).sum(), len(s["F_on"]))
        R = s["R"]
        rs = R[:, :4].argmax(1)
        wb = np.clip(R[:, FT.R_COLS.index("waited_min")] // 5, 0, 12).astype(int)
        r["rider_cancels"] = (-np.log(self.cancel[rs, wb, s["R_cancel"].astype(int)]).sum(), len(rs))
        w = rs == 3
        pb = np.clip(R[w, FT.R_COLS.index("at_pickup_min")] // 0.5, 0, 12).astype(int)
        r["rider_boards"] = (-np.log(self.board[pb, s["R_board"][w].astype(int)]).sum(), w.sum())
        lam = self.rate[:, :, h]
        y = s["Z_new"].astype(float)
        from scipy.special import gammaln
        r["new_requests"] = ((lam - y * np.log(lam) + gammaln(y + 1)).sum(), self.Z)
        nw = s["NEW"]
        if len(nw):
            ins = nw[:, NEW["external"]] < 0
            dt = np.where(~ins, self.Z + nw[:, NEW["external"]], nw[:, NEW["dest_zone"]]).astype(int)
            r["request_origin_node"] = (-np.log(self.onode_p[nw[:, NEW["origin"]].astype(int)]).sum(), len(nw))
            r["request_destination"] = (-np.log(self.od[nw[:, NEW["zone"]].astype(int), dt]).sum(), len(nw))
            r["request_dest_node"] = (-np.log(self.dnode_p[nw[ins, NEW["dest_node"]].astype(int)]).sum(), len(nw))
            sb = np.clip(np.round((nw[:, NEW["surge"]] - 1) * 10), 0, 5).astype(int)
            r["quote_declined"] = (-np.log(self.decl[nw[:, NEW["mode"]].astype(int), sb, nw[:, NEW["declined"]].astype(int)]).sum(), len(nw))
        return r


# ------------------------------------------------------------------ model pass
@torch.no_grad()
def model_pass(model, snaps):
    """Per-head NLL sums and the per-instance probabilities used by the plots."""
    sums, rec = {}, {k: [] for k in ("offer", "order", "online", "cancel", "ev")}
    for s in snaps:
        b = WD.collate([s], DEV)
        with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
            out = model(b)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        ls = WM.losses(model, b, out)
        n = {"driver_event": len(s["D_ev"]), "offer_answer": (s["D_offer"] >= 0).sum(), "order_answer": (s["D_order"] >= 0).sum(),
             "own_reposition_zone": (s["D_repo"] >= 0).sum(), "comes_online": len(s["F_on"]), "rider_cancels": len(s["R_cancel"]),
             "rider_boards": (s["R"][:, 3] > 0.5).sum(), "new_requests": len(s["Z"]),
             **{k: len(s["NEW"]) for k in ("request_origin_node", "request_destination", "request_dest_node", "quote_declined")}}
        for k, v in ls.items():                       # every head is a mean per instance
            a = sums.setdefault(k, [0.0, 0])
            a[0] += float(v) * n[k]
            a[1] += n[k]
        po = out["offer"][0].softmax(-1).cpu().numpy()
        o = s["D_offer"] >= 0
        rec["offer"] += [(s["key"], int(s["D_id"][j]), s["date"], po[j], int(s["D_offer"][j])) for j in np.flatnonzero(o)]
        pr = out["order"][0].softmax(-1).cpu().numpy()
        o = s["D_order"] >= 0
        rec["order"] += [(s["key"], int(s["D_id"][j]), s["date"], pr[j], int(s["D_order"][j])) for j in np.flatnonzero(o)]
        rec["online"] += list(zip(torch.sigmoid(out["online"][0]).cpu().numpy(), s["F_on"]))
        rec["cancel"] += list(zip(torch.sigmoid(out["cancel"][0]).cpu().numpy(), s["R_cancel"]))
        pe = out["ev"][0].softmax(-1).cpu().numpy()
        rec["ev"] += list(zip(pe, s["D_ev"]))
    return {k: v[0] / max(v[1], 1) for k, v in sums.items()}, rec


@torch.no_grad()
def answers_pass(model, key, n_days=6):
    """Predicted offer / order answers for every offer and order in all snapshots of n test days."""
    pool = WD.DayPool("test", n_days=1, swap=10 ** 9, seed=7, keys=[key])
    pick = np.random.default_rng(7).choice(len(pool.files), min(n_days, len(pool.files)), replace=False)
    offer, order = [], []
    for fi in pick:
        day = pool._load(fi)
        for i in range(len(day["epoch"])):
            s = WD.snapshot(day, i)
            o1, o2 = s["D_offer"] >= 0, s["D_order"] >= 0
            if not (o1.any() or o2.any()):
                continue
            b = WD.collate([s], DEV)
            with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
                out = model(b)
            po = out["offer"][0].float().softmax(-1).cpu().numpy()
            pr = out["order"][0].float().softmax(-1).cpu().numpy()
            offer += [(key, int(s["D_id"][j]), s["date"], po[j], int(s["D_offer"][j])) for j in np.flatnonzero(o1)]
            order += [(key, int(s["D_id"][j]), s["date"], pr[j], int(s["D_order"][j])) for j in np.flatnonzero(o2)]
    return offer, order


def calib(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, p, "right") - 1, 0, len(q) - 2)
    return np.array([[p[idx == i].mean(), y[idx == i].mean(), (idx == i).sum()] for i in range(len(q) - 1) if (idx == i).any()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wm3_base")
    ap.add_argument("--n", type=int, default=400, help="test snapshots per city")
    args = ap.parse_args()
    model, ck = TR.load(args.tag)
    model.eval()
    res, recs = {}, {}
    for key in WD.CITY_KEYS:
        base = Baseline(key)
        pool = WD.DayPool("test", n_days=10, swap=10 ** 9, seed=5, keys=[key])
        rng = np.random.default_rng(5)
        snaps = []
        for _ in range(args.n):
            day = pool.pool[rng.integers(len(pool.pool))]
            snaps.append(WD.snapshot(day, int(rng.integers(len(day["epoch"])))))
        bsum = {}
        for s in snaps:
            for k, (a, n) in base.nll(s).items():
                x = bsum.setdefault(k, [0.0, 0])
                x[0] += a; x[1] += n
        bnll = {k: v[0] / max(v[1], 1) for k, v in bsum.items()}
        mnll, rec = model_pass(model, snaps)
        res[key] = {k: {"model": mnll.get(k), "baseline": bnll.get(k), "n": int(bsum.get(k, [0, 0])[1])} for k in bnll}
        recs[key] = rec
        print(f"  {key}: test NLL per instance, model vs historical-rate baseline", flush=True)
        for k, v in res[key].items():
            print(f"    {k:22s} {v['model']:.4f}  {v['baseline']:.4f}  (n={v['n']})", flush=True)
    for key in WD.CITY_KEYS:                           # every snapshot of 6 test days: enough offers per driver
        recs[key]["offer"], recs[key]["order"] = answers_pass(model, key)
    hid = hidden(recs)
    for k in hid:
        res[k]["hidden"] = hid[k]
    (OUT / "19_eval.json").write_text(json.dumps(res, indent=1, default=float))
    plot_heads(res, OUT / "19_heads.png")
    plot_calib(recs, OUT / "19_calibration.png")
    plot_hidden(recs, hid, OUT / "19_hidden.png")
    print(json.dumps(hid, indent=1))


def hidden(recs):
    """Per driver: mean predicted acceptance / compliance vs the simulator's hidden traits."""
    out = {}
    for key, rec in recs.items():
        city = cities.get(key)
        lat = {}
        rows = []
        for kind, col, trait in (("offer", 0, "accept_bias"), ("order", 0, "reposition_compliance")):
            df = pd.DataFrame([(d, date, p[col], y) for _, d, date, p, y in rec[kind]], columns=["driver", "date", "p", "y"])
            if not len(df):
                continue
            for date in df.date.unique():
                if date not in lat:
                    f = next((city.data_dir / "hd_runs").glob(f"wm2_seed*/{date}/drivers_latent.parquet"))
                    lat[date] = pd.read_parquet(f).set_index("driver_id")
            df["trait"] = [lat[dt].loc[d, trait] for d, dt in zip(df.driver, df.date)]
            g = df.groupby("driver").agg(p=("p", "mean"), y=("y", lambda v: (v == 0).mean()), trait=("trait", "first"), n=("p", "size"))
            g = g[g.n >= 3]
            rows.append((kind, g))
            out.setdefault(key, {})[kind] = {"drivers": int(len(g)), "r_model_vs_trait": float(np.corrcoef(g.p, g.trait)[0, 1]),
                                             "r_observed_vs_trait": float(np.corrcoef(g.y, g.trait)[0, 1])}
        out.setdefault(key, {})["_frames"] = rows
    clean = {k: {kk: vv for kk, vv in v.items() if kk != "_frames"} for k, v in out.items()}
    hidden.frames = {k: v["_frames"] for k, v in out.items()}
    return clean


def plot_heads(res, path):
    heads = list(res["sf_hd"].keys())
    heads = [h for h in heads if h != "hidden"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    for ax, key in zip(axs, WD.CITY_KEYS):
        r = res[key]
        hs = [h for h in heads if h in r and r[h]["model"] is not None]
        gain = [100 * (1 - r[h]["model"] / r[h]["baseline"]) for h in hs]
        y = np.arange(len(hs))
        ax.barh(y, gain, color=[BLUE if g > 0 else ORANGE for g in gain], height=.6)
        for yi, (g, h) in enumerate(zip(gain, hs)):
            ax.text(g + (1 if g >= 0 else -1), yi, f"{g:+.0f}%   ({r[h]['model']:.3f} vs {r[h]['baseline']:.3f})",
                    va="center", ha="left" if g >= 0 else "right", fontsize=7.5, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([h.replace("_", " ") for h in hs], fontsize=8.5)
        ax.axvline(0, color=INK2, lw=.8); ax.invert_yaxis()
        ax.set_xlim(min(-20, min(gain) - 30), max(gain) + 45)
        ax.set_title(f"{CITY_NAME[key]}: test log-likelihood gain over historical rates (NLL model vs baseline)", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("% lower negative log-likelihood per instance", fontsize=8.5)
    fig.suptitle("World model 3, one step (30 s): every event head against a table of historical rates, on test days", fontsize=11)
    fig.savefig(path, dpi=120); plt.close(fig)


def plot_calib(recs, path):
    items = [("offer", "offer accepted", lambda r: ([p[0] for *_, p, y in r], [int(y == 0) for *_, p, y in r])),
             ("order", "repositioning order followed", lambda r: ([p[0] for *_, p, y in r], [int(y == 0) for *_, p, y in r])),
             ("online", "offline driver comes online (30 s)", lambda r: ([p for p, y in r], [int(y) for p, y in r])),
             ("cancel", "rider cancels (30 s)", lambda r: ([p for p, y in r], [int(y) for p, y in r])),
             ("ev", "online driver goes offline (30 s)", lambda r: ([p[1] for p, y in r], [int(y == 1) for p, y in r])),
             ("ev", "idle driver takes a break (30 s)", lambda r: ([p[2] for p, y in r], [int(y == 2) for p, y in r]))]
    fig, axs = plt.subplots(2, len(items), figsize=(3.1 * len(items), 6.4), constrained_layout=True)
    for i, key in enumerate(WD.CITY_KEYS):
        for j, (k, name, f) in enumerate(items):
            ax = axs[i, j]
            p, y = f(recs[key][k])
            if len(p) < 20:
                ax.set_axis_off(); continue
            c = calib(p, y)
            lim = max(c[:, 0].max(), c[:, 1].max()) * 1.08
            ax.plot([0, lim], [0, lim], color=GRID, lw=1.2)
            ax.plot(c[:, 0], c[:, 1], "o-", color=BLUE, ms=4, lw=1.5)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_title(f"{CITY_NAME[key]}: {name}\n(n = {len(p):,}, observed {np.mean(y):.3f})", fontsize=8)
            ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel("observed share", fontsize=8)
            ax.set_xlabel("predicted probability", fontsize=8)
    fig.suptitle("Calibration on test days (deciles of the predicted probability; the diagonal is perfect)", fontsize=11)
    fig.savefig(path, dpi=115); plt.close(fig)


def plot_hidden(recs, hid, path):
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    j = 0
    for key in WD.CITY_KEYS:
        for kind, g in hidden.frames[key]:
            ax = axs[j]; j += 1
            ax.scatter(g.trait, g.p, s=9, color=BLUE if kind == "offer" else "#1baf7a", alpha=.6)
            h = hid[key][kind]
            ax.set_title(f"{CITY_NAME[key]}: {len(g)} drivers with 3+ {kind}s on test days\n"
                         f"r(prediction, hidden trait) = {h['r_model_vs_trait']:.2f}; "
                         f"r(observed answers, trait) = {h['r_observed_vs_trait']:.2f}", fontsize=8.5)
            ax.set_xlabel("hidden " + ("accept bias" if kind == "offer" else "order compliance"), fontsize=8)
            ax.set_ylabel("mean predicted P(" + ("accept" if kind == "offer" else "follow") + ")", fontsize=8)
            ax.tick_params(labelsize=7.5); ax.grid(color=GRID, lw=.5); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Has the model inferred each driver's hidden traits from the history it reads? (never an input)", fontsize=11)
    fig.savefig(path, dpi=115); plt.close(fig)


if __name__ == "__main__":
    main()
