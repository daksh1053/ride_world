"""Stage 6: train world models on the generated data and test the survey's properties.

    .venv/bin/python scripts/06_wm_experiments.py            # everything (~1 h on an RTX 3050)

Experiments (results -> outputs/wm/):
  E1 accuracy          one-step test NLL per city and model (formulation eq. 31)
  E2 persistence       open-loop rollout error vs horizon (5 min .. 1 h)
  E3 interactability   with vs without action inputs; unseen dispatch policy (P2 ILP);
                       predicted vs actual gaps between policies on the same day
  E4 generalization    train without the 5 small cities -> test on them;
                       train on Indian cities -> test on US cities
  E5 plausibility      rate of impossible predictions (pickups > requests available)
  E6 fairness Phi      per-zone completion rate s_z (eq. 25) over a 3-h rollout: predicted
                       vs actual, and its spread across zones (eq. 32)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from world import cities, wm_data
from world import wm_models as M

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--cities", nargs="+", default=None, help="restrict to these cities (trial runs)")
_ap.add_argument("--name", default="wm", help="output / model folder name")
_args = _ap.parse_args()
OUT = cities.OUTPUTS / _args.name
MODELS = cities.DATA / f"{_args.name}_models"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL = _args.cities or [k for t in cities.TIERS.values() for k in t]
SMALL = [k for k in cities.TIERS["small"] if k in ALL]
US = [k for k in ALL if cities.CITIES[k].country == "US"]
TRAIN_POL = ["nearest", "nearest_norepo", "p2_nm", "p1_greedy", "p2_wdf"]
STARTS = list(range(24, 276, 12))        # rollout origins: 02:00 .. 22:00 every hour
H = 12                                   # 1 hour
MAIN = "gru_cons_os"                     # the model used for E3 policy gaps and E4 transfer
log = lambda *a: print(*a, flush=True)  # noqa: E731


def build(kind, train_runs, X, use_actions=True, tag=""):
    name = f"{kind}{'' if use_actions else '_noact'}{tag}"
    path = MODELS / f"{name}.pt"
    base = kind[:-3] if kind.endswith("_os") else kind
    m = M.Net(base, use_actions).to(DEV)
    if path.exists():
        m.load_state_dict(torch.load(path, map_location=DEV)["state"])
        m.kind = kind
        log(f"  loaded {name}")
        return m
    t0 = time.time()
    log(f"  training {name} on {len({r.city for r in train_runs})} cities, {len(train_runs)} runs")
    if kind.endswith("_os"):
        warm = MODELS / f"{base}{'' if use_actions else '_noact'}{tag}.pt"
        m.load_state_dict(torch.load(warm, map_location=DEV)["state"])
        M.finetune_overshoot(m, train_runs, X, log=log)
    else:
        M.train(m, train_runs, X, epochs=6 if base == "gru" else 8, log=log)
    m.kind = kind
    M.save(m, path, {"name": name, "train_cities": sorted({r.city for r in train_runs}),
                     "minutes": round((time.time() - t0) / 60, 1)})
    return m


# ---------------------------------------------------------------- evaluation helpers

@torch.no_grad()
def one_step_nll(model, runs):
    """Mean per-count NLL of next-step targets, teacher forced (eq. 31, H = 1)."""
    out = []
    for r in runs:
        h = model.init_state(r.obs.shape[1], DEV)
        tot = torch.zeros(M.N_OBS, device=DEV)
        n = 0
        for k in range(2, r.act.shape[0]):
            lr_k, h = model(M.features(r.obs[[k, k - 1, k - 2]], r.act[k], r.ctx[k], r), h)
            if k >= 14:                                       # after the state has warmed up
                y = r.obs[k + 1]
                tot += (lr_k.exp() - y * lr_k + torch.lgamma(y + 1)).mean(0)
                n += 1
        out.append((tot / n).cpu().numpy())
    return np.array(out)                                       # (runs, 6)


def nll_of_mean(pred, y):
    rate = pred.clamp(min=1e-3)
    return (rate - y * rate.log() + torch.lgamma(y + 1)).mean((0, 1))


@torch.no_grad()
def rollout_errors(predict, runs):
    """Per horizon h = 1..H, averaged over runs and origins:
      nll      Poisson NLL of the observed counts under the predicted means (proper score)
      tot_err  relative error of city totals per target (new, pickups, cancels, idle, busy, pending)
      state_mae  zone-level MAE of idle / busy / pending
    plus `diverged`: share of rollouts whose predicted fleet ends > 2x or < 0.5x the actual."""
    nll = torch.zeros(H, device=DEV)
    tot = torch.zeros(H, M.N_OBS, device=DEV)
    smae = torch.zeros(H, 3, device=DEV)
    n, div = 0, 0
    for r in runs:
        for k0 in STARTS:
            p = predict(r, k0)
            if p is None:
                return None
            y = r.obs[k0 + 1:k0 + 1 + H]
            rate = p.clamp(min=1e-3)
            nll += (rate - y * rate.log() + torch.lgamma(y + 1)).mean((1, 2))
            tot += (p.sum(1) - y.sum(1)).abs() / y.sum(1).clamp(min=1)
            smae += (p[:, :, 3:] - y[:, :, 3:]).abs().mean(1)
            ratio = p[-1, :, 3:5].sum() / y[-1, :, 3:5].sum().clamp(min=1)
            div += int(ratio > 2 or ratio < 0.5)
            n += 1
    return {"nll": (nll / n).cpu().numpy(), "tot_err": (tot / n).cpu().numpy(),
            "state_mae": (smae / n).cpu().numpy(), "diverged": div / n}


def predictor(model):
    return lambda r, k0: M.rollout(model, r, k0, H)


# ---------------------------------------------------------------- experiments

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"device {DEV}; loading data for {len(ALL)} cities")
    train_runs = M.load_runs(ALL, "train", DEV)
    test_runs = M.load_runs(ALL, "test", DEV)
    seen = [r for r in test_runs if r.policy in TRAIN_POL]
    unseen = [r for r in test_runs if r.policy not in TRAIN_POL]
    log(f"  {len(train_runs)} train runs, {len(seen)} test runs (seen policies), {len(unseen)} unseen-policy runs")
    X = M.precompute(train_runs)
    seasonal = M.Seasonal(train_runs)
    res = {"cities": ALL, "n_train_runs": len(train_runs), "n_test_runs": len(test_runs)}

    models = {}
    for kind in ["glm", "mlp", "gru", "gru_os", "gru_cons", "gru_cons_os"]:
        models[kind] = build(kind, train_runs, X)
    for kind in ["mlp", "gru"]:
        models[f"{kind}_noact"] = build(kind, train_runs, X, use_actions=False)

    # E1: one-step NLL per city
    log("E1 one-step accuracy")
    rows = []
    for r in seen:
        rows.append({"city": r.city, "policy": r.policy, "date": r.date, "model": "persistence",
                     **dict(zip(wm_data.OBS, nll_of_mean(r.obs[13:-1], r.obs[14:]).cpu().numpy()))})
        s = seasonal.predict(r, 13, r.act.shape[0] - 13)
        if s is not None:
            rows.append({"city": r.city, "policy": r.policy, "date": r.date, "model": "seasonal",
                         **dict(zip(wm_data.OBS, nll_of_mean(s, r.obs[14:]).cpu().numpy()))})
    for name, m in models.items():
        nll = one_step_nll(m, seen)
        for r, v in zip(seen, nll):
            rows.append({"city": r.city, "policy": r.policy, "date": r.date, "model": name,
                         **dict(zip(wm_data.OBS, v))})
    e1 = pd.DataFrame(rows)
    e1["nll"] = e1[wm_data.OBS].mean(axis=1)
    e1.to_csv(OUT / "e1_one_step_nll.csv", index=False)
    log(e1.groupby("model").nll.mean().sort_values().round(4).to_string())

    # E2: rollout error vs horizon
    log("E2 rollouts")
    e2 = {"persistence": rollout_errors(lambda r, k0: M.baseline_persistence(r, k0, H), seen),
          "seasonal": rollout_errors(lambda r, k0: seasonal.predict(r, k0, H), seen)}
    for name in ["glm", "mlp", "gru", "gru_os", "gru_cons", "gru_cons_os"]:
        e2[name] = rollout_errors(predictor(models[name]), seen)
    rows = []
    for n_, v in e2.items():
        if v is None:
            continue
        for h in range(H):
            rows.append({"model": n_, "h": h + 1, "nll": float(v["nll"][h]), "diverged": v["diverged"],
                         **{f"tot_err_{t}": float(v["tot_err"][h, j]) for j, t in enumerate(wm_data.OBS)},
                         **{f"mae_{t}": float(v["state_mae"][h, j]) for j, t in enumerate(wm_data.OBS[3:])}})
    e2df = pd.DataFrame(rows)
    e2df.to_csv(OUT / "e2_rollouts.csv", index=False)
    log(e2df[e2df.h.isin([1, 6, 12])].set_index(["model", "h"])[["nll", "tot_err_new", "tot_err_pickups", "tot_err_idle", "mae_idle", "diverged"]].round(4).to_string())

    # E3: interactability
    log("E3 interactability")
    e3 = {}
    for kind in ["mlp", "gru"]:
        a, b = one_step_nll(models[kind], seen), one_step_nll(models[f"{kind}_noact"], seen)
        e3[f"{kind}_nll_with_actions"] = float(a.mean())
        e3[f"{kind}_nll_without_actions"] = float(b.mean())
        e3[f"{kind}_idle_nll_with_vs_without"] = [float(a[:, 3].mean()), float(b[:, 3].mean())]
    # where repositioning moves cabs: next-step idle error in zones receiving dispatches
    for kind in ["gru", "gru_noact"]:
        m = models[kind]
        errs = []
        with torch.no_grad():
            for r in seen:
                h = m.init_state(r.obs.shape[1], DEV)
                for k in range(2, r.act.shape[0]):
                    lr_k, h = m(M.features(r.obs[[k, k - 1, k - 2]], r.act[k], r.ctx[k], r), h)
                    hit = r.act[k][:, 1] > 0
                    if k >= 14 and hit.any():
                        errs.append((lr_k.exp()[hit, 3] - r.obs[k + 1][hit, 3]).abs())
        e3[f"{kind}_idle_mae_in_dispatch_target_zones"] = float(torch.cat(errs).mean()) if errs else None
    for name in [MAIN, "mlp"]:
        e3[f"{name}_nll_seen_policies"] = float(one_step_nll(models[name], seen).mean())
        e3[f"{name}_nll_unseen_policy_p2_ilp"] = float(one_step_nll(models[name], unseen).mean())
    # policy gaps: same day, two policies; city-wide pickups + cancels over the next hour
    by_day = {}
    for r in test_runs:
        by_day.setdefault((r.city, r.date), {})[r.policy] = r
    act_d, pred_d, ctl_d = [], [], []
    g, ctl = models[MAIN], models["gru_noact"]
    for (c, d), pols in by_day.items():
        names = sorted(pols)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ra, rb = pols[names[i]], pols[names[j]]
                for k0 in STARTS[::2]:
                    pa, pb = M.rollout(g, ra, k0, H), M.rollout(g, rb, k0, H)
                    ca, cb = M.rollout(ctl, ra, k0, H), M.rollout(ctl, rb, k0, H)
                    for t in (1, 2):                          # pickups, cancels
                        act_d.append(float(ra.obs[k0 + 1:k0 + 1 + H, :, t].sum() - rb.obs[k0 + 1:k0 + 1 + H, :, t].sum()))
                        pred_d.append(float(pa[:, :, t].sum() - pb[:, :, t].sum()))
                        ctl_d.append(float(ca[:, :, t].sum() - cb[:, :, t].sum()))
    act_d, pred_d, ctl_d = np.array(act_d), np.array(pred_d), np.array(ctl_d)
    e3["policy_gap_corr"] = float(np.corrcoef(act_d, pred_d)[0, 1])
    e3["policy_gap_corr_noact_control"] = float(np.corrcoef(act_d, ctl_d)[0, 1])
    e3["policy_gap_sign_agreement"] = float(np.mean(np.sign(act_d) == np.sign(pred_d)))
    e3["policy_gap_n"] = int(len(act_d))
    pd.DataFrame({"actual": act_d, "predicted": pred_d, "noact_control": ctl_d}).to_csv(OUT / "e3_policy_gaps.csv", index=False)
    log(json.dumps(e3, indent=1))
    res["e3"] = e3

    # E4: generalization to unseen cities
    log("E4 generalization")
    e4 = []
    for tag, train_keys, test_keys in [("_nosmall", [k for k in ALL if k not in SMALL], SMALL),
                                       ("_india", [k for k in ALL if k not in US], US)]:
        tr = [r for r in train_runs if r.city in train_keys]
        Xs = [x for x, r in zip(X, train_runs) if r.city in train_keys]
        build(MAIN[:-3] if MAIN.endswith("_os") else MAIN, tr, Xs, tag=tag)
        mt = build(MAIN, tr, Xs, tag=tag)
        te = [r for r in seen if r.city in test_keys]
        for name, m in [(f"pooled {MAIN} (saw these cities)", models[MAIN]), (f"{MAIN} trained{tag}", mt)]:
            nll = one_step_nll(m, te)
            ro = rollout_errors(predictor(m), te)
            e4.append({"split": tag, "model": name, "nll": float(nll.mean()), "rollout_nll_1h": float(ro["nll"][-1]),
                       "tot_err_new_1h": float(ro["tot_err"][-1, 0]), "tot_err_idle_1h": float(ro["tot_err"][-1, 3])})
        p = rollout_errors(lambda r, k0: M.baseline_persistence(r, k0, H), te)
        e4.append({"split": tag, "model": "persistence", "nll": float(np.mean([nll_of_mean(r.obs[13:-1], r.obs[14:]).mean().item() for r in te])),
                   "rollout_nll_1h": float(p["nll"][-1]), "tot_err_new_1h": float(p["tot_err"][-1, 0]),
                   "tot_err_idle_1h": float(p["tot_err"][-1, 3])})
        del Xs
    pd.DataFrame(e4).to_csv(OUT / "e4_generalization.csv", index=False)
    log(pd.DataFrame(e4).round(4).to_string())

    # E5: plausibility and E6: fairness observable, from 3-h evening rollouts (17:00-20:00)
    log("E5/E6 plausibility and fairness observable")
    k0, H3 = 204, 36
    e5, e6 = [], []
    for name in ["glm", "mlp", "gru", "gru_os", "gru_cons", "gru_cons_os"]:
        m = models[name]
        viol, tot, fleet_err = 0, 0, []
        for r in seen:
            p = M.rollout(m, r, k0, H3)
            pend = torch.cat([r.obs[k0][None, :, 5], p[:-1, :, 5]])          # pending at the step start
            bad = p[:, :, 1] > pend + p[:, :, 0] + 0.5                       # pickups > available requests
            viol += int(bad.sum()); tot += bad.numel()
            act_fleet = r.obs[k0 + 1:k0 + 1 + H3, :, 3:5].sum((1, 2))
            fleet_err.append(float(((p[:, :, 3:5].sum((1, 2)) - act_fleet).abs() / act_fleet.clamp(min=1)).mean()))
            new_a = r.obs[k0 + 1:k0 + 1 + H3, :, 0].sum(0)
            pick_a = r.obs[k0 + 1:k0 + 1 + H3, :, 1].sum(0)
            keep = new_a >= 5                                                 # zones with a denominator
            if keep.sum() >= 3:
                s_a = (pick_a / new_a.clamp(min=1))[keep]
                s_p = (p[:, :, 1].sum(0) / p[:, :, 0].sum(0).clamp(min=1e-3))[keep]
                e6.append({"model": name, "city": r.city, "policy": r.policy, "date": r.date,
                           "s_mae": float((s_p - s_a).abs().mean()),
                           "s_corr": float(np.corrcoef(s_p.cpu(), s_a.cpu())[0, 1]) if s_a.std() > 0 else np.nan,
                           "s_std_actual": float(s_a.std()), "s_std_pred": float(s_p.std()),
                           "city_s_actual": float(pick_a.sum() / new_a.sum()),
                           "city_s_pred": float(p[:, :, 1].sum() / p[:, :, 0].sum())})
        e5.append({"model": name, "impossible_pickup_rate": viol / tot, "fleet_size_rel_err_3h": float(np.mean(fleet_err))})
    pd.DataFrame(e5).to_csv(OUT / "e5_plausibility.csv", index=False)
    pd.DataFrame(e6).to_csv(OUT / "e6_fairness_obs.csv", index=False)
    log(pd.DataFrame(e5).round(4).to_string())
    log(pd.DataFrame(e6).groupby("model")[["s_mae", "s_corr", "s_std_actual", "s_std_pred"]].mean().round(4).to_string())

    res["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"done in {res['minutes']} min")


if __name__ == "__main__":
    main()
