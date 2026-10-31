"""Tables of the saved scores of world models 2 and 3, read from the small JSON files the test steps write.

    .venv/bin/python scripts/39_saved_results.py --phase 2    # world model 2 -> outputs/wm2/saved_results.md
    .venv/bin/python scripts/39_saved_results.py --phase 3    # world model 3 -> outputs/wm3/saved_results.md

The test steps themselves need the full held-out data (several GB); this step needs only their
saved results, so it is the evaluation step of phases 2 and 3 in the run/ scripts.
Reads:
  phase 2  outputs/wm2/step3/results.json (scores on held-out days), outputs/wm2/step4d_cf2/13_cf_eval.json
           (response to an order vs simulator twins), outputs/wm2/step5/14_plan.json (choosing an order)
  phase 3  outputs/wm3/stage1/16_stage1.json (rebuilt state vs simulator), stage2/19_eval.json (next 30 s),
           stage3/20_rollout.json (imagined hour), stage3/24_hourly.json (events per day),
           stage4/23_counterfactual.json (response to orders vs simulator twins)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world import cities

CITY = {"sf_hd": "San Francisco", "pune_hd": "Pune"}


def load(rel):
    return json.loads((cities.OUTPUTS / rel).read_text())


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def phase2():
    md = ["# World model 2 (64 x 64 grid): saved scores\n"]
    res = load("wm2/step3/results.json")
    md.append("## Scores on held-out days (`step3/results.json`)\n\nMean Poisson negative log-likelihood of the next "
              "5-minute counts per grid cell, lower is better.\n")
    md.append(table(["model"] + [CITY[c] for c in CITY],
                    [[m] + [res[m].get(c, {}).get("poisson_mean", "-") for c in CITY] for m in res]))
    cf = load("wm2/step4d_cf2/13_cf_eval.json")
    md.append("\n## Response to a repositioning order (`step4d_cf2/13_cf_eval.json`)\n\nThe model's change in the "
              "counts after an order, against the true change in simulator twins. Relative error 1 means the model "
              "predicts no effect at all; map correlation is over a 4 x 4 grid of areas.\n")
    rows = []
    for c in CITY:
        for m, v in cf.get(c, {}).items():
            if isinstance(v, dict) and "relative_error" in v:
                rows.append([CITY[c], m, cf[c]["n_pairs"], v["relative_error"], v["map_corr_4x4"]])
    md.append(table(["city", "model", "twin pairs", "relative error", "map correlation"], rows))
    plan = load("wm2/step5/14_plan.json")
    md.append("\n## Choosing an order (`step5/14_plan.json`)\n\nCandidate orders ranked by the model against their "
              "true effect in the simulator; regret is the pickups lost by taking the model's choice.\n")
    rows = []
    for c in CITY:
        for m, v in plan.get(c, {}).items():
            if isinstance(v, dict) and "spearman_pickups" in v:
                rows.append([CITY[c], m, len(plan[c]["cands"]), v["spearman_pickups"], v["regret_pickups"]])
    md.append(table(["city", "model", "candidates", "rank correlation (pickups)", "regret (pickups)"], rows))
    return "wm2/saved_results.md", md


def phase3():
    md = ["# World model 3 (every driver, request and road): saved scores\n"]
    s1 = load("wm3/stage1/16_stage1.json")
    md.append("## Stage 1: the world rebuilt from logs vs the simulator (`stage1/16_stage1.json`)\n\nShare of "
              "5-minute zone counts that match the simulator exactly over a whole day.\n")
    md.append(table(["city", "day", "idle drivers", "busy drivers", "pending riders"],
                    [[CITY[c], s1[c]["day"], s1[c]["idle_zone_match"], s1[c]["busy_zone_match"],
                      s1[c]["pending_zone_match"]] for c in CITY if c in s1]))
    ev = load("wm3/stage2/19_eval.json")
    md.append("\n## Stage 2: the next 30 seconds on held-out days (`stage2/19_eval.json`)\n\nNegative "
              "log-likelihood per event, lower is better; the baseline is the historical rate.\n")
    rows, wins = [], {}
    for c in CITY:
        for head, v in ev.get(c, {}).items():
            if isinstance(v, dict) and "model" in v and "baseline" in v:
                rows.append([CITY[c], head, v["model"], v["baseline"], v.get("n", "-")])
                wins.setdefault(c, [0, 0])
                wins[c][0] += v["model"] < v["baseline"]
                wins[c][1] += 1
    md.append(table(["city", "event", "model", "baseline", "cases"], rows))
    md.append("\n" + "; ".join(f"{CITY[c]}: model better on {w} of {n} event types" for c, (w, n) in wins.items()) + "\n")
    ro = load("wm3/stage3/20_rollout.json")
    md.append("\n## Stage 3: imagined hours vs the simulator (`stage3/20_rollout.json`)\n\nRelative error of city "
              "totals after 15, 30 and 60 imagined minutes; persistence assumes nothing changes.\n")
    rows = []
    for c in CITY:
        for k, v in ro.get(c, {}).items():
            if isinstance(v, dict) and "model_rel_err" in v:
                rows.append([CITY[c], k, v["model_rel_err"], v["persistence_rel_err"], v["truth_in_range"]])
    md.append(table(["city", "quantity @ horizon", "model", "persistence", "truth inside the imagined range"], rows))
    hr = load("wm3/stage3/24_hourly.json")
    md.append("\n## Stage 3: rare events over whole days (`stage3/24_hourly.json`)\n")
    md.append(table(["model / city / event", "imagined per day", "true per day", "hourly error share"],
                    [[k, v["pred_day"], v["true_day"], v["hourly_abs_err_share"]] for k, v in hr["summary"].items()]))
    cf = load("wm3/stage4/23_counterfactual.json")
    md.append("\n## Stage 4: response to repositioning orders vs simulator twins (`stage4/23_counterfactual.json`)\n\n"
              "r is the correlation of the model's effect with the true effect over all twin pairs; mae zero is the "
              "error of predicting no effect.\n")
    rows = []
    for c in CITY:
        for k, v in cf.get(c, {}).items():
            if isinstance(v, dict) and "r" in v:
                rows.append([CITY[c], k, v["r"], v["mae_model"], v["mae_zero"], v["mean_true"], v["mean_model"]])
    md.append(table(["city", "quantity @ minutes", "r", "mae model", "mae zero", "true effect", "model effect"], rows))
    return "wm3/saved_results.md", md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[2, 3], required=True)
    rel, md = (phase2 if ap.parse_args().phase == 2 else phase3)()
    out = cities.OUTPUTS / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print("wrote", out)
