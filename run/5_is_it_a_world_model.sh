#!/usr/bin/env bash
# Phase 5 of the slides: is it a world model?
# Eight imagined hours against the real day, shocks the model never saw, policy ranking, the
# ten-criterion scorecard, and the readable HTML reports for stages 6-8.
source "$(dirname "$0")/lib.sh"
PHASE="5_is_it_a_world_model"
PHASE_TITLE="Phase 5 - Is it a world model? (stage 8)"
PHASE_DESC="Needs phases 3 and 4. Ends with the scorecard and the three HTML reports."
PHASE_RESULTS="wm3/stage8/37_scorecard.png, wm3/stage8/35_*.png, 36_*.png, and wm3/stage{6,7,8}/stage*_report.html"

HDC="sf_hd pune_hd"
ARMS="rule mbc_pre+m price15 supply30 supply60"
TABLES="context drivers drivers_latent events legs offers requests status"
each() { local pat=$1 out="" c; shift; for c in "$@"; do out="$out ${pat//@/$c}"; done; echo "$out"; }
records() { local c a t out=""; for c in $HDC; do for a in $ARMS; do for t in $TABLES; do out="$out data/$c/wm3_rl/exec/seed700_$a/*/$t.parquet"; done; done; done; echo "$out"; }

# ---- full-run ----
step "8-hour imagined rollouts, SF (13 days)" 55 sim \
  "data/sf_hd/wm3_long/seed512.npz" \
  "$PY scripts/35_wm3_long_rollout.py --city sf_hd --seeds 13"

step "8-hour imagined rollouts, Pune (8 days)" 105 sim \
  "data/pune_hd/wm3_long/seed507.npz" \
  "$PY scripts/35_wm3_long_rollout.py --city pune_hd --seeds 8"

step "Unseen shocks, SF (drivers leave, prices x1.5; 20 days)" 85 sim \
  "data/sf_hd/wm3_rl/exec/shock_seed719.json" \
  "$PY scripts/36_wm3_shock.py --city sf_hd --seed0 700 --n-seeds 20"

step "Unseen shocks, Pune (8 days)" 70 sim \
  "data/pune_hd/wm3_rl/exec/shock_seed707.json" \
  "$PY scripts/36_wm3_shock.py --city pune_hd --seed0 700 --n-seeds 8"

step "Imagine the stage 6-7 policies (for the policy ranking)" 45 sim \
  "" \
  "$PY scripts/27_wm3_rl_execute.py --city sf_hd --seed0 700 --n-seeds 20 --agents mbc_pre+m mbc_scratch+m mbc_pre+m@start mbc_pre+m@drv_pre"

step "Record the map replays for the reports" 20 sim \
  "data/sf_hd/wm3_reports/s8_imagined.pkl data/pune_hd/wm3_reports/s8_imagined.pkl" \
  "$PY scripts/38_wm3_reports.py --collect"

# ---- eval ----
step "Long-rollout figures" 1 eval \
  "outputs/wm3/stage8/35_long.json outputs/wm3/stage8/35_long_fan.png outputs/wm3/stage8/35_long_errors.png outputs/wm3/stage8/35_long_rank.png" \
  "$PY scripts/35_wm3_long_rollout.py --report"

step "Shock figures" 1 eval \
  "outputs/wm3/stage8/36_shock.json outputs/wm3/stage8/36_shock_effects.png outputs/wm3/stage8/36_shock_scatter.png outputs/wm3/stage8/36_shock_traj.png" \
  "$PY scripts/36_wm3_shock.py --report"

step "Policy ranking and the ten-criterion scorecard" 1 eval \
  "outputs/wm3/stage8/37_scorecard.png outputs/wm3/stage8/37_scorecard.json outputs/wm3/stage8/37_ranking.png" \
  "$PY scripts/37_wm3_scorecard.py"

step "Build the stage 6, 7 and 8 HTML reports" 2 eval \
  "outputs/wm3/stage6/stage6_report.html outputs/wm3/stage7/stage7_report.html outputs/wm3/stage8/stage8_report.html" \
  "$PY scripts/38_wm3_reports.py --build"

EVAL_INPUTS="data/sf_hd/wm3_long/seed5??.npz data/pune_hd/wm3_long/seed5??.npz
  $(each 'data/@/wm3_rl/exec/shock_seed7??.json' $HDC) $(each 'data/@/wm3_rl/exec/rec_seed7??.json' $HDC)
  outputs/wm3/stage1/16_stage1.json outputs/wm3/stage2/19_eval.json outputs/wm3/stage4/23_counterfactual.json
  $(each data/@/edges.parquet $HDC) $(each data/@/nodes.parquet $HDC) $(each data/@/zones.parquet $HDC)
  $(each data/@/traffic/bg_volume.npy $HDC) $(each data/@/traffic/edge_params.parquet $HDC)
  $(each data/@/wm3_reports/s7_menus.pkl $HDC) $(each data/@/wm3_reports/s8_imagined.pkl $HDC)
  $(records)"
# the report pages show the stage 6-7 figures made by the phase 4 evaluation
EVAL_NEEDS="outputs/wm3/stage6/34_execute.png=4_decisions.sh outputs/wm3/stage6/34_imitation.png=4_decisions.sh
  outputs/wm3/stage6/34_training.png=4_decisions.sh outputs/wm3/stage7/34_execute.png=4_decisions.sh
  outputs/wm3/stage7/34_drivers_training.png=4_decisions.sh"

run_phase "$@"
