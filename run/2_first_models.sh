#!/usr/bin/env bash
# Phase 2 of the slides: the first attempts.
# World model 1 (counts per zone, all 15 cities) and world model 2 (counts per cell of a 64 x 64
# grid, high-detail SF and Pune). Both predict counts well but miss the effect of platform actions.
source "$(dirname "$0")/lib.sh"
PHASE="2_first_models"
PHASE_TITLE="Phase 2 - First attempts (world models 1 and 2)"
PHASE_DESC="Needs phase 1. Zone-count model on 15 cities, then the 64 x 64 grid transformer and its order test."
PHASE_RESULTS="wm/fig_*.png, wm/results.md, wm/06_wm_viewer.html (world model 1), wm2/saved_results.md (world model 2)"

CITIES="sf pune navimumbai thane arlington nagpur coimbatore chandigarh pittsburgh portland panaji udaipur puducherry boulder annarbor"
SIZED="navimumbai thane arlington nagpur coimbatore chandigarh pittsburgh portland panaji udaipur puducherry boulder annarbor"
each() { local pat=$1 out="" c; shift; for c in "$@"; do out="$out ${pat//@/$c}"; done; echo "$out"; }
M=data/wm2_models

# ---- full-run ----
step "World model 1: simulate 405 training days (12 workers)" 40 data \
  "data/wm_runs.csv" \
  "$PY scripts/05_wm_generate.py --city all --workers 12"

step "World model 1: train all models and run experiments E1-E6" 50 train \
  "outputs/wm/summary.json data/wm_models/gru_cons_os.pt" \
  "$PY scripts/06_wm_experiments.py"

step "World model 2: simulate 340 days in 6 seasons (3 at a time)" 360 data \
  "data/sf_hd/wm2/days_seed1.csv data/sf_hd/wm2/days_seed3.csv data/pune_hd/wm2/days_seed1.csv data/pune_hd/wm2/days_seed2.csv data/pune_hd/wm2/days_seed3.csv data/pune_hd/wm2/days_seed4.csv" \
  "$PY scripts/10_hd_dataset.py --city pune_hd --days 30 --start 2025-03-06 --seed 3 --blocks" \
  "$PY scripts/10_hd_dataset.py --city pune_hd --days 30 --start 2025-04-05 --seed 4 --blocks" \
  "$PY scripts/10_hd_dataset.py --city sf_hd --days 60 --start 2024-11-06 --seed 3 --blocks" \
  "$PY scripts/10_hd_dataset.py --city pune_hd --days 50 --start 2025-06-24 --seed 2" \
  "$PY scripts/10_hd_dataset.py --city pune_hd --days 50 --start 2025-05-05 --seed 1" \
  "$PY scripts/10_hd_dataset.py --city sf_hd --days 120 --start 2025-01-06 --seed 1"

step "World model 2: pack the grid data (about 12 GB)" 10 data \
  "data/sf_hd/wm2/store/dirs.npy data/pune_hd/wm2/store/dirs.npy" \
  "$PY scripts/11_wm2_train.py --build && $PY scripts/11_wm2_train.py --build-dirs"

step "World model 2: train the first grid models (transformer, CNN, rollout fine-tune)" 145 train \
  "$M/st_small.pt $M/cnn.pt $M/st_ro.pt" \
  "$PY scripts/11_wm2_train.py --model st --steps 20000 --tag st_small && $PY scripts/11_wm2_train.py --model cnn --steps 6000 --tag cnn && $PY scripts/11_wm2_train.py --finetune st_small --rollout 4 --steps 2500 --lr 1e-4 --eval-every 500 --tag st_ro"

step "World model 2: train the best grid model (st4_skip_ros, 4 runs)" 255 train \
  "$M/st4_blocks.pt $M/st4_focus_ft.pt $M/st4_skip_ft.pt $M/st4_skip_ros.pt" \
  "$PY scripts/11_wm2_train.py --model st --dirs --steps 20000 --tag st4_blocks --outdir step4 && $PY scripts/11_wm2_train.py --finetune st4_blocks --dirs --act-focus 0.3 --steps 5000 --lr 2e-4 --tag st4_focus_ft --outdir step4 && $PY scripts/11_wm2_train.py --finetune st4_focus_ft --skip --dirs --act-focus 0.3 --steps 5000 --lr 2e-4 --tag st4_skip_ft --outdir step4 && $PY scripts/11_wm2_train.py --finetune st4_skip_ft --dirs --act-focus 0.5 --rollout 6 --roll-feed sample --steps 2500 --lr 1e-4 --eval-every 500 --tag st4_skip_ros --outdir step4d"

step "World model 2: scores on held-out days, rollouts, prediction pages" 10 test \
  "outputs/wm2/step3/results.json outputs/sf_hd/12_wm2_predictions_st_ro.html" \
  "$PY scripts/11_wm2_train.py --eval st_small st_ro cnn && $PY scripts/12_wm2_analysis.py --tag st_ro"

step "World model 2: simulator twins with and without an order (2 parallel jobs)" 50 sim \
  "data/sf_hd/wm2_cf2/pairs.json data/pune_hd/wm2_cf2/pairs.json" \
  "$PY scripts/13_cf_truth.py --city sf_hd --pairs 8 --t-order 28950 --out wm2_cf2" \
  "$PY scripts/13_cf_truth.py --city pune_hd --pairs 8 --t-order 28950 --out wm2_cf2"

step "World model 2: the model's order effect vs the true effect" 5 test \
  "outputs/wm2/step4d_cf2/13_cf_eval.png" \
  "$PY scripts/13b_cf_eval.py --tags st4_skip_ros --cf wm2_cf2 --outdir step4d_cf2"

step "World model 2: simulate candidate orders (2 parallel jobs)" 60 sim \
  "data/sf_hd/wm2_plan2/rec_seed211.json data/pune_hd/wm2_plan2/rec_seed211.json" \
  "$PY scripts/14_wm_plan.py --city sf_hd --seeds 12" \
  "$PY scripts/14_wm_plan.py --city pune_hd --seeds 12"

step "World model 2: let the models choose an order" 5 test \
  "outputs/wm2/step5/14_plan.png" \
  "$PY scripts/14_wm_plan.py --score st_ro st4_skip_ros"

# ---- eval ----
step "World model 1: figures and results table" 1 eval \
  "outputs/wm/fig_rollouts.png outputs/wm/fig_cities.png outputs/wm/fig_interactability.png outputs/wm/fig_fairness.png outputs/wm/results.md" \
  "$PY scripts/06c_wm_figures.py"

step "World model 1: imagined vs real page" 1 eval \
  "outputs/wm/06_wm_viewer.html" \
  "$PY scripts/06b_wm_viewer.py"

step "World model 2: tables of the saved scores" 1 eval \
  "outputs/wm2/saved_results.md" \
  "$PY scripts/39_saved_results.py --phase 2"

EVAL_INPUTS="outputs/wm2/step3/results.json outputs/wm2/step4d_cf2/13_cf_eval.json outputs/wm2/step5/14_plan.json
  outputs/wm/e1_one_step_nll.csv outputs/wm/e2_rollouts.csv outputs/wm/e3_policy_gaps.csv outputs/wm/e4_generalization.csv
  outputs/wm/e5_plausibility.csv outputs/wm/e6_fairness_obs.csv outputs/wm/summary.json
  data/wm_models/gru.pt data/wm_models/gru_cons_os.pt
  $(each data/@/meta.json $CITIES) $(each data/@/zone_geoms.geojson $CITIES) $(each data/@/fleet.json $SIZED)
  $(each data/@/wm/static.npz $CITIES) $(each 'data/@/wm/test_*_nearest.npz' $CITIES) $(each 'data/@/wm/test_*_p2_ilp.npz' $CITIES)"

run_phase "$@"
