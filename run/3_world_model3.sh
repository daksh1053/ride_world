#!/usr/bin/env bash
# Phase 3 of the slides: world model 3, which models every driver, request and road.
# Rebuild the entity-level world, build the data, train, test on held-out days, imagine hours, and
# test the response to repositioning orders against simulator twins.
source "$(dirname "$0")/lib.sh"
PHASE="3_world_model3"
PHASE_TITLE="Phase 3 - World model 3 (every driver, rider and road)"
PHASE_DESC="Needs the world model 2 seasons of phase 2 (its step 3). Every test here reads held-out days (GBs), so all of them run in full-run."
PHASE_RESULTS="wm3/saved_results.md (scores of stages 1-4 in tables)"

# ---- full-run ----
step "Rebuild the world every 30 s and check it against the simulator" 5 test \
  "outputs/wm3/stage1/16_stage1.json outputs/wm3/stage1/sf_hd_world.html" \
  "$PY scripts/16_wm3_world.py"

step "Build the training data (about 32 GB)" 20 data \
  "data/sf_hd/wm3v2/meta.json data/pune_hd/wm3v2/meta.json" \
  "$PY scripts/17_wm3_dataset.py"

step "Train world model 3 (30,000 steps)" 85 train \
  "data/wm3_models/wm3_v2.pt" \
  "$PY scripts/18_wm3_train.py --steps 30000 --eval-every 1500 --tag wm3_v2"

step "Fine-tune: demand gets its own encoder (final model wm3_exo)" 30 train \
  "data/wm3_models/wm3_exo.pt" \
  "$PY scripts/18_wm3_train.py --init wm3_v2 --exo --steps 10000 --lr 2e-4 --tag wm3_exo"

step "Next-30-second scores on held-out days and prediction maps" 5 test \
  "outputs/wm3/stage2/19_eval.json outputs/wm3/stage2/sf_hd_predictions.html" \
  "$PY scripts/19_wm3_eval.py --tag wm3_exo && $PY scripts/21_wm3_pred_page.py --tag wm3_exo"

step "Imagine whole hours of held-out days and compare with the simulator" 15 test \
  "outputs/wm3/stage3/20_rollout.json outputs/wm3/stage3/24_hourly.json" \
  "$PY scripts/20_wm3_rollout.py --tag wm3_exo && $PY scripts/24_wm3_hourly.py --tags wm3_exo"

step "Simulator twins with and without an extra order (2 parallel jobs)" 240 sim \
  "data/sf_hd/wm3_twins/rec_seed512.json data/pune_hd/wm3_twins/rec_seed511.json" \
  "$PY scripts/22_wm3_twins.py --city sf_hd --seed0 500 --n-seeds 13" \
  "$PY scripts/22_wm3_twins.py --city pune_hd --seed0 500 --n-seeds 12"

step "The model's response to the orders vs the twins" 75 test \
  "outputs/wm3/stage4/23_counterfactual.json outputs/wm3/stage4/23_scatter.png" \
  "$PY scripts/23_wm3_counterfactual.py --tag wm3_exo"

# ---- eval ----
step "Tables of the saved scores of stages 1-4" 1 eval \
  "outputs/wm3/saved_results.md" \
  "$PY scripts/39_saved_results.py --phase 3"

EVAL_INPUTS="outputs/wm3/stage1/16_stage1.json outputs/wm3/stage2/19_eval.json outputs/wm3/stage3/20_rollout.json
  outputs/wm3/stage3/24_hourly.json outputs/wm3/stage4/23_counterfactual.json"

run_phase "$@"
