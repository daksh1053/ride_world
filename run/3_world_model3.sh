#!/usr/bin/env bash
# Phase 3 of the slides: world model 3, which models every driver, request and road.
# Rebuild the entity-level world, build the data, train, evaluate, imagine hours, and test
# the response to repositioning orders against simulator twins.
source "$(dirname "$0")/lib.sh"
PHASE="3_world_model3"
PHASE_TITLE="Phase 3 - World model 3 (every driver, rider and road)"
PHASE_DESC="Needs the world model 2 seasons from phase 2 (step 4). The trained model is in data/wm3_models/wm3_exo.pt."
PHASE_RESULTS="outputs/wm3/stage1 to stage4, e.g. outputs/wm3/stage1/sf_hd_world.html, outputs/wm3/stage3/sf_hd_imagined_vs_real.html"

step "Rebuild the world every 30 s and check it against the simulator" 5 eval \
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

step "Next-step test scores and interactive prediction maps" 5 eval \
  "outputs/wm3/stage2/19_eval.json outputs/wm3/stage2/sf_hd_predictions.html" \
  "$PY scripts/19_wm3_eval.py --tag wm3_exo && $PY scripts/21_wm3_pred_page.py --tag wm3_exo"

step "Imagine whole hours and compare with the simulator" 15 eval \
  "outputs/wm3/stage3/20_rollout.json outputs/wm3/stage3/24_hourly.png" \
  "$PY scripts/20_wm3_rollout.py --tag wm3_exo && $PY scripts/24_wm3_hourly.py --tags wm3_exo"

step "Simulator twins with and without an extra order (2 parallel jobs)" 240 data \
  "data/sf_hd/wm3_twins/rec_seed512.json data/pune_hd/wm3_twins/rec_seed511.json" \
  "$PY scripts/22_wm3_twins.py --city sf_hd --seed0 500 --n-seeds 13" \
  "$PY scripts/22_wm3_twins.py --city pune_hd --seed0 500 --n-seeds 12"

step "Model's response to the orders vs the twins" 75 eval \
  "outputs/wm3/stage4/23_counterfactual.json outputs/wm3/stage4/23_scatter.png" \
  "$PY scripts/23_wm3_counterfactual.py --tag wm3_exo"

run_phase "$@"
