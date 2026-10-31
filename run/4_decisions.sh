#!/usr/bin/env bash
# Phase 4 of the slides: decisions made with the world model.
# Repositioning (RL alone, imitation, imitation + GRPO), matching riders to drivers, and drivers
# as small earning-maximising agents. Every agent is then carried out in the simulator.
source "$(dirname "$0")/lib.sh"
PHASE="4_decisions"
PHASE_TITLE="Phase 4 - Decisions (repositioning, matching, driver agents)"
PHASE_DESC="Needs phase 3. Long GPU training; each trained agent is saved in data/wm3_models/."
PHASE_RESULTS="wm3/stage5/28_*.png (repositioning), wm3/stage6/34_*.png (matching), wm3/stage7/34_*.png (driver agents)"

M=data/wm3_models
AG5="rl_pre rl_scratch bc_pre bc_scratch rl_bc_pre rl_bc_scratch"
AG67="mbc_pre+m mbc_scratch+m mbc_pre+m@start mbc_pre+m@drv_pre"

# ---- full-run ----
step "Start states from real days" 12 data \
  "data/sf_hd/wm3_rl/starts/val_*.pkl data/pune_hd/wm3_rl/starts/val_*.pkl" \
  "$PY scripts/25_wm3_rl_starts.py"

step "RL only, from the pretrained world model (rl_pre)" 375 train \
  "$M/rl_pre.pt" \
  "$PY scripts/26_wm3_grpo.py --init wm3_exo --tag rl_pre"

step "RL only, from random weights (rl_scratch)" 225 train \
  "$M/rl_scratch.pt" \
  "$PY scripts/26_wm3_grpo.py --init none --tag rl_scratch"

step "How noisy one driver's effect is at each horizon (2 parallel jobs)" 30 sim \
  "outputs/wm3/stage5/probe/probe_sf_hd.csv outputs/wm3/stage5/probe/probe_pune_hd.csv" \
  "$PY scripts/29_wm3_rl_probe.py sf_hd 4 15" \
  "$PY scripts/29_wm3_rl_probe.py pune_hd 3 11"

step "Record the platform rule's choices, imagined" 17 data \
  "data/sf_hd/wm3_rl/bc_val.pkl data/pune_hd/wm3_rl/bc_val.pkl" \
  "$PY scripts/30_wm3_rl_bc.py --collect"

step "Imitate the rule (pretrained and from scratch)" 26 train \
  "$M/bc_pre.pt $M/bc_scratch.pt" \
  "$PY scripts/30_wm3_rl_bc.py --init wm3_exo --tag bc_pre && $PY scripts/30_wm3_rl_bc.py --init none --tag bc_scratch"

step "Imitation + GRPO, pretrained (rl_bc_pre)" 120 train \
  "$M/rl_bc_pre.pt" \
  "$PY scripts/26_wm3_grpo.py --init-agent bc_pre --tag rl_bc_pre --iters 40 --beta 0.1 --lr-scalar 1e-3"

step "Imitation + GRPO, from scratch (rl_bc_scratch)" 120 train \
  "$M/rl_bc_scratch.pt" \
  "$PY scripts/26_wm3_grpo.py --init none --init-agent bc_scratch --tag rl_bc_scratch --iters 40 --beta 0.1 --lr-scalar 1e-3"

step "Carry the 6 repositioning agents out in the simulator (2 parallel jobs)" 500 sim \
  "" \
  "$PY scripts/27_wm3_rl_execute.py --city sf_hd --seed0 700 --n-seeds 20 --agents $AG5" \
  "$PY scripts/27_wm3_rl_execute.py --city pune_hd --seed0 700 --n-seeds 12 --agents $AG5"

step "Record the platform's matching offers, imagined" 15 data \
  "data/sf_hd/wm3_rl/match_val.pkl data/pune_hd/wm3_rl/match_val.pkl" \
  "$PY scripts/31_wm3_match_bc.py --collect"

step "Matching: imitate the platform (pretrained and from scratch)" 135 train \
  "$M/mbc_pre.pt $M/mbc_scratch.pt" \
  "$PY scripts/31_wm3_match_bc.py --init-agent rl_bc_pre --tag mbc_pre && $PY scripts/31_wm3_match_bc.py --init-agent rl_bc_scratch --tag mbc_scratch"

step "Matching: GRPO (mrl_pre)" 160 train \
  "$M/mrl_pre.pt" \
  "$PY scripts/32_wm3_match_grpo.py --init-agent mbc_pre --tag mrl_pre"

step "Drivers as earning-maximising agents choosing from menus (drv_pre)" 215 train \
  "$M/drv_pre.pt" \
  "$PY scripts/33_wm3_driver_grpo.py --platform mbc_pre --tag drv_pre"

step "Carry matching and driver agents out in the simulator (2 parallel jobs)" 720 sim \
  "" \
  "$PY scripts/27_wm3_rl_execute.py --city sf_hd --seed0 700 --n-seeds 20 --no-imagine --agents $AG67" \
  "$PY scripts/27_wm3_rl_execute.py --city pune_hd --seed0 700 --n-seeds 12 --no-imagine --agents $AG67"

# ---- eval ----
step "Stage 5 figures: repositioning agents in the simulator and in imagination" 1 eval \
  "outputs/wm3/stage5/28_execute.png outputs/wm3/stage5/28_training.png outputs/wm3/stage5/28_versions.png outputs/wm3/stage5/28_scatter.png" \
  "$PY scripts/28_wm3_rl_report.py --tags rl_pre rl_scratch rl_bc_pre rl_bc_scratch"

step "Stage 6 and 7 figures: matching and driver agents" 1 eval \
  "outputs/wm3/stage6/34_execute.png outputs/wm3/stage6/34_imitation.png outputs/wm3/stage6/34_training.png outputs/wm3/stage7/34_execute.png outputs/wm3/stage7/34_drivers_training.png" \
  "$PY scripts/34_wm3_match_report.py --platform mbc_pre --drivers drv_pre"

EVAL_INPUTS="data/sf_hd/wm3_rl/exec/rec_seed7??.json data/pune_hd/wm3_rl/exec/rec_seed7??.json
  outputs/wm3/stage5/eval_rl_pre.csv outputs/wm3/stage5/eval_rl_scratch.csv outputs/wm3/stage5/eval_rl_bc_pre.csv outputs/wm3/stage5/eval_rl_bc_scratch.csv
  outputs/wm3/stage5/probe/probe_sf_hd.csv outputs/wm3/stage5/probe/probe_pune_hd.csv
  outputs/wm3/stage6/31_mbc_pre.csv outputs/wm3/stage6/31_mbc_scratch.csv outputs/wm3/stage6/eval_mrl_pre.csv outputs/wm3/stage7/eval_drv_pre.csv"
# earlier versions of the RL method, kept from the original development run (drawn in 28_versions.png when present)
EVAL_OPTIONAL="outputs/wm3/stage5/v*/eval_rl_pre.csv"

run_phase "$@"
