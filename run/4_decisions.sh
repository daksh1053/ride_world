#!/usr/bin/env bash
# Phase 4 of the slides: decisions made with the world model.
# Repositioning (RL alone, imitation, imitation + GRPO), matching riders to drivers, and
# drivers as small earning-maximising agents. Every agent is then carried out in the simulator.
source "$(dirname "$0")/lib.sh"
PHASE="4_decisions"
PHASE_TITLE="Phase 4 - Decisions (repositioning, matching, driver agents)"
PHASE_DESC="Needs phase 3. Long GPU training; each trained agent is saved in data/wm3_models/."
PHASE_RESULTS="outputs/wm3/stage5/28_execute.png, outputs/wm3/stage6/stage6_report.html, outputs/wm3/stage7/stage7_report.html"

M=data/wm3_models
AG5="rl_pre rl_scratch bc_pre bc_scratch rl_bc_pre rl_bc_scratch"
AG67="mbc_pre+m mbc_scratch+m mbc_pre+m@start mbc_pre+m@drv_pre"

step "Start states from real days" 12 data \
  "data/sf_hd/wm3_rl/starts/val_*.pkl data/pune_hd/wm3_rl/starts/val_*.pkl" \
  "$PY scripts/25_wm3_rl_starts.py"

step "RL only, from the pretrained world model (rl_pre)" 375 train \
  "$M/rl_pre.pt" \
  "$PY scripts/26_wm3_grpo.py --init wm3_exo --tag rl_pre"

step "RL only, from random weights (rl_scratch)" 225 train \
  "$M/rl_scratch.pt" \
  "$PY scripts/26_wm3_grpo.py --init none --tag rl_scratch"

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
  "outputs/wm3/stage5/28_execute.png" \
  "$PY scripts/27_wm3_rl_execute.py --city sf_hd --seed0 700 --n-seeds 20 --agents $AG5" \
  "$PY scripts/27_wm3_rl_execute.py --city pune_hd --seed0 700 --n-seeds 12 --agents $AG5"

step "Stage 5 figures" 3 eval \
  "outputs/wm3/stage5/28_execute.png outputs/wm3/stage5/28_versions.png" \
  "$PY scripts/28_wm3_rl_report.py --tags rl_pre rl_scratch rl_bc_pre rl_bc_scratch"

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
  "outputs/wm3/stage7/34_execute.csv" \
  "$PY scripts/27_wm3_rl_execute.py --city sf_hd --seed0 700 --n-seeds 20 --no-imagine --agents $AG67" \
  "$PY scripts/27_wm3_rl_execute.py --city pune_hd --seed0 700 --n-seeds 12 --no-imagine --agents $AG67"

step "Stage 6 and 7 figures" 3 eval \
  "outputs/wm3/stage6/34_execute.png outputs/wm3/stage7/34_execute.png" \
  "$PY scripts/34_wm3_match_report.py --platform mbc_pre --drivers drv_pre"

run_phase "$@"
