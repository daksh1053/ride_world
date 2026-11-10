#!/bin/bash
cd /home/daksh/Desktop/btp/ride_world
for cfg in "0.5 2.0" "0.5 3.0" "0.0 3.0"; do
  set -- $cfg
  .venv/bin/python scripts/40_wm3_market_probe.py --menu exclusive --batch 4 --supply $1 --demand $2 --groups 60 40 > outputs/wm3/stage7b/logs/probe_b4_s$1_d$2.log 2>&1
done
.venv/bin/python scripts/41_wm3_market_tune.py > outputs/wm3/stage7b/logs/tune_demand.log 2>&1
