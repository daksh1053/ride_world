#!/bin/bash
cd /home/daksh/Desktop/btp/ride_world
L=outputs/wm3/stage7b/logs
P="scripts/40_wm3_market_probe.py --menu exclusive --batch 4 --groups 60 40 --workers 2 2"
.venv/bin/python $P --cities sf_hd --supply 0.5 --demand 3.0 > $L/probe_b4_s0.5_d3.0_sf.log 2>&1
.venv/bin/python $P --cities pune_hd --supply 0.5 --demand 3.0 > $L/probe_b4_s0.5_d3.0_pune.log 2>&1
.venv/bin/python $P --cities pune_hd --supply 0.7 --demand 3.0 > $L/probe_b4_s0.7_d3.0_pune.log 2>&1
.venv/bin/python scripts/41_wm3_market_tune.py > $L/tune_demand.log 2>&1
