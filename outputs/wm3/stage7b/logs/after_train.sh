#!/bin/bash
# after training ends: simulator test of the round-20 agent, both cities
cd /home/daksh/Desktop/btp/ride_world
while kill -0 569812 2>/dev/null; do sleep 60; done
L=outputs/wm3/stage7b/logs
.venv/bin/python scripts/43_wm3_driver2_execute.py --city sf_hd --drivers drv2_d3_it20 --n-seeds 1 --seed0 700 > $L/43_smoke_sf.log 2>&1 || exit 1
.venv/bin/python scripts/43_wm3_driver2_execute.py --city sf_hd --drivers drv2_d3_it20 --n-seeds 6 > $L/43_sf.log 2>&1
.venv/bin/python scripts/43_wm3_driver2_execute.py --city pune_hd --drivers drv2_d3_it20 --n-seeds 6 > $L/43_pune.log 2>&1
