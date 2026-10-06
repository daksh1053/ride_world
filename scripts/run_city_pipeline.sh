#!/usr/bin/env bash
# Run stages 2 -> 2b -> 3 -> 5 for each city once its stage-1 map exists.
#   scripts/run_city_pipeline.sh chandigarh panaji ...
# Logs to data/pipeline_<city>.log; skips stages whose outputs already exist.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for c in "$@"; do
  until [ -f "data/$c/zone_geoms.geojson" ] && [ -f "outputs/$c/01_explorer.html" ]; do sleep 30; done
  log="data/pipeline_$c.log"
  echo "== $c $(date +%H:%M)" | tee -a "$log"
  [ -f "data/$c/traffic/zone_tt_s.npy" ] || $PY -u scripts/02_traffic.py --city "$c" >> "$log" 2>&1
  [ -f "data/$c/fleet.json" ] || $PY -u scripts/02b_size_fleet.py --city "$c" >> "$log" 2>&1
  $PY -u scripts/03_simulate.py --city "$c" >> "$log" 2>&1
  $PY -u scripts/05_wm_generate.py --city "$c" --workers 5 >> "$log" 2>&1
  echo "   $c done $(date +%H:%M): $(grep -c 'target TTI' "$log") calibrated hours, $(grep '== ' "$log" | tail -1)" | tee -a "$log"
done
echo PIPELINE_DONE
