#!/usr/bin/env bash
# Phase 1 of the slides: build the world.
# Real road maps of 15 cities, calibrated traffic, one simulated day per city, the two
# fairness papers' methods in every city, and the two high-detail cities (sf_hd, pune_hd).
source "$(dirname "$0")/lib.sh"
PHASE="1_build_world"
PHASE_TITLE="Phase 1 - Building the world"
PHASE_DESC="Maps, traffic, the ride company and the paper methods in 15 cities, plus the high-detail SF and Pune."
PHASE_RESULTS="outputs/index.html (watch every cab), outputs/04_dashboard.html (paper methods), outputs/<city>/"

CITIES="sf pune navimumbai thane arlington nagpur coimbatore chandigarh pittsburgh portland panaji udaipur puducherry boulder annarbor"
HD="sf_hd pune_hd"
each() { local pat=$1 out="" c; shift; for c in "$@"; do out="$out ${pat//@/$c}"; done; echo "$out"; }

step "Download road maps and build zones (17 cities, needs internet)" 150 data \
  "$(each data/@/zone_geoms.geojson $CITIES $HD) $(each outputs/@/01_explorer.html $CITIES $HD)" \
  "$PY scripts/01_import_maps.py --city $CITIES $HD"

step "Land use for the high-detail cities" 5 data \
  "data/sf_hd/hd/landuse.gpkg data/pune_hd/hd/landuse.gpkg" \
  "$PY -c \"from world import cities; from world.hd import landuse; [landuse.build(cities.get(k)) for k in ('sf_hd', 'pune_hd')]\""

step "Calibrate hourly traffic (17 cities)" 150 data \
  "$(each data/@/traffic/calibration.csv $CITIES $HD)" \
  "$PY scripts/02_traffic.py --city $CITIES $HD"

step "Simulate one day of the ride company per city" 15 data \
  "$(each outputs/@/03_replay.html $CITIES)" \
  "$PY scripts/03_simulate.py --city $CITIES"

step "Run the 11 paper dispatch methods in every city" 60 sim \
  "$(each outputs/@/04_replay_p2_vfdcfmvd.html $CITIES)" \
  "$PY scripts/04_papers.py --city $CITIES"

step "Compare the methods across cities" 2 eval \
  "outputs/04_dashboard.html outputs/04_tradeoff.png" \
  "$PY scripts/04b_compare.py"

step "Two-week seasons of high-detail SF and Pune (2 parallel jobs)" 40 data \
  "outputs/sf_hd/09_season.png outputs/pune_hd/09_season.png" \
  "$PY scripts/09_hd_season.py --city sf_hd --days 14" \
  "$PY scripts/09_hd_season.py --city pune_hd --days 14 --start 2025-07-07"

step "City summary, map gallery and the start page" 3 eval \
  "outputs/cities_gallery.png outputs/index.html" \
  "$PY scripts/07_city_summary.py && $PY scripts/08_index.py"

run_phase "$@"
