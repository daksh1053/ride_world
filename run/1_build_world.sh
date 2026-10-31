#!/usr/bin/env bash
# Phase 1 of the slides: build the world.
# Real road maps of 15 cities, calibrated traffic, one simulated day per city, the two fairness
# papers' dispatch methods in every city, and the two high-detail cities (sf_hd, pune_hd).
source "$(dirname "$0")/lib.sh"
PHASE="1_build_world"
PHASE_TITLE="Phase 1 - Building the world"
PHASE_DESC="Maps, traffic, the ride company and the paper methods in 15 cities, plus the high-detail SF and Pune."
PHASE_RESULTS="04_dashboard.html, 04_tradeoff.png, 04_allcities.png (paper methods), cities_summary.md, cities_gallery.png"

CITIES="sf pune navimumbai thane arlington nagpur coimbatore chandigarh pittsburgh portland panaji udaipur puducherry boulder annarbor"
SIZED="navimumbai thane arlington nagpur coimbatore chandigarh pittsburgh portland panaji udaipur puducherry boulder annarbor"
HD="sf_hd pune_hd"
each() { local pat=$1 out="" c; shift; for c in "$@"; do out="$out ${pat//@/$c}"; done; echo "$out"; }

# ---- full-run ----
step "Download road maps and build zones (17 cities, needs internet)" 150 data \
  "$(each data/@/zone_geoms.geojson $CITIES $HD)" \
  "$PY scripts/01_import_maps.py --city $CITIES $HD"

step "Land use for the high-detail cities" 5 data \
  "data/sf_hd/hd/landuse.gpkg data/pune_hd/hd/landuse.gpkg" \
  "$PY -c \"from world import cities; from world.hd import landuse; [landuse.build(cities.get(k)) for k in ('sf_hd', 'pune_hd')]\""

step "Calibrate hourly traffic (17 cities)" 150 data \
  "$(each data/@/traffic/calibration.csv $CITIES $HD)" \
  "$PY scripts/02_traffic.py --city $CITIES $HD"

step "Choose each generated city's fleet size (13 cities)" 30 data \
  "$(each data/@/fleet.json $SIZED)" \
  "$PY scripts/02b_size_fleet.py --city $SIZED"

step "Simulate one day of the ride company per city" 15 sim \
  "$(each outputs/@/03_replay.html $CITIES)" \
  "$PY scripts/03_simulate.py --city $CITIES"

step "Run the 11 paper dispatch methods in every city" 60 sim \
  "outputs/04_results.csv $(each outputs/@/04_replay_p2_vfdcfmvd.html $CITIES)" \
  "$PY scripts/04_papers.py --city $CITIES"

step "Two-week seasons of high-detail SF and Pune (2 parallel jobs)" 40 sim \
  "outputs/sf_hd/09_season.png outputs/pune_hd/09_season.png" \
  "$PY scripts/09_hd_season.py --city sf_hd --days 14" \
  "$PY scripts/09_hd_season.py --city pune_hd --days 14 --start 2025-07-07"

step "Start page for watching the cabs" 1 page \
  "outputs/index.html" \
  "$PY scripts/08_index.py"

# ---- eval ----
step "Compare the dispatch methods across cities" 1 eval \
  "outputs/04_dashboard.html outputs/04_tradeoff.png outputs/04_allcities.png outputs/04_results.md" \
  "$PY scripts/04b_compare.py"

step "City summary table and map gallery" 1 eval \
  "outputs/cities_summary.md outputs/cities_gallery.png" \
  "$PY scripts/07_city_summary.py"

EVAL_INPUTS="outputs/04_results.csv
  $(each data/@/meta.json $CITIES) $(each data/@/edges.parquet $CITIES) $(each data/@/nodes.parquet $CITIES)
  $(each data/@/raw/boundary.gpkg $CITIES) $(each data/@/traffic/calibration.csv $CITIES) $(each data/@/traffic/meta.json $CITIES)
  $(each data/@/fleet.json $SIZED) $(each 'data/@/sim/*_nearest/metrics.json' $CITIES)"

run_phase "$@"
