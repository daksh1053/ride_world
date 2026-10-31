# Ride World

Ride World is a simulated ride-hailing company running on the real road networks of 15 cities.
It generates the data needed to train a learned world model of a ride-hailing marketplace.
It also tests the dispatch methods of two published fairness papers in every city.
The world model itself is specified in `01_all_inputs_formulation.md`.

## Start here

Open `outputs/index.html` in a browser.
It lets you watch every cab in all 15 cities as a moving dot with a short trail.
You can switch between the baseline dispatcher and the 11 methods from the two papers.

## The cities

| Group | Cities |
|---|---|
| Core | San Francisco, Pune |
| Metro siblings | Navi Mumbai, Thane, Arlington TX |
| Medium | Nagpur, Coimbatore, Chandigarh, Pittsburgh, Portland |
| Small | Panaji, Udaipur, Puducherry, Boulder, Ann Arbor |

## Folders

### `world/`

This folder holds the library code that every script uses.

| File | What it does |
|---|---|
| `cities.py` | It defines every city with its map extent, traffic targets, fares, fleet and demand. |
| `roadnet.py` | It downloads the road network from OpenStreetMap and cleans it into a connected graph. |
| `zones.py` | It splits each city into hexagonal zones of roughly neighbourhood size. |
| `traffic.py` | It computes travel times on every road for each hour and calibrates them to real congestion levels. |
| `routing.py` | It finds the fastest route for each cab on the live traffic of the simulated day. |
| `market.py` | It creates the drivers, the customers and their hidden behaviour, and it generates the ride requests. |
| `policies.py` | It defines the dispatcher interface and the baseline nearest-cab dispatcher. |
| `paper_policies.py` | It runs the methods of the two fairness papers unchanged inside this world through small adapters. |
| `sim.py` | It is the event-driven simulation of one day of the ride company and it logs every event, payment and rating. |
| `setup.py` | It loads a city and builds a ready-to-run simulated day. |
| `metrics.py` | It computes service, earnings and fairness measures from the logged tables. |
| `wm_data.py` | It turns a simulated day into the zone-level tensors used to train the world models. |
| `wm_models.py` | It defines, trains and rolls out world model 1 and its baselines. |
| `hd/` | The high-detail simulator for `sf_hd` and `pune_hd`: people carried across days, land use, surge, breaks and shifts, and the 64 x 64 raster of world model 2. |
| `wm2/` | World model 2: data loading, the grid models and the twin-run pairs. |
| `wm3/` | World model 3: the entity-level world rebuilt from logs (`state.py`), its features (`features.py`), the model (`model.py`), the imagined world (`rollout.py`) and the map viewers. |
| `explorer.py` | It builds the interactive map and traffic explorers. |
| `replay.py` | It builds the animated cab replay pages. |
| `viz.py` | It holds the shared plotting style. |

### `scripts/`

This folder holds one script per stage. Run them in order from inside the `ride_world` folder.

| Script | What it does |
|---|---|
| `01_import_maps.py` | It downloads each city's roads and points of interest, builds the zones and draws the maps. |
| `02_traffic.py` | It calibrates hourly traffic for weekdays and weekends and draws the congestion figures. |
| `02b_size_fleet.py` | It chooses the smallest fleet that serves most requests in each city. |
| `03_simulate.py` | It simulates one full day with the baseline dispatcher and builds the cab replay. |
| `04_papers.py` | It runs every paper method on the same test day in each city and builds a replay per method. |
| `04b_compare.py` | It compares all dispatch methods across cities and builds the comparison dashboard. |
| `05_wm_generate.py` | It simulates many days under several dispatchers to create the world model training data. |
| `06_wm_experiments.py` | It trains the world models and runs the six experiments on accuracy, rollouts, actions, transfer, plausibility and fairness. |
| `06b_wm_viewer.py` | It builds the page that compares the world model's imagined future with what really happened. |
| `06c_wm_figures.py` | It draws the world model result figures and writes the results summary. |
| `09_hd_season.py` to `15_pairs.py` | High-detail cities and world model 2 (see [World models](#world-models)). |
| `16_wm3_world.py` to `38_wm3_reports.py` | World model 3 (see [World models](#world-models)). |
| `07_city_summary.py` | It writes one summary row per city and draws the gallery of all road maps. |
| `08_index.py` | It builds the start page that shows the cabs in every city. |
| `run_city_pipeline.sh` | It runs the traffic, fleet, simulation and data generation stages for a list of cities. |

### `outputs/`

This folder holds every figure and interactive page.

| Path | What it contains |
|---|---|
| `index.html` | The start page for watching the cabs in all cities. |
| `<city>/01_explorer.html` | An interactive map of the city's roads and zones with a route checker. |
| `<city>/02_explorer.html` | An interactive traffic map with a time-of-day slider. |
| `<city>/03_replay.html` | The cab replay for the evening rush with the baseline dispatcher. |
| `<city>/04_replay_<method>.html` | The cab replay for each paper method. |
| `<city>/*.png` | Static figures for maps, traffic and the simulated day. |
| `04_dashboard.html` | An interactive comparison of all dispatch methods in all cities. |
| `04_allcities.png` | Every method against the baseline in every city. |
| `cities_summary.md` | One table with the key numbers of every city. |
| `cities_gallery.png` | The road maps of all 15 cities side by side. |
| `wm/06_wm_viewer.html` | An interactive view of the world model's imagined futures. |
| `wm/results.md` | The tables of the world model experiments. |
| `wm/fig_*.png` | The figures of the world model experiments. |

### `data/`

This folder holds the small result files that are kept in the repository.
Each city folder has `meta.json` with network facts and `fleet.json` with the chosen fleet size.
The `traffic/` subfolder of each city has the calibration table and its settings.
The `wm_models/`, `wm2_models/` and `wm3_models/` folders hold the trained world models 1, 2 and 3.
The large files such as road graphs, traffic arrays and simulated days are not in the repository.
Running the scripts recreates them.

### `reproduce/`

This folder holds the reproductions of the two fairness papers.
`paper1_ltf/` reproduces Long-term Fairness in Ride-Hailing Platform by Kang and others.
`paper2_vfdcfmvd/` reproduces A Fair Order Matching and Idle Vehicle Dispatching Algorithm for Ride-Hailing by Shi and others.
Each one has its method code, its scripts, its results, its figures and a `NOTES.md` that compares the results with the published numbers.
Stage 4 of Ride World imports these methods without changing them.

### Other files

| File | What it contains |
|---|---|
| `01_all_inputs_formulation.md` | The formal specification of the world model. |
| `NOTES.md` | Detailed notes on every stage, the assumptions and the known limitations. |
| `requirements.txt` | The Python packages and their versions. |

## Running with the scripts

The `run/` folder has one script to set up and one script per phase of the project slides.
Each script prints a live progress line, saves a log per step and skips steps that are already done.
If a step fails or you stop it with Ctrl-C, run the same command again to continue.

```bash
run/0_init.sh                   # set up .venv, packages and GPU check; safe to rerun (~15 min first time, ~10 s after)
run/1_build_world.sh            # maps, traffic, ride company, paper methods, high-detail cities
run/2_first_models.sh           # world models 1 and 2
run/3_world_model3.sh           # world model 3: data, training, rollouts, order test
run/4_decisions.sh              # repositioning, matching and driver agents
run/5_is_it_a_world_model.sh    # 8 h rollouts, shocks, ranking, scorecard, HTML reports
```

| Script | Full run | `--eval-only` (uses existing data and models) |
|---|---|---|
| `0_init.sh` | ~15 min first time, ~10 s after | - |
| `1_build_world.sh` | ~7 h | ~5 min |
| `2_first_models.sh` | ~16.5 h | ~25 min |
| `3_world_model3.sh` | ~8 h | ~1 h 40 min |
| `4_decisions.sh` | ~44 h | ~6 min |
| `5_is_it_a_world_model.sh` | ~6.5 h | ~30 min |

Times are for a 16-core PC with a 4 GB RTX 3050; map downloads also depend on the internet connection.
Useful options for every phase script: `--list` (steps, times and status), `--force` (run done steps again), `--only N`, `--from N`, `--dry-run`.
Logs are in `logs/run/<phase>/`, and the time each finished step took is added to `logs/run/times.csv`.
The commands each script runs are listed below.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

The `reproduce` folder needs its own data before its scripts can run.
Its `scripts/prepare_data.py` files download the NYC taxi data they use.

## Running everything

The commands below rebuild every city, run the dispatch comparison and build the start page.
Fleet sizes are already in `data/<city>/fleet.json`, so `02b_size_fleet.py` is only needed for a new city.

```bash
.venv/bin/python scripts/01_import_maps.py --city sf pune navimumbai thane arlington nagpur coimbatore chandigarh pittsburgh portland panaji udaipur puducherry boulder annarbor sf_hd pune_hd
.venv/bin/python -c "from world import cities; from world.hd import landuse; [landuse.build(cities.get(k)) for k in ('sf_hd', 'pune_hd')]"
.venv/bin/python scripts/02_traffic.py --city sf pune navimumbai thane arlington nagpur coimbatore chandigarh pittsburgh portland panaji udaipur puducherry boulder annarbor sf_hd pune_hd
.venv/bin/python scripts/03_simulate.py --city sf pune navimumbai thane arlington nagpur coimbatore chandigarh pittsburgh portland panaji udaipur puducherry boulder annarbor
.venv/bin/python scripts/04_papers.py --city sf pune navimumbai thane arlington nagpur coimbatore chandigarh pittsburgh portland panaji udaipur puducherry boulder annarbor
.venv/bin/python scripts/04b_compare.py
.venv/bin/python scripts/09_hd_season.py --city sf_hd --days 14
.venv/bin/python scripts/09_hd_season.py --city pune_hd --days 14 --start 2025-07-07
.venv/bin/python scripts/07_city_summary.py
.venv/bin/python scripts/08_index.py
```

`sf_hd` and `pune_hd` are the high-detail versions of San Francisco and Pune (land use, people carried across days, surge, breaks and shifts); world models 2 and 3 use them.
For a new city, run `01_import_maps.py` and then `scripts/run_city_pipeline.sh <city>`.

## World models

Three generations of learned world models were built on this simulator.
Each one learns to predict how the ride company's world evolves and how it responds to platform actions.
The final model is world model 3; the earlier two stay as baselines.

| | World model 1 | World model 2 | World model 3 (final) |
|---|---|---|---|
| What it models | Counts per hexagonal zone (about 100 zones) | Counts per cell of a 64 x 64 grid | Every driver, open request, zone and road |
| Time step | 5 min | 5 min | 30 s (the platform's decision step) |
| Cities | All 15 (standard detail) | `sf_hd`, `pune_hd` (high detail) | `sf_hd`, `pune_hd` |
| Network | GRU and baselines, Poisson outputs | Spatio-temporal transformer, Poisson outputs | Transformer over entity tokens, one probability head per event |
| Code | `world/wm_data.py`, `world/wm_models.py` | `world/hd/`, `world/wm2/` | `world/wm3/` |
| Step log | `NOTES.md` (stage 5-6) | `outputs/wm2/STEPS.md` | `outputs/wm3/STEPS.md` |
| Checkpoints | `data/wm_models/` | `data/wm2_models/` | `data/wm3_models/` (final: `wm3_exo.pt`) |

All commands run from inside the `ride_world` folder with the virtual environment of [Setup](#setup).
Trained checkpoints are in the repository, so the evaluation steps can run without retraining.
The training data is not in the repository (tens of GB); the data steps below recreate it.
Training times are for a 4 GB RTX 3050.

### World model 1: zone counts

Run [Running everything](#running-everything) first: world model 1 uses all 15 standard cities.

Each simulated day becomes, every 5 minutes and for each zone, the counts of new requests, pickups, cancellations, idle drivers, busy drivers and pending riders.
The platform's offers and repositioning orders per zone are the action inputs.
Models: persistence, time-of-day average, Poisson GLM, MLP, GRU, GRU with multi-step training, a fleet-conserving GRU, and a fleet-conserving GRU with multi-step training (the main one, `gru_cons_os`).
Data: 3 training days under 5 dispatchers and 2 test days under 6 dispatchers per city; the 6th dispatcher (P2 ILP) is never seen in training.

```bash
.venv/bin/python scripts/05_wm_generate.py --city all --workers 12   # simulate 405 days, ~40 min
.venv/bin/python scripts/06_wm_experiments.py                         # train all models + experiments E1-E6, ~50 min
.venv/bin/python scripts/06c_wm_figures.py                            # figures and results table
.venv/bin/python scripts/06b_wm_viewer.py                             # interactive imagined-vs-real page
```

Results: `outputs/wm/results.md`, `outputs/wm/fig_*.png`, `outputs/wm/06_wm_viewer.html`.
The experiments cover one-step accuracy, 1-hour rollouts, response to actions and to an unseen dispatcher, transfer to unseen cities, impossible predictions and per-zone fairness.

### World model 2: 64 x 64 grid

The high-detail cities (`world/hd/`: people carried across days, land use, surge, breaks, shifts) are run for long seasons.
The platform's dispatcher settings are drawn afresh every day, and some seasons add random concentrated repositioning orders, so the data shows the effect of different actions.
Each day becomes a raster: 288 frames of 5 min, 22 channels per grid cell.
One transformer is trained on both cities, then fine-tuned on its own rollouts.

1. Simulate the seasons (SF 180 days, Pune 160 days; seasons with different seeds can run in parallel):

   ```bash
   .venv/bin/python scripts/10_hd_dataset.py --city sf_hd   --days 120 --start 2025-01-06 --seed 1
   .venv/bin/python scripts/10_hd_dataset.py --city sf_hd   --days 60  --start 2024-11-06 --seed 3 --blocks
   .venv/bin/python scripts/10_hd_dataset.py --city pune_hd --days 50  --start 2025-05-05 --seed 1
   .venv/bin/python scripts/10_hd_dataset.py --city pune_hd --days 50  --start 2025-06-24 --seed 2
   .venv/bin/python scripts/10_hd_dataset.py --city pune_hd --days 30  --start 2025-03-06 --seed 3 --blocks
   .venv/bin/python scripts/10_hd_dataset.py --city pune_hd --days 30  --start 2025-04-05 --seed 4 --blocks
   .venv/bin/python scripts/10b_wm2_viewer.py --city sf_hd            # optional: browse the rasters
   ```

2. Pack the data and train:

   ```bash
   .venv/bin/python scripts/11_wm2_train.py --build
   .venv/bin/python scripts/11_wm2_train.py --build-dirs
   .venv/bin/python scripts/11_wm2_train.py --model st --steps 20000 --tag st_small
   .venv/bin/python scripts/11_wm2_train.py --model cnn --steps 6000 --tag cnn
   .venv/bin/python scripts/11_wm2_train.py --finetune st_small --rollout 4 --steps 2500 --lr 1e-4 --eval-every 500 --tag st_ro
   ```

   The best grid model, `st4_skip_ros`, starts from `st4_blocks` (the same transformer with the dispatch-direction channels) and is then fine-tuned three times:

   ```bash
   .venv/bin/python scripts/11_wm2_train.py --model st --dirs --steps 20000 --tag st4_blocks --outdir step4
   .venv/bin/python scripts/11_wm2_train.py --finetune st4_blocks --dirs --act-focus 0.3 --steps 5000 --lr 2e-4 --tag st4_focus_ft --outdir step4
   .venv/bin/python scripts/11_wm2_train.py --finetune st4_focus_ft --skip --dirs --act-focus 0.3 --steps 5000 --lr 2e-4 --tag st4_skip_ft --outdir step4
   .venv/bin/python scripts/11_wm2_train.py --finetune st4_skip_ft --dirs --act-focus 0.5 --rollout 6 --roll-feed sample --steps 2500 --lr 1e-4 --eval-every 500 --tag st4_skip_ros --outdir step4d
   ```

3. Evaluate:

   ```bash
   .venv/bin/python scripts/11_wm2_train.py --eval st_small st_ro cnn             # test scores vs baselines
   .venv/bin/python scripts/12_wm2_analysis.py --tag st_ro                        # rollouts, probes, prediction pages
   .venv/bin/python scripts/13_cf_truth.py --city sf_hd --pairs 8 --t-order 28950 --out wm2_cf2   # simulator twins with and without an order
   .venv/bin/python scripts/13_cf_truth.py --city pune_hd --pairs 8 --t-order 28950 --out wm2_cf2
   .venv/bin/python scripts/13b_cf_eval.py --tags st4_skip_ros --cf wm2_cf2 --outdir step4d_cf2   # model's effect vs the true effect
   .venv/bin/python scripts/14_wm_plan.py --city sf_hd --seeds 12                 # simulate candidate orders
   .venv/bin/python scripts/14_wm_plan.py --city pune_hd --seeds 12
   .venv/bin/python scripts/14_wm_plan.py --score st_ro st4_skip_ros              # let the models choose
   ```

Results: `outputs/wm2/step3/` to `outputs/wm2/step5/`, and `outputs/<city>_hd/12_wm2_predictions_<tag>.html`.
The grid models predict 5-20 minutes well, but their response to repositioning orders is too small, because a count per cell cannot show which drivers will comply.

### World model 3: every driver, request and road (final)

The model follows `01_all_inputs_formulation.md`.
Every 30 s it reads the full state: each online driver (position on the road, status, current drive, today's history, long-run history, profile, the offer or order it just got), each offline driver, each open request, each zone, and the time, weather and congestion.
For every entity it predicts the probability of each event in the next 30 s: drivers' own decisions, answers to offers and orders, coming online, rider cancellations and boardings, and new requests with their exact pickup and drop-off road junctions.
Driving along roads, routes, fares, surge and the request lifecycle are fixed rules.
To imagine the future, `world/wm3/rollout.py` samples the events, applies the rules and repeats, with the platform's own dispatcher acting on the imagined state.
The model has 3.9 M parameters; 2.1 M of them are per-junction tables for request locations.

It reuses world model 2's simulated seasons (`data/<city>/hd_runs/`), so run step 1 of world model 2 first.

1. Rebuild the entity-level world and check it against the simulator:

   ```bash
   .venv/bin/python scripts/16_wm3_world.py                   # outputs/wm3/stage1/
   ```

2. Build the training data (every 3rd minute of every day, `data/<city>/wm3v2/`, about 32 GB):

   ```bash
   .venv/bin/python scripts/17_wm3_dataset.py
   ```

3. Train. The final model is the base training followed by a fine-tune that gives demand its own encoder, which sees no driver supply:

   ```bash
   .venv/bin/python scripts/18_wm3_train.py --steps 30000 --eval-every 1500 --tag wm3_v2              # ~80 min
   .venv/bin/python scripts/18_wm3_train.py --init wm3_v2 --exo --steps 10000 --lr 2e-4 --tag wm3_exo   # ~30 min
   ```

4. Evaluate:

   ```bash
   .venv/bin/python scripts/19_wm3_eval.py --tag wm3_exo         # test scores per event type, calibration, hidden traits
   .venv/bin/python scripts/21_wm3_pred_page.py --tag wm3_exo    # interactive prediction maps
   .venv/bin/python scripts/20_wm3_rollout.py --tag wm3_exo      # imagined hours vs the simulator, fairness
   .venv/bin/python scripts/24_wm3_hourly.py --tags wm3_exo      # rare events per hour of the day
   ```

5. Test the response to platform orders. The simulator runs fresh seasons twice from the same state, with and without an extra repositioning order; the model imagines both versions from the state before the order, and the two effects are compared:

   ```bash
   .venv/bin/python scripts/22_wm3_twins.py --city sf_hd   --seed0 500 --n-seeds 13
   .venv/bin/python scripts/22_wm3_twins.py --city pune_hd --seed0 500 --n-seeds 12
   .venv/bin/python scripts/23_wm3_counterfactual.py --tag wm3_exo
   ```

6. Let the world model choose (RL). The agent head on the world model orders idle drivers every 5 minutes; it is trained inside the frozen world model, and its choices are then carried out in the simulator. Each step is about 4 hours on one 4 GB GPU, except the imitation steps (about 15 min):

   ```bash
   .venv/bin/python scripts/25_wm3_rl_starts.py                                   # start states from real days
   .venv/bin/python scripts/26_wm3_grpo.py --init wm3_exo --tag rl_pre            # RL only, pretrained world model
   .venv/bin/python scripts/26_wm3_grpo.py --init none    --tag rl_scratch        # RL only, random weights
   .venv/bin/python scripts/30_wm3_rl_bc.py --collect                             # the platform rule's choices, imagined
   .venv/bin/python scripts/30_wm3_rl_bc.py --init wm3_exo --tag bc_pre           # imitation (fine-tuning on the rule)
   .venv/bin/python scripts/30_wm3_rl_bc.py --init none    --tag bc_scratch
   .venv/bin/python scripts/26_wm3_grpo.py --init-agent bc_pre --tag rl_bc_pre --iters 40 --beta 0.1 --lr-scalar 1e-3
   .venv/bin/python scripts/26_wm3_grpo.py --init none --init-agent bc_scratch --tag rl_bc_scratch --iters 40 --beta 0.1 --lr-scalar 1e-3
   .venv/bin/python scripts/27_wm3_rl_execute.py --city sf_hd   --seed0 700 --n-seeds 20 --agents rl_pre rl_scratch bc_pre bc_scratch rl_bc_pre rl_bc_scratch
   .venv/bin/python scripts/27_wm3_rl_execute.py --city pune_hd --seed0 700 --n-seeds 12 --agents rl_pre rl_scratch bc_pre bc_scratch rl_bc_pre rl_bc_scratch
   .venv/bin/python scripts/28_wm3_rl_report.py --tags rl_pre rl_scratch rl_bc_pre rl_bc_scratch
   .venv/bin/python scripts/29_wm3_rl_probe.py sf_hd 4 15                         # optional: noise of one driver's effect
   ```

7. Let the world model also match riders to drivers (stage 6). Every 30 s each pending request, oldest first, picks one idle driver or waits; the agent first imitates the platform's matching, then GRPO (about 1 h per imitation, 2.5 h per GRPO run):

   ```bash
   .venv/bin/python scripts/31_wm3_match_bc.py --collect                                   # the rule's offers, imagined
   .venv/bin/python scripts/31_wm3_match_bc.py --init-agent rl_bc_pre     --tag mbc_pre
   .venv/bin/python scripts/31_wm3_match_bc.py --init-agent rl_bc_scratch --tag mbc_scratch
   .venv/bin/python scripts/32_wm3_match_grpo.py --init-agent mbc_pre --tag mrl_pre
   ```

8. Drivers as small agents maximising their own earnings (stage 7). The platform (mbc_pre) shows each idle driver a menu of up to 4 rides with the fare it will get; one small policy shared by all drivers picks one or declines, trained with GRPO on each driver's own earnings (about 3 h):

   ```bash
   .venv/bin/python scripts/33_wm3_driver_grpo.py --platform mbc_pre --tag drv_pre
   ```

   Then carry stages 6 and 7 out in the simulator and draw the figures:

   ```bash
   .venv/bin/python scripts/27_wm3_rl_execute.py --city sf_hd   --seed0 700 --n-seeds 20 --no-imagine --agents mbc_pre+m mbc_scratch+m mbc_pre+m@start mbc_pre+m@drv_pre
   .venv/bin/python scripts/27_wm3_rl_execute.py --city pune_hd --seed0 700 --n-seeds 12 --no-imagine --agents mbc_pre+m mbc_scratch+m mbc_pre+m@start mbc_pre+m@drv_pre
   .venv/bin/python scripts/34_wm3_match_report.py --platform mbc_pre --drivers drv_pre
   ```

9. Is it a world model? (stage 8). Eight imagined hours against the real day, shocks the model never saw (30% / 60% of idle drivers leave, prices x1.5 for 2 h), and a scorecard of ten criteria (about 4 h SF, 6 h Pune):

   ```bash
   .venv/bin/python scripts/35_wm3_long_rollout.py --city sf_hd   --seeds 13
   .venv/bin/python scripts/35_wm3_long_rollout.py --city pune_hd --seeds 8
   .venv/bin/python scripts/35_wm3_long_rollout.py --report
   .venv/bin/python scripts/36_wm3_shock.py --city sf_hd   --seed0 700 --n-seeds 20
   .venv/bin/python scripts/36_wm3_shock.py --city pune_hd --seed0 700 --n-seeds 8
   .venv/bin/python scripts/36_wm3_shock.py --report
   .venv/bin/python scripts/27_wm3_rl_execute.py --city sf_hd --seed0 700 --n-seeds 20 --agents mbc_pre+m mbc_scratch+m mbc_pre+m@start mbc_pre+m@drv_pre   # imagine the stage 6-7 policies
   .venv/bin/python scripts/37_wm3_scorecard.py
   ```

10. Readable HTML reports for stages 6-8: a one-hour map replay with every action drawn (offers, move orders, drivers' menus, shocks), side panels, and every figure explained (recording about 20 min, then 1 min to build):

   ```bash
   .venv/bin/python scripts/38_wm3_reports.py --collect
   .venv/bin/python scripts/38_wm3_reports.py --build
   ```

Outputs, all in `outputs/wm3/`:

| Stage | Main files |
|---|---|
| 1. World state | `16_verify.png`, `16_snapshot_<city>.png`, `16_driver_day.png`, `16_hidden.png`, `<city>_world.html` (interactive map: every driver, rider and road) |
| 2. Model | `19_heads.png`, `19_calibration.png`, `19_hidden.png`, `<city>_predictions.html` (each entity's predicted probabilities on the map) |
| 3. Rollouts | `20_kpis_<city>.png`, `20_errors.png`, `20_drivers.png`, `20_fairness.png`, `24_hourly.png`, `<city>_imagined_vs_real.html` (imagined and real side by side) |
| 4. Orders | `23_effects.png`, `23_scatter.png`, `23_followers.png` |
| 5. Choices (RL) | `stage5/28_execute.png` (every agent carried out in the simulator), `stage5/28_training.png`, `stage5/28_versions.png`, `stage5/28_scatter.png`, `stage5/28_paired.txt` |
| 6. Matching | **`stage6/stage6_report.html`** (report with map replay), `stage6/34_execute.png` (the agent's matching carried out in the simulator), `stage6/34_imitation.png`, `stage6/34_training.png`, `stage6/34_execute_paired.txt` |
| 7. Driver agents | **`stage7/stage7_report.html`** (report with map replay and drivers' menus), `stage7/34_execute.png` (menus and earning-maximising drivers in the simulator), `stage7/34_drivers_training.png`, `stage7/34_execute_paired.txt` |
| 8. World-model test | **`stage8/stage8_report.html`** (report: scorecard, shock replays real vs imagined), `stage8/37_scorecard.png` (ten criteria), `stage8/35_long_errors.png`, `stage8/35_long_fan.png`, `stage8/35_long_rank.png`, `stage8/36_shock_effects.png`, `stage8/36_shock_scatter.png`, `stage8/37_ranking.png` |

`outputs/wm3/STEPS.md` has every number, the problems found and the fixes.
