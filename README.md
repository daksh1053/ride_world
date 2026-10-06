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
| `wm_models.py` | It defines, trains and rolls out the learned world models and their baselines. |
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
The `wm_models/` folder holds the trained world models.
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

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

The `reproduce` folder needs its own data before its scripts can run.
Its `scripts/prepare_data.py` files download the NYC taxi data they use.

## Running everything

```bash
.venv/bin/python scripts/01_import_maps.py --city sf pune
.venv/bin/python scripts/02_traffic.py --city sf pune
.venv/bin/python scripts/03_simulate.py --city sf pune
.venv/bin/python scripts/04_papers.py --city sf pune
.venv/bin/python scripts/04b_compare.py
.venv/bin/python scripts/05_wm_generate.py --city all
.venv/bin/python scripts/06_wm_experiments.py
.venv/bin/python scripts/06c_wm_figures.py
.venv/bin/python scripts/06b_wm_viewer.py
.venv/bin/python scripts/07_city_summary.py
.venv/bin/python scripts/08_index.py
```

For a new city, run `01_import_maps.py` and then `scripts/run_city_pipeline.sh <city>`.
