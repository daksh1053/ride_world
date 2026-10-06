# ride_world — the simulated world behind the ride-service world model

The learned world model (`../formulations/01_all_inputs_formulation.md`,
`02_database_formulation.md`) needs trajectories with things public trip data does not
have: stable driver and customer identities, offers and rejections, cancellations,
ledgers, ratings and observation masks. This project builds a **ground-truth world**
on real road networks that generates exactly that data. The world model itself
comes later and will be trained on its output.

Cities (15, see `outputs/cities_summary.md` and `outputs/cities_gallery.png`): core **Pune** (Pune City Subdistrict, OSM relation 10351626, 312 km². OSM has no PMC polygon and "Pune" geocodes to the whole district) and **San Francisco** (city/county).

Every stage is a script that writes data to `data/<city>/` and something you can
look at to `outputs/<city>/`: PNGs plus a self-contained interactive HTML explorer
that opens straight from disk.

**Start here:** `outputs/index.html` — watch the cabs in all 15 cities under any of the 12 dispatchers (rebuild with `scripts/08_index.py`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Stages

| # | Script | Status | What you can check |
|---|---|---|---|
| 1 | `scripts/01_import_maps.py` | done | `01_roads.png`, `01_speeds.png`, `01_zones.png`, `01_explorer.html` |
| 2 | `scripts/02_traffic.py` | done | `02_tti_profile.png`, `02_congestion.png`, `02_volume.png`, `02_zone_tt.png`, `02_sample_week.png`, `02_explorer.html` |
| 3 | `scripts/03_simulate.py` | done | `03_replay.html`, `03_kpis.png`, `03_drivers.png`, `03_zones.png`, `03_snapshot.png` |
| 4 | `scripts/04_papers.py`, `04b_compare.py` | done | `04_dashboard.html`, `04_tradeoff.png`, `<city>/04_replay_<method>.html`, `04_results.md` |

### Stage 1: maps

```bash
.venv/bin/python scripts/01_import_maps.py --city sf pune
```

1. **Download** (cached in `data/<city>/raw/`, skipped if present; `--redownload` forces it):
   the admin boundary, the OSM `drive` network and points of interest (amenity, shop,
   office, tourism, stations, ...).
2. **Process**: keep the largest *strongly* connected component, so every node can
   reach every other. Free-flow speed comes from the OSM `maxspeed` tag, or from a
   per-city road-class default (`world/cities.py`) when the tag is missing.
   Lanes per direction come from `lanes`, or a class default.
3. **Zones** `Z`: H3 hexagons (Pune res 7 ≈ 5.2 km², SF res 8 ≈ 0.74 km²). A cell
   with fewer than 15 road nodes merges into its largest neighbour. Each zone has a
   *centre node* for zone-to-zone travel times and per-category POI counts.
4. **Save**: `graph.graphml`, `nodes.parquet`, `edges.parquet`, `zones.parquet`,
   `zone_geoms.geojson`, `meta.json`.
5. **Render**: the three PNGs and `01_explorer.html`.

**Explorer** (`outputs/<city>/01_explorer.html`): click two points to get the fastest
free-flow route on the processed graph, with length, time, average speed, zones
crossed and road classes used. Hover zones for their attributes, switch the choropleth
metric, and toggle road classes and the basemap.

### Stage 2: traffic

```bash
.venv/bin/python scripts/02_traffic.py --city sf pune            # ~8 min (SF), ~12 min (Pune) on 16 cores
.venv/bin/python scripts/02_traffic.py --city sf --render-only   # re-plot from saved arrays
```

**Typical conditions.** Each day type (weekday, weekend) and hour gets a background
private-traffic OD matrix: a gravity model between zones plus **external traffic**
through *gateways*. Gateways are the motorway, trunk and primary roads where they cross
the city boundary (SF 8, including both bridges; Pune 24). No road outside the city
is modelled: a gateway is a source or sink of flow. External trips make up 35% (SF) or
25% (Pune) of the total. They mostly enter towards activity zones in the morning and
leave in the evening, and 15% pass straight through. The OD is assigned with MSA under
BPR delays `t = t0 (1 + α (v/c)^β)`, where capacity is lanes × a per-class lane
capacity. Traffic is spread over the whole street network in two ways. Each of the 8
MSA iterations loads a zone's trips at 10 nodes redrawn from *all* its nodes (weighted
towards arterials). Each step also routes on link times with lognormal noise
(σ = 0.25, probit route choice), so parallel streets share the load. In SF, 90% of
edges carry traffic. The curve is classic (0.15, 4) on streets and the NCHRP-365 freeway curve
(0.83, 5.5) on motorways and trunks. The hour's total demand is solved for until
the flow-weighted network travel-time index `TTI = Σvt / Σvt0` matches that hour's target.
Targets (`world/cities.py`) are anchored to the **TomTom Traffic Index 2025**:

| | TomTom 2025 | model anchor |
|---|---|---|
| Pune | 33:20 per 10 km (congestion 71%), AM 38:13, PM 41:40 | TTI 1.96 at 09:30, 2.14 at 18:30 |
| SF | congestion 49.7%, 29:42 per 10 km, no peak split published | TTI 1.80 at 08:30, 1.95 at 17:30 (assumed) |

The hourly shape between anchors and the weekend profile are assumptions. The
calibrated demand level is an *effective* quantity: it depends on the assumed lane
capacities and on the loading scheme. Read it as "the load that reproduces observed
congestion", not as a count of cars.

Calibration results: every hour lands within ±0.011 of its target TTI. From the
busiest zone, the median PM-peak travel time to other zones is 13.5 min in SF
(9.0 free-flow) and 29.6 min in Pune (16.9 free-flow).

**Realised conditions.** `world.traffic.TrafficModel` gives edge times for any
second of a simulated day: BPR on the interpolated background volume, times a daily
lognormal demand factor (σ = 6%), a rain *spell* (seasonal probability, random start,
median 5 h, intensity 0.3–1, 30-min ramps; at full intensity −8% capacity, −6%
free-flow speed, +3% demand) and Poisson incidents on major roads (capacity × 0.3–0.6,
median 45 min; part of the link's traffic diverts, volume × factor^0.7).
`edge_times(..., extra_volume=...)` lets the ride fleet add its own load.

**Known limitations.** Calibration is city-wide, and nothing checks where the
congestion appears. In SF, downtown's dense grid comes out nearly free-flowing: flow
spreads over many parallel streets, and signal delay isn't modelled. There are no
public street-level speeds to calibrate against (Uber Movement is discontinued, and
PeMS covers freeways only). Heavy monsoon rain adds up to ~45% to Pune's peak TTI,
because at v/c ≈ 1.9 its roads are far more sensitive than SF's. A rain spell that
crosses midnight is cut off at 24:00.

**Explorer** (`02_explorer.html`): weekday/weekend toggle, time slider with play,
a map of either *congestion* (delay factor) or *traffic volume* (veh/h, every road),
and a TTI chart with the targets (click the chart to jump to an hour). In the
congestion view grey means free-flowing, not unused; the volume view shows usage. *Route* mode shows the fastest path at the chosen time and that
trip's travel time over the whole day. *Zone travel times* mode colours every zone by
travel time from the clicked zone.

### Stage 3: the ride company

```bash
.venv/bin/python scripts/03_simulate.py --city sf pune                     # ~30-40 s per city-day
.venv/bin/python scripts/03_simulate.py --city pune --date 2025-07-08 --seed 3 --replay 8 11
```

One simulated day is an event-driven world (`world/sim.py`) on the stage-1 graph. Every
cab leg is routed exactly (numba Dijkstra, about 1 ms) on the *realised* stage-2 traffic
of that date: demand level, rain, incidents.

- **People** (`world/market.py`). Drivers have an ID, vehicle type and capacity,
  operating cost κ_i, tenure, prior rating sum/count, home zone and usual shift.
  Customers have a home zone, tenure and ratings. Behavioural parameters are
  saved separately as `*_latent` tables. They are the hidden factors Z_t, never an
  input to a world model: offer-acceptance intercept, reposition compliance,
  service quality, shift length, days per week, patience, tipping, rating
  leniency, riding frequency.
- **Demand.** Hourly request volumes per city profile. Origin and destination come
  from a ride gravity model between zones (home→activity in the morning, reverse in
  the evening). Pickup and drop-off nodes are street-weighted within each zone. Riders
  are often a resident of one end of the trip, drawn by heavy-tailed activity, so
  repeat customers occur. A share of requests are **intercity**: the cab drives to
  the gateway, leaves the map for the sampled outside time with an outside-distance
  fare, then drives back empty and re-enters through the same gateway.
- **Platform actions** every 30 s come from a `Policy` (`world/policies.py`): offers
  (driver→request) and dispatches (driver→zone). The policy sees only platform
  knowledge: idle drivers, pending requests, typical zone travel times and recent
  demand. The simulator drops infeasible instructions (eq. 7-8). The baseline
  `nearest` does Hungarian min-ETA batch matching within the pickup radius, and
  every 5 min moves long-idle cabs from surplus zones to deficit zones. Supply
  counts cabs already en route; need is the next 10 minutes' expected demand.
- **World events**, each with `t_event`, `t_available` and `recorded`: request, driver
  online/offline, offer accepted/rejected/timeout (logit in ETA, fare, night,
  intercity), dispatch accepted/declined, leg start, arrival, pickup, drop-off, city
  exit/re-entry, customer cancellation (patience before match, long ETA or random after
  acceptance), expiry, settlement, tip and rating (delayed), rain and incidents.
  Telemetry reaches the platform after 1-6 s, and 1-2% of movement events are never
  recorded (the observation model).
- **Ledger** (eq. 15-18): customer charge, fare, commission, booking fee,
  cancellation fee after a 2-min grace, tips, operating cost κ_i × km for every leg
  (including empty pickups, repositioning and the intercity return), and online
  working time.

Per run, `data/<city>/sim/<date>_seed<k>_<policy>/` holds these tables:

| Table | Formulation relation |
|---|---|
| `drivers`, `drivers_latent` | Driver (profile) / latent Z |
| `customers`, `customers_latent` | Customer / latent Z |
| `requests`, `trips` | Request, Trip (full lifecycle times, outcome, fare, tip, distance) |
| `actions`, `offers` | Action (per epoch: candidates, offers, dispatches, policy version) |
| `events` | Event (typed, entity-linked, event/availability time, recorded flag) |
| `ledger` | Ledger |
| `ratings` | Rating |
| `context` | Context (per 5 min × zone: new requests, pending, idle/busy cabs, network TTI, rain) |
| `legs`, `status` | driven paths and driver status timeline (movement; replay) |

`metrics.json` has the trajectory observables Φ of sec. 7. Service: completion, cancel
and expiry rates, waits. Drivers: F_var, F_min, Gini, hourly v_i, weighted-time
F_log (reported as undefined when any v_i ≤ 0, as the formulation requires) and
U_dist (eq. 20). `zone_service.parquet` has s_z and w_z (eq. 25) with denominators.

Baseline day, `2025-03-04`, seed 0:

| | SF | Pune |
|---|---|---|
| requests / completed | 12,036 / 94.9% | 9,751 / 84.2% |
| mean (p90) pickup wait | 2.9 (4.9) min | 5.6 (10.0) min |
| offer acceptance | 78.9% | 72.1% |
| driver net per online hour, mean (p10) | $17.7 ($10.6) | ₹76 (₹23) |
| Gini of daily net earnings | 0.25 | 0.37 |

**Replay** (`03_replay.html`): every cab is a dot with a short fading trail, coloured
by status: grey idle, blue to pickup, orange with passenger, green repositioning.
Hollow rings are waiting requests, red × a cancellation. There are play/pause,
10-300× speed and a time slider, live counts, whole-day charts (click to jump),
click-a-cab details (vehicle, rating, earnings so far, trips) with *follow*, and the
day summary. Static views: `03_kpis.png`, `03_drivers.png`, `03_zones.png`,
`03_snapshot.png`.

**Known limitations.** Only outbound intercity trips; the cab returns empty, and there
are no airport pickups into the city. No surge pricing: the tariff Γ is fixed. Drivers
don't reposition on their own, only when dispatched. The ride fleet does not add to
traffic volume; the hook exists, but ~1,000 cabs are negligible next to the
background flow. Online time is counted within the calendar day, so a driver can show
two shifts: the end of last night's and this evening's.

### Stage 4: the papers' methods as the ride company's dispatcher

```bash
.venv/bin/python scripts/04_papers.py --city sf pune     # 24 jobs, ~8 min on 6 workers
.venv/bin/python scripts/04b_compare.py                  # 04_tradeoff.png, 04_dashboard.html, 04_results.md
```

`world/paper_policies.py` runs the method code in `../reproduce` **unchanged**,
behind facade objects that answer each paper's environment calls from the world's
observation:

- **Paper 1** (Kang et al.): Greedy, REASSIGN, LAF, Balance Ride-Pooling, MOMAQL.
  L = zones, Geo = typical zone road distance, utility = U_dist (eq. 20) from
  platform records, fleet = drivers online today, one round per 30 s epoch, no
  repositioning.
- **Paper 2** (Shi et al.): NM, WDF, LAF, ILP, SID, VFDCFMVD. Slot = 30 s epoch,
  p_o = upfront quote, c_d = κ_i, income = platform-recorded net income, `reposition`
  → dispatch instructions, t_o^w = 15 min.
- **Protocol.** Learning methods train on 3 warm-up days (other dates and seeds),
  then every method is scored on the same test day (Tue 4 Mar 2025, seed 0): same
  drivers, riders, requests and traffic.
- **What the world adds that neither paper models:** drivers reject offers,
  customers cancel, trips take routed time on congested roads, and income settles at
  drop-off.

Runs are saved under `data/<city>/sim/stage4/`, separate from the stage-3 run, which is the baseline dispatcher with its own replay `03_replay.html`. Every method also gets a replay, `outputs/<city>/04_replay_<method>.html`.

**How to look at the results:** `outputs/04_dashboard.html` is interactive. Pick the city and any two metrics as axes; hover a dot for details and click it to watch that method's cabs. It also has a sortable table with the best value per column highlighted. `outputs/04_tradeoff.png` is the static version: riders served vs earnings inequality, plus each paper's own trade-off, with better always up and to the right.

Results:

| | SF completion | SF Gini | SF P1 F̂ | SF P2 unfairness | Pune completion | Pune Gini | Pune P1 F̂ | Pune P2 unfairness |
|---|---|---|---|---|---|---|---|---|
| Baseline nearest | **94.9%** | 0.248 | 0.485 | 1,560 | **84.2%** | 0.366 | 0.636 | 3,642 |
| P1 MOMAQL | 91.8% | 0.199 | **0.320** | 1,569 | 73.5% | 0.222 | **0.311** | 3,569 |
| P1 Balance Ride-Pooling | 84.9% | 0.231 | **0.314** | 1,610 | 66.3% | 0.254 | 0.349 | 3,529 |
| P2 SID | 89.9% | **0.158** | 0.446 | 1,486 | 72.7% | 0.189 | 0.382 | 3,502 |
| P2 ILP | 81.4% | 0.172 | 0.577 | 1,497 | 66.4% | **0.183** | 0.386 | **3,325** |
| P2 LAF | 51.4% | 0.259 | 0.552 | **1,395** | 65.6% | 0.263 | 0.475 | 3,418 |
| P2 VFDCFMVD | 57.6% | 0.304 | 0.873 | 1,760 | 56.6% | 0.285 | 0.533 | 3,424 |

- **Paper 1's claim holds in both cities.** MOMAQL gives the lowest normalised
  unfairness of U_dist of the paper-1 methods (tied with Balance Ride-Pooling in SF),
  and the lowest Gini, as in the NYC reproduction. It costs 3 points of completion in
  SF and 11 in Pune, where its utility (trip minus pickup distance) accepts longer
  pickups that customers then cancel.
- **Paper 2's claim does not hold.** VFDCFMVD is not the fairest by the paper's own
  unfairness in either city: LAF, SID and ILP beat it. It serves the fewest riders.
  The dV > 0 pruning leaves 32% (SF) of requests never offered, and those riders
  cancel. The NYC reproduction showed the same mechanism (49.8% service).
- **Paper 2's SID is the best all-round fair method:** lowest Gini in SF, nearly
  baseline completion and earnings.
- **Every fairness method costs completion** against plain nearest dispatch with
  repositioning, and more in supply-short Pune than in SF.
- These are single test days; seed-to-seed variance has not been measured yet.

### Adding cities (13 more)

Cities are generated from templates in `world/cities.py` (India = Pune's shapes, US = SF's), scaled to
each city: TomTom 2025 congestion level where published (Mumbai, Dallas, Fort Worth, Pittsburgh,
Portland), otherwise a size-tier assumption; demand scaled from population; intercity trips through
the three biggest gateways; H3 zone size chosen automatically (~100 zones). Where OSM has no
city-sized admin polygon (Thane, Coimbatore, Udaipur, Panaji) the extent is a circle of the
municipal area. Fleets are sized by `scripts/02b_size_fleet.py`: the smallest fleet that completes
min(88%, best achievable - 1 pt) of baseline requests. `scripts/run_city_pipeline.sh` runs stages
2 -> 2b -> 3 -> 5 per city.

| Group | Cities |
|---|---|
| core | San Francisco, Pune |
| metro siblings | Navi Mumbai, Thane (Mumbai region), Arlington TX (Dallas-Fort Worth) |
| medium | Nagpur, Coimbatore, Chandigarh, Pittsburgh, Portland |
| small | Panaji, Udaipur, Puducherry, Boulder, Ann Arbor |

### Stage 5-6: learned world models

```bash
.venv/bin/python scripts/05_wm_generate.py --city all --workers 12    # 405 simulated days, ~40 min
.venv/bin/python scripts/06_wm_experiments.py                          # train + E1-E6, ~50 min (RTX 3050)
.venv/bin/python scripts/06c_wm_figures.py && .venv/bin/python scripts/06b_wm_viewer.py
```

Data (`world/wm_data.py`): per city 3 training days x 5 dispatch policies and 2 held-out days x 6
policies (P2 ILP never seen in training); same date and seed = same people and traffic, so policies
are true counterfactuals. Observations are per-zone counts every 5 min (new requests, pickups,
cancellations, idle, busy, pending) from *recorded* events bucketed by *t_available*; actions are the
boundary epoch's offers and the step's reposition instructions (later offers would leak arrivals).

Models (`world/wm_models.py`), all Poisson-likelihood, one shared model for every city: persistence,
time-of-day average, Poisson GLM, MLP, GRU, GRU + overshooting, fleet-conserving GRU, and
fleet-conserving GRU + overshooting (main). Papers that shaped them are in `../papers/`.

Outputs `outputs/wm/`: `results.md`, `fig_rollouts.png`, `fig_cities.png`, `fig_interactability.png`,
`fig_fairness.png`, and `06_wm_viewer.html` (pick a city, dispatcher and hour; see the model's
imagined hour against what happened, as city totals and a zone map).

## Layout

```
world/        cities.py (per-city config) · roadnet.py (G) · zones.py (Z) · traffic.py (W_t traffic)
              routing.py (live point-to-point routing) · market.py (people, demand, tariff)
              policies.py (action sources) · paper_policies.py (paper adapters) · setup.py
              sim.py (the world) · metrics.py (Φ, paper metrics)
              viz.py (PNG style) · explorer.py / replay.py (interactive HTML)
scripts/      one script per stage
data/<city>/  saved world data (raw OSM pulls under raw/)
outputs/<city>/  PNGs and HTML explorers
```
