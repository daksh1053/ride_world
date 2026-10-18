# World model 3: the world at the level of each driver, rider and road

World model 2 (outputs/wm2/) turned each simulated day into counts per cell of a 64 x 64 grid
and predicted those counts. That is not the world of formulations/01_all_inputs_formulation.md,
where the state is every driver (position on the road graph, status, assignment, history),
every request and every road, and the model predicts their events. World model 3 does that.
The grid models remain as a baseline.

Plan, one stage at a time, each with a script, PNGs and an interactive page:
  1. the entity-level world state and the events to predict, rebuilt from the simulator logs
  2. the model: per-driver and per-request event heads on a graph of drivers, riders and roads
  3. rollouts: sample events, apply the fixed rules (lifecycle, one trip per driver, ledger)
  4. actions: offers and repositioning orders, driver by driver; fairness from trajectories

## Stage 1: the entity-level world (2026-10-18)
    Code: world/wm3/state.py (CityGraph, DayLog), scripts/16_wm3_world.py, world/wm3/viewer.html
    Data: the simulator already logs everything per entity for all 180 SF and 160 Pune days
    (data/<city>/hd_runs/wm2_seed*/<date>/: status changes, driven legs as road-node lists with
    arrival times, offers and answers, requests, all events with t_event and t_available,
    driver profiles, hidden driver traits). Nothing new is simulated and nothing is copied:
    a day is rebuilt in ~2 s (SF) / ~5 s (Pune).
    State every 30 s (the policy epoch), at boundary tau_t:
      drivers   status (8 kinds), exact position (edge u->v + share driven, or a node), current
                leg (pickup / trip / trip to a gateway / reposition / self-reposition / go home),
                its destination and metres left, assigned request, time in status, time online,
                today's trips, income, offers (rejected / unanswered), repositioning orders
                (declined), breaks, profile (mode, cost per km, tenure, rating, home, usual shift)
      requests  every open request: status (pending / offered / accepted / driver waiting),
                origin and destination node, mode, time waited, offers so far, quote
      roads     travel time / free-flow time of every edge (the router's 5-minute refresh,
                recomputed from the day's seeded traffic draw), 27,543 edges SF, 119,369 Pune
      actions   offers (driver -> request) and repositioning orders (driver -> zone) at tau_t
      events    everything in (tau_t, tau_t + 30 s]
    Counters use only events received by tau_t (t_available, eq. 5). Hidden traits are never
    in the state.
    Check against the simulator's own per-zone counts every 5 min of a whole day:
                 idle drivers   busy drivers   pending requests     (share in the right zone)
      SF         99.99%         99.89%         100%
      Pune       99.99%         99.85%         100%
      The few differences are drivers whose status changed in the same instant as the count.
      Two fixes were needed: the simulator counts a moving driver at the node where its leg
      started (kept as DayLog.stop), and Pune requests can expire unserved (request_expired).
    What the model must predict per 30-s step (one training day each):
      SF   69 events per step on average; offers accepted 81%, rejected 15%, unanswered 4%;
           repositioning orders followed 73%, declined 27%; moving drivers cover ~160 m per step
      Pune 113 events per step; offers 74 / 22 / 4%; orders followed 74%; ~190 m per step
    Hidden traits show in the history the state carries: share of offers accepted vs hidden
    accept bias r = .51 SF / .44 Pune; share of orders followed vs hidden compliance r = .53 / .46
    (8 days of one season). So per-driver history is the route to the compliance the grid
    model could not see.
    Viewer self-check: the page rebuilds 48 random driver records itself and all 48 equal
    state.py's.
    Outputs (outputs/wm3/stage1/):
      16_verify.png, 16_snapshot_sf_hd.png, 16_snapshot_pune_hd.png, 16_driver_day.png,
      16_targets.png, 16_hidden.png, 16_stage1.json,
      sf_hd_world.html, pune_hd_world.html (07:00-10:00; pan, zoom, play; click a driver or
      rider for its record, the action it received and its next-30-s events)

## Stage 2: the model (2026-10-18)
    Code: world/wm3/features.py (snapshot -> tokens and targets), world/wm3/data.py,
    world/wm3/model.py (WorldModel3), scripts/17_wm3_dataset.py, scripts/18_wm3_train.py,
    scripts/19_wm3_eval.py, scripts/21_wm3_pred_page.py
    One token per online driver, open request and zone, plus a global token (time, weekday,
    rain, congestion); a 4-layer transformer whose attention is biased by road distance between
    tokens (heads from city-wide to ~300 m). Heads, each a likelihood of the next 30 s:
      online driver   own decision (none / offline / break / head home / self-reposition /
                      break end / re-enter / abandon reposition); answer to an offer (accept /
                      reject / no answer); answer to an order (follow / decline); zone of a
                      self-reposition
      offline driver  comes online
      request         rider cancels; rider boards (driver waiting at the pickup)
      zone            number of new requests per mode; each new request's origin road node,
                      destination zone or gateway, destination node, and whether the rider
                      declines the quote
    Driving, routes, fares and the request lifecycle are fixed rules (stage 3), not learned.
    Data: every 3rd minute of 180 SF + 160 Pune days (data/<city>/wm3v2/<date>.npz);
    train / val / test by date as world model 2. 3.5 M parameters (2.1 M in node tables).

    Leaks found and fixed (the first model, wm3_base, had them; kept in stage2/wm3_base/):
      quote      the simulator applies surge to the stored quote only when the rider accepts,
                 so "quote without surge while surge > 1" meant "declined". Now every request
                 carries the pre-surge quote (features.q_base).
      truncated  when a rider cancels during the pickup, or a shift or the day ends mid-leg,
                 the simulator stores the leg only up to the last node reached, so in the
                 rebuilt world the driver stood still before the event (83% of such requests
                 then cancelled vs 0.4%). state.DayLog._untruncate re-routes those legs with
                 the simulator's own router (it reproduces 190/190 complete pickup legs exactly)
                 and ends them at the cut. Stage 1's busy-driver zone match rose from
                 99.89 / 99.85% to 99.99 / 99.99% (SF / Pune).
    wm3_v2 (data without the leaks): 30,000 steps, 82 min on a 4 GB GPU, val total 15.48.
    Test NLL per instance, model vs historical-rate tables (lower is better), SF / Pune:
      driver decision      .0286 vs .0333   /  .0223 vs .0251
      offer answer         .533 vs .552     /  .616 vs .652
      order answer         .585 vs .591     /  .570 vs .582
      self-reposition zone .22 vs 3.29      /  .83 vs 3.81
      comes online         .0021 vs .0026   /  .0024 vs .0030
      rider cancels        .0332 vs .0360   /  .0365 vs .0474
      rider boards         .546 vs .568     /  .540 vs .565
      new requests         .105 vs .109     /  .231 vs .239
      origin node          3.69 vs 3.71     /  4.78 vs 4.91
      destination zone     4.22 vs 4.40     /  4.24 vs 4.33
      destination node     3.35 vs 3.40     /  4.69 vs 4.81
      quote declined       .212 vs .212     /  .211 vs .221
    Calibration (deciles) close to the diagonal for every binary head in both cities.
    Hidden traits: correlation of the model's per-driver acceptance / order-follow
    probability with the hidden trait, vs the driver's observed share so far:
      SF offer .31 vs .23, order .41 vs .37; Pune offer .19 vs .20, order .35 vs .30.
    (Numbers above are wm3_v2; the final model wm3_exo is at the end of this file.)
    Outputs (outputs/wm3/stage2/): 19_heads.png, 19_calibration.png, 19_hidden.png,
      19_eval.json, train_wm3_v2.csv, sf_hd_predictions.html, pune_hd_predictions.html;
      stage2/wm3_base/ the same for the leaky first model.

## Stage 3: rollouts (2026-10-18)
    Code: world/wm3/rollout.py (CityRules, ImaginedWorld), scripts/20_wm3_rollout.py,
    scripts/24_wm3_hourly.py, world/wm3/duo.html
    From the true state at 07:30, 12:00 and 17:30 of 6 test days per city, the model runs the
    next hour on its own samples (3 draws each); the platform is the simulator's own dispatch
    policy, run on the imagined state. Rules: routes on the live road times, pickup, trip and
    reposition legs, fares and surge, one trip per driver, the request lifecycle.
    Rollout mismatches found and fixed before wm3_v2's run:
      offer ETA  the simulator answers offers with a GPS-style ETA, the rollout used the zone
                 table (7.8 vs 3.3 min): Pune offers were rejected 1.7x too often
      modes      offers between different vehicle modes are dropped by the simulator
      online_s   time online today was frozen at the start; now counted every step
    Error after 60 min (mean abs. error / true value), wm3_base -> wm3_v2, persistence:
                           SF                      Pune
      online drivers       .048 -> .043  (.093)    .022 -> .029  (.068)
      idle                 .19  -> .24   (.49)     .070 -> .098  (.29)
      with rider           .114 -> .109  (.116)    .047 -> .041  (.139)
      repositioning        .23  -> .17   (.32)     .17  -> .17   (.24)
      on break             .40  -> .31   (.51)     .41  -> .32   (.31)
      riders waiting       .37  -> .34   (.44)     .41  -> .35   (.37)
      pickups (cum.)       .067 -> .089  (.095)    .069 -> .078  (.075)
      completed trips      .066 -> .082  (.113)    .064 -> .067  (.122)
      cancellations        .67  -> .19   (.35)     .66  -> .10   (.24)
      offers rejected      .30  -> .22   (.17)     1.04 -> .14   (.16)
      drivers going online 1.90 -> 2.06  (1.78)    .36  -> .35   (1.98)
    Drivers: AUC of "online at +60 min" .69 / .67; per-driver trips r .64 / .66, income r
    .74 / .86. Fairness: income Gini of the hour, mean abs. error .025 / .017 (true .51 / .77);
    share of online drivers with no income .031 / .025.
    Remaining gap, SF drivers coming online at noon and 17:30 (model ~20, simulator ~2): log-ons
    come in waves (usual shift start +- 45 min, once a day); the per-driver hazard is right on
    the waves but too high 2-3 h before a wave (24_hourly.png). Tried: Gaussian time bumps into
    the driver heads, fine-tuned 10k steps (wm3_tf) -> sharper tails but peaks too low and
    worse hourly sums; rejected, wm3_v2 stays.
    Outputs (outputs/wm3/stage3/): 20_kpis_sf_hd.png, 20_kpis_pune_hd.png, 20_errors.png,
      20_drivers.png, 20_fairness.png, 20_rollout.json, 20_drivers.csv, 24_hourly.png,
      24_hourly.json, sf_hd_imagined_vs_real.html, pune_hd_imagined_vs_real.html;
      stage3/wm3_base/ the first model's run.

## Stage 4: actions — does the model respond to a platform order as the world does? (2026-10-18)
    Code: scripts/22_wm3_twins.py (simulator twins), scripts/23_wm3_counterfactual.py,
    world/wm3/rollout.py (common random numbers), world/wm3/model.py (exogenous demand path)
    Twins: 13 SF and 12 Pune fresh seasons (new drivers and riders, never in training), 3
    warm-up days so the platform can read each driver's history, then one test-split day run 3
    times from the same copied state: the usual dispatch, and the same plus one extra order of
    30 / 60 (SF) or 100 / 200 (Pune) idle drivers near a busy zone to a zone 10-25 min away,
    at a random time 11:00-13:00. The arms are identical until the order (checked: first
    differing step = the order's step). Data: data/<city>/wm3_twins/.
    Test: from the base run's state just before the order the model imagines the next 90 min
    with and without the same order (4 draws, same seed per pair); effect = difference.
    Compared with the simulator's effect (twin minus base) step by step.
    Three problems found and fixed on the way:
      shared random stream  one random stream for the whole world: as soon as the order changed
                 how many things were sampled, the two imagined arms drew unrelated luck and
                 city-wide counts differed by +-45 pickups from noise alone. Now every driver,
                 rider and zone gets its own uniform number per step from (seed, step, kind,
                 entity) (rollout.keyed_uniform; new requests keyed by step, zone, mode, index):
                 two identical arms are now identical, noise per draw 45 -> 3 pickups.
      demand saw supply  the request-rate and origin/destination heads read zone tokens that
                 include idle and busy drivers. Watching the world, "few idle drivers here" does
                 precede more requests (drivers were used up by demand), so moving drivers away
                 made the model invent demand: +10-17% requests in the source zones, while the
                 simulator's requests are identical in both arms (838 = 838). Fix: a separate
                 zone encoder from exogenous inputs only (place, POIs, time, weather, congestion,
                 recent requests; not idle / busy / pending / repositioning / surge) feeds the
                 demand heads; the quote-decline head keeps the full state (riders do react to
                 ETA and price). Fine-tuned 10k steps from wm3_v2 -> wm3_exo, the final model.
                 Request-count NLL .102 -> .104 SF, .231 -> .231 Pune.
      (wm3_tf, time bumps into the driver heads, was rejected; see stage 3.)
    Results, wm3_exo, 26 SF and 24 Pune pairs; r between model and simulator effects, mean abs.
    error of the model vs of "no effect":
                                              SF                    Pune
      idle drivers in target zone, +30 min    r .93, 4.4 vs 18.5    r .97, 5.2 vs 85
      idle drivers in source zones, +30 min   r .57, 5.6 vs 6.7     r .97, 6.1 vs 83
      drivers still repositioning, +30 min    r .40 (all ~0)        r .87, 3.7 vs 7.8
      ordered drivers who follow              r .96, 2.9 vs 33      r .99, 4.5 vs 111
      ordered drivers' trips in 90 min        r .79, 10 vs 37       r .81, 9 vs 35
      ordered drivers' net income in 90 min   r .73, 85 vs 282      r .84, 872 vs 3529
      all pickups, +60 min                    r .49, 8.3 vs 8.3     r .05, 17 vs 17
      cancellations, +60 min                  r .32, 4.8 vs 4.9     r -.07, 10 vs 10
    Pooled over both cities: target idle r .99, source idle r .99, movers' income r .94.
    City-wide pickups and cancellations hardly change in the simulator either (within its own
    twin noise); the model now also predicts ~0 there (before the demand fix: +9 vs +5 SF).
    Before the fixes (wm3_v2 with one random stream): pickups +60 min MAE 30 vs 8 for "no
    effect", draw s.d. 45 (stage4/wm3_v2/).
    Followers per pair: model 3.7 drivers mean abs. error, population rate 3.4: with 30-200
    orders the count is mostly binomial noise. Per driver (who follows), AUC: model .56 SF /
    .50 Pune, the driver's observed follow rate .58 / .54, the hidden trait .72 / .71 — with
    only 3 days of history the model knows about as much as the observed rate (on the
    long-history test days of stage 2 the order head beats the observed rate, r .41 vs .37).
    Outputs (outputs/wm3/stage4/):
      23_effects.png    effect over 90 min for the largest order per seed, model draws + mean
                        vs simulator twins
      23_scatter.png    every pair, model vs true effect, 8 quantities
      23_followers.png  followers per pair: model vs population-average compliance
      23_counterfactual.json, 23_pairs.csv, 23_drivers.csv
      wm3_v2/           the same before the two fixes

## Final model: wm3_exo (data/wm3_models/wm3_exo.pt)
    Stage 2 and 3 outputs in outputs/wm3/stage2/ and stage3/ are wm3_exo's; wm3_v2's are in
    the wm3_v2/ subfolders. Test NLL vs historical rates: better on every head except the
    origin node within a zone in SF (3.73 vs 3.71). Rollout error after 60 min (SF / Pune):
    online drivers .023 / .026, idle .14 / .07, pickups .065 / .083, completed trips .060 / .051,
    cancellations .33 / .13, new requests .074 / .106 (persistence .09/.07, .49/.29, .10/.08,
    .11/.12, .35/.24, .10/.08). Fairness: income Gini mean abs. error .023 / .013,
    share of drivers with no income .034 / .020; per-driver income r .74 / .86.
    Hourly sums of rare events (24_hourly.png): log-ons per day 1315 vs 1274 SF, 3075 vs 3060
    Pune; rider cancellations 608 vs 628, 2090 vs 2090.
