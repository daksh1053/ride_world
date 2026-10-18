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
