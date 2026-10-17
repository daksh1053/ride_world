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
