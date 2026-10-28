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
    rain, congestion); a 4-layer transformer whose attention is biased by straight-line distance between
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

## Stage 5: the world model chooses — RL (GRPO) on repositioning, executed in the simulator (2026-10-27)
    Goal: turn the world model from a predictor into a decision maker: it orders idle drivers
    where to go, the orders are carried out in the real simulator, and two starting points are
    compared: RL on the pretrained world model (wm3_exo) vs the same network from random weights.

    The agent (world/wm3/agent.py): a head on WorldModel3's token states. Every 5 min each
    deciding driver (idle >= 5 min, not just offered a ride) gets a choice over
    [stay, zone_1..zone_Z]: move logits = query(driver state) . key(zone state) + travel-time
    term, only zones within 15 min; stay logit = MLP(driver state) + bias + logsumexp(moves).
    Matching stays the platform's rule (HDDispatch). Reward = riders picked up + drivers'
    proportional-fairness welfare (sum over drivers of log(1 + net earnings / median fare), net
    of km driven). AgentWorld imagines the city with the frozen wm3_exo and the agent's orders;
    step_many batches many imagined worlds on the GPU (identical to single-world stepping).
    Start states: 120 train + 24 val per city from real days (scripts/25_wm3_rl_starts.py).

    Execution (scripts/27_wm3_rl_execute.py): fresh seasons on test days, 2-h windows at
    8/12/17 h, 20 SF + 12 Pune seeds. LiveAgentPolicy rebuilds the entity state from the
    simulator's own logs every 5 min (stage-1 DayLog, leak-checked) and the agent decides; the
    simulator carries the orders out. Every arm is also imagined by the world model from the
    same start (3 draws).

    How the method was arrived at (28_versions.png, stage5/v*/):
      v0  GRPO, 16 drivers varied per copy, 30 min      Pune worse every eval (-14.6 -> -18.2)
      v1  one driver varied per copy, 30 min             no signal: 28 comparisons/iter, noise >> effect
      probe (scripts/29_wm3_rl_probe.py): one driver's move changes the imagined reward by
          ~-0.03 (its driving cost) with s.d. 0.07-0.7 up to 20 min, but 3.0 at 30 min and 7.2
          at 60 min (SF): one changed match ripples through later matches. Riders and quotes are
          identical across the copies (demand is exogenous); it is who-gets-whom.
      v2  GRPO group = ONE driver's choices (stay + 7 destinations drawn from the agent), same
          world otherwise, reward after 20 min; probability-weighted advantages -> Pune learns,
          slowly (stay bias moved ~lr per Adam step)
      v3  + fast learning rate for the stay bias: both cities collapse to "never move"
          (Pune right, SF loses the rule's gain); destination learning was scaled by P(move)
      v4  + destination credit not scaled by P(move): still converges to (almost) never moving.
          Final held-out imagined reward minus none (60 min): pretrained +0.7 SF / +0.4 Pune,
          scratch +0.1 / +0.8; rule +8.2-8.9 / -2.6.
    Why: the rule's gain is ~+0.1 pickup per move and appears after ~1 h; a single choice
    cannot be credited over 1 h (noise 3-7), and over 20 min the average move is a small loss.
    So, as language models are fine-tuned on demonstrations before RL:
      imitation (scripts/30_wm3_rl_bc.py): the rule's choices in imagination (72k SF, 473k Pune
          decisions; all its orders are expressible as agent actions) -> cross-entropy + the
          world model's own loss (aux). Held-out accuracy .87 SF / .99 Pune.
      then GRPO (v4 credit) from the imitation agent, KL to it with beta 0.1, 40 iterations.

    Results in the simulator, 2-h window, minus no repositioning (mean +- s.e. over seeds):
                                        SF reward      moved   Pune reward    moved
      platform rule                     +25.0 +- 6.8    196     +7.7 +- 7.0     230
      RL only, pretrained (v4)           +0.5 +- 3.9      5     +0.9 +- 6.0       8
      RL only, scratch (v4)              -1.7 +- 1.0      0     -1.7 +- 3.0       1
      imitation, pretrained             +19.4 +- 4.7    199     -3.4 +- 7.1     189
      imitation, scratch                +16.0 +- 5.2    173    +11.6 +- 7.4     254
      imitation + RL, pretrained        +23.8 +- 5.7    113     -1.0 +- 6.2      13
      imitation + RL, scratch           +13.6 +- 5.1     65     -1.0 +- 5.2      33
    Paired over seeds (28_paired.txt): SF imitation+RL pretrained vs rule -1.2 +- 4.6 (the
    rule's gain with 42% fewer moves and positive drivers' welfare, +0.5 vs -0.3); RL added
    +4.4 +- 3.6 to the pretrained imitation agent and -2.4 +- 2.7 to the scratch one;
    pretrained vs scratch after imitation + RL +10.2 +- 3.5 (15 of 20 seeds). Pune: no arm
    differs from no repositioning beyond the noise (rule +7.7 +- 7.0); RL again learns to move
    little there (13 vs 230), because over 20 min a Pune move costs and brings no rider.
    The world model's imagined effects track the real ones: r .61 reward, .64 pickups over
    seeds and arms (28_scatter.png); imagined rule effect +28.8 SF / +7.5 Pune vs real
    +25.0 / +7.7. The pretrained world model also stays a world model under RL (validation
    NLL 14.89 -> 15.12), the scratch network ends at 17.6.
    Outputs (outputs/wm3/stage5/):
      28_training.png   held-out imagined reward, pickups, welfare, moves and world-model NLL
                        during GRPO, for RL-only and imitation + RL, pretrained and scratch
      28_versions.png   every method version, and the noise-vs-horizon probe
      28_execute.png    all arms executed in the simulator (bars) and imagined (diamonds)
      28_scatter.png    imagined vs real effect per seed and arm
      28_execute.csv, 28_summary.json, 28_paired.txt, 30_bc_*.csv (imitation curves),
      train_*.csv / eval_*.csv (GRPO logs), v0_group16/ v1_pairwise/ v2_driver_lr3e-4/
      v3_collapse/ (earlier versions), probe/
    Models (data/wm3_models/): rl_pre, rl_scratch (RL only, v4), bc_pre, bc_scratch
    (imitation), rl_bc_pre, rl_bc_scratch (imitation + RL).

## Stage 6: the world model also matches riders to drivers (2026-10-28)
    Goal: the agent decides not only where idle drivers go (stage 5) but which driver gets which
    rider, and its offers are carried out in the simulator.

    Matching head (world/wm3/agent.py, Agent.match_logits): every 30-s epoch the pending requests,
    oldest first, each choose one idle driver (same mode, pickup ETA within the platform's limit,
    not one who declined that request, not one taken by an earlier request) or "hold":
        driver j  qm(h_request) . km(h_driver) / sqrt(d) + w_eta * pickup minutes
        hold      MLP(h_request) + b_hold
    so a whole assignment has a probability (product of the requests' choices). Same code in
    imagination and in the simulator (LiveAgentPolicy match="agent"). The simulator's state is now
    rebuilt every epoch: state.DayLog takes the simulator's tables in memory and, in live mode,
    skips re-routing (every cut is in the past) and fills positions only for legs being driven:
    ~1 s per rebuild in SF instead of 2-8 s, features identical to the full rebuild at every
    check. With the rule matching, the refactored stepping reproduces stage 5 exactly.

    Imitation (scripts/31_wm3_match_bc.py): the rule's offers (HDDispatch min-ETA assignment per
    mode, airport queue) recorded in imagination every 2nd epoch from the 120+24 start states:
    SF 15.4k train choices (100% of rule offers expressible), Pune 25.8k (99.3%; the rest are
    airport-queue offers beyond the pickup limit). Cross-entropy + KL to the stage-5 agent's
    repositioning (so it is kept) + the world model's own loss. Held-out: exact-driver accuracy
    .86 SF / .90 Pune (pretrained, from rl_bc_pre), .87 / .90 (scratch, from rl_bc_scratch);
    chosen pickup ETA 2.00 vs rule 1.96 min SF, 3.81 vs 3.84 Pune; repositioning KL < 0.01.
    GRPO on matching (scripts/32_wm3_match_grpo.py): group = one pending request at one moment,
    copies differing only in its choice (hold + 5 drivers from the agent), 20 imagined min,
    reward pickups + welfare. The choice does matter (reward spread ~1.3 SF) but 40 iterations
    did not improve the imitation agent (held-out imagined reward minus none, SF +4.2 -> +4.3,
    Pune +0.4 -> -2.3): starting from min-ETA, which driver one rider gets changes little.
    The scratch GRPO run was skipped; pretrained vs scratch is compared on the imitation agents.

    Results in the simulator (20 SF + 12 Pune seeds, 2-h windows; 34_execute.png), reward minus
    rule matching with no repositioning:
                                              SF reward      pickups    Pune reward   pickups
      platform rule (matching + rebalancing)  +25.0 +- 6.8   +25.4     +7.7 +- 7.0   +11.7
      stage 5: rule matches, agent moves      +23.8 +- 5.7   +23.3     -1.0 +- 6.2    -1.0
      agent matches + moves, pretrained       +22.7 +- 6.0   +22.4     -0.7 +- 6.0    -4.1
      agent matches + moves, scratch          +11.5 +- 4.5   +11.4     +3.1 +- 6.2    +3.8
    Paired (34_execute_paired.txt): the pretrained agent's own matching vs stage 5 (rule's
    matching, same repositioning) -1.1 +- 3.9 SF, +0.3 +- 5.5 Pune: the world model matches as
    well as the platform; drivers rejected fewer of its offers in SF (-10.7 +- 5.3 per 2 h;
    Pune -19 +- 11). Pretrained vs scratch: +11.3 +- 3.5 SF (15 of 20 seeds), as in stage 5;
    Pune -3.7 +- 5.4 (noise). Pune's rule beats both agents (-8.3 +- 4.5) through its
    rebalancing, not its matching (stage 5 shows the same gap).
    Models: data/wm3_models/mbc_pre.pt, mbc_scratch.pt (imitation), mrl_pre.pt (+ GRPO).

## Stage 7: drivers as small agents that maximise their own earnings (2026-10-28)
    Goal: the drivers become agents; the platform's world model gives each a menu of options with
    the fare it will get, and each driver chooses to maximise its own earnings over its shift.

    world/wm3/drivers.py. Every 30-s epoch the platform (mbc_pre) scores every allowed
    (request, idle driver) pair; each idle driver sees its K = 4 best requests with, per option:
    platform rank, pickup min, trip min, fare to the driver (quote incl. surge minus commission),
    fare per engaged hour, km cost, net per hour, intercity, surge, the drop-off zone's requests
    per idle driver and minutes to the driver's home. Own features: hours online, hours since
    usual shift start, today's earnings, minutes idle, own zone's demand, cost per km, mode.
    One small policy (two MLPs, ~10k weights) shared by every driver picks an option or declines;
    drivers choose simultaneously and a request taken by several goes to the platform's best-scored
    one. Chosen rides are accepted rides (imagination: rollout preaccept; simulator: HDWorld
    preaccepted). Start policy: take the platform's first option (decline 4%).
    GRPO (scripts/33_wm3_driver_grpo.py): group = one driver's menu at one moment (a driver who
    would win at least one of its options -- otherwise its choice changes nothing; a first run that
    sampled any driver got zero signal in Pune), copies = decline + each option, reward = that
    driver's own net earnings over 30 imagined min (fares after commission minus km cost, a ride in
    progress counted at its fare), in median fares. Declining cost the driver ~0.4 fares, the best
    option beat the mean by ~0.5-0.9. Learned in ~20 iterations: never decline, take the
    platform's first (nearest) option (P = .98 SF, 1.00 Pune) -- unpaid pickup time is what a
    driver loses most, so self-interest agrees with the platform's ranking.

    Results in the simulator (same seeds; 34_execute.png), minus single offers answered by the
    simulator's people (same platform mbc_pre):
                                    SF pickups   SF wait min   Pune pickups   Pune wait min  net/h
      menus, start policy           -1.0 +- 4.2  -0.18 +- .03  +31.8 +- 4.3   -0.23 +- .03   +0.006 Pune
      menus, earning-max. agents    -1.5 +- 3.1  -0.25 +- .04  +21.8 +- 6.6   -0.15 +- .02   +0.005 Pune
    Menus with drivers who never reject cut rider waits in both cities and add ~1-2% pickups in
    Pune (where people reject 22% of offers); drivers' earnings per online hour are unchanged in SF
    and +1% in Pune (rides are limited by riders, not drivers). Trained vs start policy, paired:
    drivers' own earnings per hour equal (SF +0.001, Pune -0.001 +- 0.002), the 10th-percentile
    driver slightly up (SF +0.03 +- 0.02, Pune +0.003 +- 0.002), but in Pune riders lose (pickups
    -10.1 +- 5.1, wait +0.08 +- 0.04 min; 3 of 12 seeds up): each driver gains by never passing,
    the city does slightly worse. The world model predicted the same direction in imagination
    (Pune -6 pickups per hour, 34_drivers_training.png). In SF the trained drivers shorten waits
    (-0.07 +- 0.03 min).
    Outputs: stage6/ 34_imitation.png, 34_training.png, 34_execute.png (+ .csv, _paired.txt),
    31_*.csv, train_/eval_mrl_pre.csv; stage7/ 34_drivers_training.png, 34_execute.png (+ .csv,
    _paired.txt), train_/eval_drv_pre.csv. Models: data/wm3_models/drv_pre.pt (driver policy).
