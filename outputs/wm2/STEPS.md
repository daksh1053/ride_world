# World model v2: log of steps

Each step keeps its own results folder (outputs/wm2/stepN/) and its own checkpoint names
(data/wm2_models/<tag>.pt), so later steps never overwrite earlier ones.

## Step 1: detailed cities (sf_hd, pune_hd)
Scripts 09_hd_season.py, 09b_hd_validate.py. Results in outputs/sf_hd/09_* and outputs/pune_hd/09_*.

## Step 2: high-resolution dataset
Script 10_hd_dataset.py, viewer 10b_wm2_viewer.py (outputs/<city>/10_wm2_viewer.html).
SF 120 days (2025-01-06 to 2025-05-05, seed 1), Pune 100 days (2025-05-05 to 2025-08-12, seeds 1 and 2).
64 x 64 grid, 288 frames of 5 min per day, 22 channels, random platform policy per day.

## Step 3: first models (outputs/wm2/step3/)
Commands:
    11_wm2_train.py --build
    11_wm2_train.py --model st --steps 20000 --tag st_small
    11_wm2_train.py --model cnn --steps 6000 --tag cnn
    11_wm2_train.py --finetune st_small --rollout 4 --steps 2500 --lr 1e-4 --eval-every 500 --tag st_ro
    11_wm2_train.py --eval st_small st_ro cnn
    12_wm2_analysis.py --tag st_small --only results rollout probes
    12_wm2_analysis.py --tag st_ro
Test Poisson NLL (SF / Pune): persistence .228/.501, climatology .127/.322, cnn .117/.306,
st_small .096/.248, st_ro .103/.264.
Rollout to 60 min: st_small explodes (12.1 / 18.5); st_ro stays below climatology in SF
(.123 vs .128), in Pune until about 40 min (.612 at 60 min vs .333).
Congestion probe R2 (SF / Pune): trained tokens .70/.77, raw inputs .59/.66, untrained .45/.45.
Counterfactual (move 40 drivers): drivers leave the source but never arrive in the target.

## Step 4: action understanding, direction channels, action-rich data (outputs/wm2/step4/)
4a. Ground-truth counterfactuals (13_cf_truth.py, data/<city>/wm2_cf/): same morning twice, with
    and without one extra order at 08:00 (SF 40, Pune 80 drivers to an under-supplied zone),
    SF 12 seeds, Pune 8 seeds; 13b_cf_eval.py compares the models' predicted effect.
    First finding (SF, 4 seeds): true effect -20 idle in the source at +10 min, +8 idle in
    the target at +30 min; st_small one-step predicts only -5.7 of the -39 immediate drop and its
    rollout diverges; st_ro barely reacts. Cause: the training data almost never has such
    concentrated orders (99.99th percentile 6 orders per 4x4 block per 5 min).
4b. Dispatch-direction channels (raster DIR_CHANNELS, store/dirs.npy, --dirs): orders counted at
    their origin by the compass octant of their target. Model st4_k6_dirs = st_small + dirs.
4c. Action-rich data: 10_hd_dataset.py --blocks (HDDispatchBlocks: random block moves of 10-60 / 10-120
    drivers about every 40 min). SF seed 3 60 days from 2024-11-06, Pune seeds 3, 4 30 days each from
    2025-03-06. Split fixed by date (data/<city>/wm2/split.json) so these are training days only and
    the val/test days are unchanged.
    Stores rebuilt: SF 180 days (150 train / 12 val / 18 test), Pune 160 days (135 / 10 / 15).
    Models (all 20k steps unless noted, K=6):
      st4_large      8.79M params (d 256, 8 layers), old data, 15k steps   test NLL SF .0960 Pune .2456 (best one-step)
      st4_blocks     st + dirs on the action-rich data                       SF .0966 Pune .2478
      st4_blocks_ro  st4_blocks fine-tuned on 6-step rollouts, 2500 steps   SF .1044 Pune .2689
    Counterfactual result (outputs/wm2/step4/13_cf_eval.*): more action-rich data did NOT fix it.
    Teacher-forced models still blow up in 60-min rollouts (relative error 25-75 SF, 2.3-7.6 Pune);
    rollout fine-tuned ones are stable but barely react (rel. error .96 SF / .82 Pune, map r .38 / .72).

## Step 4d: why the models miss the action (outputs/wm2/step4d/, step4d_sample/)
    Diagnosis with teacher forcing (true history, scratch scripts):
    - one step after the order (10 min) all models match the truth (SF source idle -21.4 vs -21.3);
      at the order frame itself they predict -9 vs -40 (SF), -38 vs -78 (Pune).
    - on ordinary validation frames the models are calibrated per dispatch (1 order: -0.62 vs -0.77
      idle; 2 orders: -1.05 vs -0.96). About 30% of dispatch orders are declined within 2-10 s
      and the driver is idle again, so in the data an order means about -0.75 idle drivers.
    - the stage-13 order is issued at exactly 08:00:00, the instant of a snapshot, so that frame
      catches every ordered driver before declines (all 40 gone). Only 0.9% of ordinary orders sit
      on a snapshot boundary: the benchmark's first frame was out of distribution. Fixed by
      re-simulating with the order at 08:02:30 (13_cf_truth.py --t-order 28950 --out wm2_cf2,
      14_wm_plan.py now writes wm2_plan2). The old truth is kept in wm2_cf/ and wm2_plan/.
    Tried:
    - action-focused sampling (--act-focus p: share of training windows centred on frames with
      >= 8 orders from one 4x4 block; store/act_peak.npy): st4_focus_ft (st4_blocks + 5k steps,
      p .3) val .2786, action response unchanged.
    - per-cell skip path in the transformer (--skip: 3x3 conv MLP from each step's inputs added to
      the output, zero-initialised): st4_skip_ft (st4_focus_ft + 5k steps) val .2746, the best
      validation loss so far, but the same action response and rollouts still blow up.
    - rollouts that feed back Poisson samples clamped to 0-255 instead of means (13b --feed sample):
      blow-ups are smaller (SF rel. error 17 vs 72, Pune .91 vs 2.3) but not gone.
    - st4_skip_ro: st4_skip_ft fine-tuned on 6-step rollouts with --act-focus .5 (running).

## Step 5: planning with the world model (14_wm_plan.py, outputs/wm2/step5/)
    At 08:00 send N drivers (SF 40, Pune 80) from the crowded source area to one of the 6 most
    under-supplied zones within 20 min, or do nothing. Each of the 7 candidates simulated on the
    same morning, 4 seeds (data/<city>/wm2_plan/), outcome = pickups / lost riders 08:00-09:00.
    Truth: SF every forced move costs 5-12 pickups (the dispatcher is already balanced; doing nothing
    is best). Pune differs by target: zone 6 +7, zone 31 +5, zone 45 -19 pickups.
    Models (first version, order on the snapshot boundary): rank correlation of predicted vs true
    pickup effects SF -0.86 to 0.07, Pune -0.04 to 0.43. st4_large and st4_blocks choose "do nothing"
    in SF (regret 0) and in Pune (regret 7); none can rank the candidates yet.
