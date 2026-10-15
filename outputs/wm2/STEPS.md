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
