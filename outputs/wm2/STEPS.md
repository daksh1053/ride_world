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
