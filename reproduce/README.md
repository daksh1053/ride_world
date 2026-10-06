# Ride-hailing fairness paper reproductions

Two independent reproductions, one per paper. Neither paper released code, so both
implementations are written from the papers' text; every point where a paper is
under-specified is marked `ASSUMPTION:` in the source and collected in that
project's `NOTES.md`, together with an honest comparison against the published
numbers.

| Directory | Paper |
| --- | --- |
| `paper1_ltf/` | Kang, Chan, Shao, Salim, Leckie — *Long-term Fairness in Ride-Hailing Platform*, ECML-PKDD 2024 (LNCS 14949, pp. 217–233; arXiv:2407.17839) |
| `paper2_vfdcfmvd/` | Shi, Zhang, Yue, Xu, Zhang, Mao — *A Fair Order Matching and Idle Vehicle Dispatching Algorithm for Ride-Hailing*, IEEE T-CSS 2026 |

The two codebases share no modules. They both build on NYC TLC yellow-taxi data
(2016-03) and the Manhattan taxi-zone map, but each downloads and preprocesses it
independently under its own `data/` directory.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r paper1_ltf/requirements.txt -r paper2_vfdcfmvd/requirements.txt
```

## Running

```bash
# Paper 1
cd paper1_ltf
../.venv/bin/python scripts/prepare_data.py
../.venv/bin/python scripts/train_forecast.py
../.venv/bin/python scripts/run_experiments.py
../.venv/bin/python scripts/make_figures.py
../.venv/bin/python tests/test_core.py

# Paper 2
cd ../paper2_vfdcfmvd
../.venv/bin/python scripts/prepare_data.py
../.venv/bin/python scripts/run_experiments.py
../.venv/bin/python scripts/run_ablation.py
../.venv/bin/python scripts/make_figures.py
../.venv/bin/python tests/test_core.py
```

Each project writes tables to `results/*.csv` + `results/*.json` and figures to
`figures/*.png`, and prints the paper's published numbers next to the reproduced
ones so the two can be compared directly.

## Read this first

`paper1_ltf/NOTES.md` and `paper2_vfdcfmvd/NOTES.md` record what reproduced, what
did not, and why — including two data-availability problems that affect both papers
(the TLC withdrew the per-trip coordinates the first paper's preprocessing needs)
and the modelling decisions that turned out to drive the results.
