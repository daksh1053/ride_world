# Paper 1 — Long-term Fairness in Ride-Hailing Platform (ECML-PKDD 2024)

Reproduction of Kang, Chan, Shao, Salim, Leckie, *Long-term Fairness in Ride-Hailing
Platform*, LNCS 14949, pp. 217–233 (arXiv:2407.17839).

No official implementation was released by the authors, so everything here is written
from the paper description. Every place where the paper is under-specified is marked
with an `ASSUMPTION:` comment in the source and summarised in `NOTES.md`.

## What is implemented

| Paper component | Module |
| --- | --- |
| Directed graph over Manhattan, `Geo(·,·)` shortest path (Sec. 3.1) | `ltf/graph.py` |
| Request stream / driver state / utility (Sec. 3.1) | `ltf/simulator.py` |
| Efficiency `pi(M)`, long-term fairness `F(M)`, normalised fairness (Eqs. 1, 2, 8) | `ltf/metrics.py` |
| MLP time-series request prediction (Sec. 4.2) | `ltf/forecasting.py` |
| MOMAQL + scalarisation `SR(M)` (Sec. 4.3, 4.4, Eqs. 6, 7) | `ltf/methods/momaql.py` |
| Greedy baseline (Eq. 3) | `ltf/methods/greedy.py` |
| REASSIGN (Lesmana et al.) | `ltf/methods/reassign.py` |
| LAF (Shi et al.) | `ltf/methods/laf.py` |
| Balance Ride-Pooling (Raman et al.) | `ltf/methods/ride_pooling.py` |
| Table 1, Table 2, Fig. 4, Fig. 5 | `scripts/run_experiments.py`, `scripts/make_figures.py` |

## Quick start

```bash
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt

python scripts/prepare_data.py      # download + preprocess NYC TLC 2016-03
python scripts/train_forecast.py    # MLP request predictor, reports MSE
python scripts/run_experiments.py   # Table 1, Table 2, horizon sweeps
python scripts/make_figures.py      # Fig. 4, Fig. 5
```

Outputs land in `results/` (json + csv) and `figures/`.

## Data

NYC TLC yellow taxi trip records, `2016-03`, filtered to trips whose pickup *and*
dropoff fall inside the Manhattan bounding box, dates `2016-03-01` … `2016-04-01`
(Sec. 5.1). Training data is everything before `2016-03-26`; the test horizon is the
seven days `2016-03-26` … `2016-04-01`. Only the daily peak 2-hour window is used
(Sec. 5.2).
