# Paper 2 — A Fair Order Matching and Idle Vehicle Dispatching Algorithm for Ride-Hailing

Reproduction of Shi, Zhang, Yue, Xu, Zhang & Mao, *A Fair Order Matching and Idle
Vehicle Dispatching Algorithm for Ride-Hailing*, IEEE Transactions on Computational
Social Systems (2026), DOI 10.1109/TCSS.2026.3712745.

No official implementation was released, so everything here is written from the paper.
Under-specified points are marked `ASSUMPTION:` in the source and collected in
`NOTES.md`. This codebase is completely independent of `../paper1_ltf`.

## What is implemented

| Paper component | Module |
| --- | --- |
| Road network `G = (L, E)`, `dis(·,·)`, shortest-path cache (Def. 1, Sec. VI-A) | `vfd/graph.py` |
| Drivers, orders, income `u_d^t`, cost `C_d^o` (Defs. 2–5, Eqs. 1–3) | `vfd/simulator.py` |
| Weighted unit time income `F_d`, temporal earnings fairness `F` (Defs. 6–7, Eqs. 4–5) | `vfd/metrics.py` |
| State value function `V(s)`, `R_gamma`, `dV` (Sec. V-B, Eqs. 11–14) | `vfd/value_function.py` |
| Driver dynamic clustering DQN (Sec. V-A, Alg. 1, Eqs. 8–10) | `vfd/dqn.py` |
| VFDCFMVD: clustering + KM matching + idle dispatching + update (Sec. V-C, Alg. 2) | `vfd/methods/vfdcfmvd.py` |
| NM, LAF, WDF, ILP, SID benchmarks (Sec. VI-B) | `vfd/methods/baselines.py` |
| Fig. 5(a)–(e), Table III, Table IV | `scripts/` |

## Quick start

```bash
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt

python scripts/prepare_data.py     # download + preprocess NYC TLC 2016-03
python scripts/run_experiments.py  # Fig. 5 vehicle-count sweep
python scripts/run_ablation.py     # Table III + Table IV
python scripts/make_figures.py     # Fig. 5(a)-(e)
python tests/test_core.py          # 34 unit tests
```

Outputs land in `results/` (json + csv) and `figures/`.

## Setting

* **Data** — NYC TLC yellow-taxi records, Manhattan taxi zones, weekdays 18:00–22:00
  (Sec. VI-A). 66 zones survive the isolated-zone filter.
* **Horizon** — `T = 240` time slots of 60 s (Table II).
* **Fleet** — 1500 … 3500 vehicles, `V_avg = 7.2` mph, unit cost
  `c_d ∈ {1.379, 1.838, 2.298}` $/km, passenger waiting time 3–8 min (Table II).
* **Metrics** — unfairness, drivers' total income, idle driver rate, order service
  rate, running time (Sec. VI-B).
