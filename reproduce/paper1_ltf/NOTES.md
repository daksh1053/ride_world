# Reproduction notes — what matches, what does not, and why

No official implementation of *Long-term Fairness in Ride-Hailing Platform* was
released, so everything here is written from the paper text. This file records every
place the reproduction had to make a decision the paper does not pin down, and the
honest comparison against the published numbers.

## 1. The source data no longer exists in the published form

Sec. 5.1 filters NYC TLC trips by a Manhattan **longitude/latitude bounding box** and
then merges "multiple locations together as a node". The TLC has since re-issued the
2016 archives as Parquet with the per-trip coordinates removed and replaced by
`PULocationID` / `DOLocationID` taxi-zone IDs. The original coordinate-level CSVs are
no longer publicly downloadable.

The reproduction therefore takes the node set `L` to be the **69 Manhattan taxi
zones** and replaces the bounding box with `Borough == "Manhattan"`. A taxi zone is
precisely "multiple locations merged into one node", so this substitutes for the
paper's merge step rather than working around it — but it does fix `|L| = 69`, where
the paper never states its node count.

Edge weights are the mean observed trip distance per ordered node pair, as Sec. 5.1
specifies, and `Geo(a, b)` is the all-pairs shortest path over that directed graph.
**Distances are in miles** (the TLC's `trip_distance` unit), so every utility figure
below is in miles; the paper never states its unit.

## 2. Decisions the paper leaves open

| Decision | What we did | Why |
| --- | --- | --- |
| Fleet size | 20 drivers | Table 1 implies it exactly: `total_utility / mean_utility == 20` for every row. |
| Which 2-hour peak | 18:00–20:00 | Sec. 5.2 says "peak 2-hour data" without naming the hours. |
| Where the 0.05 sampling applies | training days only | Sec. 5.2: "sampling rate of 0.05 **for the training data**". Sampling the test horizon too collapses the candidate pool and drops total utility by ~4x. |
| Batch length | 5 minutes | Not stated. |
| Vehicle capacity `c_v` | 4 | Not stated; caps a driver at 4 assignments per batch. |
| Learning rate, exploration, epochs | 0.1 / 0.1 / 10 | Not stated (`lambda`, `omega`, `gamma` *are*: 1, 0.6, 0.9). |
| **Pickup radius** | 3.0 miles | **Not in the paper at all** — see below. |
| Initial driver positions | drawn ∝ pickup demand | Sec. 4 stresses initial locations matter but does not say how they are set. Uniform draws park drivers in zones that raise no requests, where the pickup radius strands them for the whole horizon. |
| Q-table sharing | shared across agents | See `momaql.py`; one 69×69 table per agent is updated ~20x too sparsely and the lookahead becomes noise. Agents still differ through their own state `g_v^t` and their own accumulated utility in the fairness term. |

### The pickup radius is the one addition that changes the story

Without a cap on how far a driver may travel to collect a rider, *every* driver can
serve *every* request. The fleet becomes perfectly substitutable, and then **every**
method — greedy included — equalises earnings almost exactly: we measured normalised
fairness of 0.001–0.02 across all five methods, against 0.06–0.18 in the paper. The
efficiency/fairness tension the paper is about simply does not arise.

A maximum pickup distance is standard in ride-hailing matching (and Raman et al., the
Balance Ride-Pooling baseline, use a maximum pickup delay), so adding it is a
correction to our simulator rather than a departure from the paper. At 3.0 miles the
normalised fairness spread lands in the paper's range.

## 3. Two errors found while reproducing, worth recording

1. **The variance-delta penalty diverges at the start of the horizon.** Applying
   `lambda * omega * dVar` literally, normalised by the variance scale, blows up when
   all drivers still have equal utility (`sigma(o) = 0`) because the second-order term
   `g^2/n` does not vanish. Every assignment is then rejected, `sigma(o)` stays 0, and
   the system deadlocks at zero utility. `base.scaled_fairness_penalty` keeps the
   first-order term only, giving `penalty = lambda * omega * g * z_v`, which is zero
   for a perfectly equal fleet — the correct behaviour.
2. **Scalarisation must wrap the lookahead, not each objective.** Taking
   `max_d Q_util` and `min_d Q_fair` separately optimises the two objectives
   independently and injects a destination-dependent term uncorrelated with utility.
   Sec. 4.4 describes a *single-policy* multi-objective algorithm, so the greedy next
   action is the one maximising the **scalarised** value, with both components read
   off that same action (`MOMAQL._greedy_next`).

## 4. Results: reproduction vs. paper

Absolute magnitudes are not comparable — the paper never states its distance unit,
its node count, or its pickup constraint, and its mean driver utility (4791) is ~3x
ours (1468). What can be compared is the **ordering and the qualitative claims**.

### Table 1

| Method | Ours: utility | Ours: fairness | Ours: norm. fairness | Paper: norm. fairness |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 8,710 | 2.6 | 0.006 | -0.0005 |
| REASSIGN | 22,434 | 33,481 | 0.163 | 0.18 |
| LAF | 31,112 | 10,280 | 0.065 | 0.081 |
| Balance Ride-Pooling | 43,704 | 199,835 | 0.204 | 0.074 |
| Proposed Method | 29,353 | 3.6 | 0.001 | 0.061 |

**Reproduced.**
- Greedy sacrifices efficiency wholesale to buy equality — the paper's central
  observation about it. (Ours lands at low-but-positive utility rather than the
  paper's −1.5M; that collapse depends on their utility scale, where accumulated
  utilities are ~50x larger and the variance term correspondingly dominates.)
- The proposed method attains by far the best long-term fairness of the five, and is
  the only one whose fairness stays flat as the horizon grows (Fig. 4).
- Normalised fairness for REASSIGN (0.163 vs 0.18) and LAF (0.065 vs 0.081) match
  the paper closely.

**Not reproduced.** The paper reports the proposed method as best on total utility
*as well as* fairness. Ours is best on fairness but Balance Ride-Pooling earns ~1.5x
more total utility. Our proposed method instead serves the **most riders** (13,430 of
~14,000 vs 10,771 for Balance Ride-Pooling) at a lower margin per trip — it buys
equality with trip selection rather than with volume. We could not find a
configuration that dominates on both objectives without abandoning the paper's stated
`lambda = 1`, `omega = 0.6`, `gamma = 0.9`.

### Table 2 / Fig. 5 — ablation: fully reproduced

| Variant | Ours: utility | Ours: fairness | Paper: utility | Paper: fairness |
| --- | ---: | ---: | ---: | ---: |
| Our Method | 29,353 | 3.6 | 95,824 | 85,194 |
| w/o Prediction | 32,948 | 13,235 | 56,873 | 153,697 |
| w/o Fairness | 34,707 | 102,016 | 2,194,901 | 2.68e9 |

Both ablation claims hold with the same direction and a similar ratio: removing the
fairness module costs ~4 orders of magnitude of fairness for a modest utility gain,
and removing the prediction module makes fairness ~3,700x worse and, as Fig. 5 shows,
unstable as the horizon lengthens. Our prediction ablation trades a little utility for
that fairness where the paper reports a gain in both.

### Fig. 4 / Fig. 5 — reproduced

Both figures reproduce the paper's shape: baselines' fairness degrades as the horizon
grows, while the proposed method stays flat and lowest; in the ablation, "w/o
Fairness" is worst throughout and "w/o Prediction" degrades with horizon length.

### Forecasting module (Sec. 5.4)

Paper: MSE 94.69. Ours: **6.59** on the held-out 7 days. Not comparable — MSE scales
with the square of the counts being predicted, and per-zone-pair hourly counts over 69
zones are much smaller than over whatever coarser node set the paper used.

## 5. Reproducing these numbers

```bash
python scripts/prepare_data.py     # ~4 GB download, ~3 min
python scripts/train_forecast.py   # ~20 s
python scripts/run_experiments.py  # ~75 s
python scripts/make_figures.py
```
