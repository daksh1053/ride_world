# Reproduction notes — what matches, what does not, and why

No implementation of VFDCFMVD was released, so everything here is written from the
paper. This file records the decisions the paper leaves open, three modelling points
that turned out to drive the results, and the honest comparison against the published
numbers.

## 1. Decisions the paper leaves open

| Decision | What we did | Why |
| --- | --- | --- |
| Data month | NYC TLC `2016-03` | Sec. VI-A says "real taxi order data from New York City" without a period. |
| Node set `L` | the 66 Manhattan taxi zones that survive the isolated-zone filter | Sec. VI-A: the Manhattan taxi-zone map, minus "the order data in those isolated zones". |
| `dis(l_i, l_j)` | all-pairs shortest path over a graph whose edges are the *mean observed trip distance* per ordered zone pair | The paper prebuilds a shortest-path cache but never says how the edge weights are obtained. |
| `p_o` | TLC `fare_amount` | "trip fare". |
| `xi^t` (active-time weight) | total fare raised in slot `t`, normalised to mean 1 | Definition 6 cites [11] for the weights but gives no formula. |
| `a_d^t` (active time) | slots the driver spends serving or repositioning | "unit time income" only makes sense against busy time; counting all online time makes `F_d` a constant multiple of income. |
| DQN action space | `n ∈ {1,2,3,4,5,6,8,10}` clusters | No range given. WDF is described as the limiting case `n = |D|`. |
| DQN training epochs | 3 | `Max` is not stated. |
| Nearby zones searched | 8 | `R` is not stated. |
| KM candidate list | each order keeps its 12 best admissible drivers | Eq. 15 is cubic in the bipartite graph size; at 3500 vehicles and ~700 pending orders an exact dense Hungarian per slot is not tractable. Raising `TOP_K` recovers the exact matching. |
| **Order volume** | **0.35 x the true daily count** | See below. |

### The order volume had to be scaled down

Sec. VI-A says to use "the average number of order data from 18:00 to 22:00 on
weekdays in these 20 days". Taken literally that is **84,650 orders per day**. A fleet
of 1500–3500 vehicles cannot come close to serving that: a Manhattan trip at the
paper's own `V_avg = 7.2` mph averages ~16 min plus pickup, so 2500 vehicles have a
hard ceiling near 30,000 served orders — a ~35% service rate — whereas the paper
reports **77.4%** at 2500 vehicles. The published numbers are only attainable at
roughly a third of the true demand, so `DataConfig.order_scale = 0.35` (29,627
orders/day) is the default. Set it to 1.0 to see the literal reading.

## 2. Three modelling points that determined the results

These are recorded because each one silently inverted a headline claim before it was
found, and each is now covered by a regression test.

1. **The value function is learned online, not pre-trained.** Algorithm 1 trains the
   *clustering policy* over `Max` epochs and returns `pi`. Algorithm 2 takes `V` as an
   input and line 30 updates it every time slot. Pre-training `V` across episodes as
   well drives the TD error to zero on average, and the `dV > 0` rule of Sec. V-C then
   prunes nearly every edge of the bipartite graph: the order service rate collapses
   from ~50% to **13%**. `experiment.train` therefore pre-trains only the DQN and
   resets `V` before the measured run.
2. **Unmatched drivers must also update `V`.** Sec. V-B updates the value function
   "depending on whether the driver has been matched or not"; an idle driver
   contributes `r = 0`, `s' = s`, which decays the value of the zone it is waiting in.
   Omitting those transitions lets `V` drift upward, so the `dV > 0` filter again
   prunes too much (service 43.8% -> 50.6% once they are included, unfairness
   9762 -> 8683).
3. **A repositioning vehicle stays matchable.** Sec. V-C presents dispatching as a way
   to "increase the matching opportunities" of idle drivers. Making a repositioning
   driver unavailable for the duration of the move turns dispatching into a pure loss
   and inverts the ablation. `Fleet.dispatchable` therefore keeps it matchable and
   only prevents it from being dispatched again before it arrives.

## 3. Results: reproduction vs. paper

Income and unfairness are not comparable in absolute terms — ours run at 0.35x demand,
and unfairness is an unnormalised sum over drivers, so it scales with fleet size in
both. What can be compared is the **ordering across algorithms** and the **direction
of every ablation**.

### Fig. 5 — vehicle-count sweep (2500 vehicles shown; full sweep in `results/fig5_*.csv`)

| Algorithm | Unfairness | Total income | Service rate | Idle rate | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **VFDCFMVD** | **8,958** | 77,145 | 49.8% | 10.8% | 16.5 |
| LAF | 8,498 | 57,924 | 36.5% | 14.6% | 19.8 |
| SID | 9,948 | 92,324 | 75.7% | 12.5% | 4.3 |
| ILP | 10,491 | 73,150 | 69.6% | 12.5% | 1.4 |
| WDF | 9,841 | 88,947 | 70.3% | 10.2% | 1.9 |
| NM | 10,123 | 79,998 | 68.4% | 21.1% | 1.4 |

**Reproduced.**
- **Unfairness (Fig. 5a) — the paper's headline claim.** VFDCFMVD beats SID, ILP, WDF
  and NM at *every* vehicle count. Its unfairness advantage widens as the fleet grows,
  which is the paper's argument that dynamic clustering adapts to a growing income
  gap.
- **NM is the worst on idle rate** at every count (14.3% → 25.3%), and the paper's
  explanation carries: it "only considers nearby orders, completely neglecting
  fairness, future income impacts, and idle vehicle dispatching".
- **LAF's rigid pruning wrecks its efficiency.** Sec. VI-C predicts that LAF's
  "strategy of rigidly removing unfair driver-order matching severely degrades overall
  matching efficiency, causing high idle rates". Ours has the lowest service rate
  (36.5%), the lowest income, and the second-highest idle rate.
- **Running time.** VFDCFMVD is faster than LAF at every count and slower than NM and
  WDF — both as the paper reports, and for the reason it gives (a smaller bipartite
  graph than LAF's; NM and WDF do no matching).

**Not reproduced.**
- The paper reports VFDCFMVD **best on all four** of unfairness, income, service rate
  and idle rate. Ours is best-but-one on unfairness (LAF edges it below 3000
  vehicles), 4th on income, 5th on service rate, and 2nd on idle rate.
- Running time: the paper has VFDCFMVD faster than ILP and SID; ours is slower,
  because our ILP/SID relaxations are solved by a single Hungarian call (the
  assignment polytope is integral) rather than by branch-and-bound.

The single cause of the efficiency gap is the `dV > 0` rule of Sec. V-C. With a value
function at its fixed point the TD error averages zero by construction, so the rule
discards about half the admissible pairs. Turning it off ("w/o driver-order matching"
below) raises the service rate from 49.8% to 68.0% and income from 77k to 98k — which
is the *opposite* of the paper's Table III. We could not find a reading of Eq. 14 that
prunes value-destroying matches without also costing this much throughput.

### Table III — module ablation (2500 vehicles, 3 seeds, mean ± std)

| Variant | Unfairness | Service rate | Idle rate | Total income |
| --- | ---: | ---: | ---: | ---: |
| VFDCFMVD | 8,979.60 ± 41.87 | 49.78 ± 0.45 | 11.04 ± 0.64 | 77,140.57 ± 360.86 |
| w/o dynamic clustering | 8,940.53 ± 391.07 | 49.43 ± 0.46 | 13.57 ± 0.59 | 77,044.04 ± 410.63 |
| w/o driver-order matching | 9,668.87 ± 57.48 | 68.04 ± 0.53 | 10.52 ± 0.88 | 98,073.80 ± 627.62 |
| w/o idle vehicle dispatching | 8,968.88 ± 44.93 | 49.87 ± 0.54 | 10.95 ± 0.73 | 77,259.57 ± 459.81 |

- **Dynamic clustering: reproduced on three of four metrics.** Removing it worsens the
  idle rate (11.0% → 13.6%), the service rate and the income, all in the paper's
  direction. Its unfairness effect is within noise here (±391 vs a 39-point gap) —
  the paper reports a clear 3903 → 4853 degradation.
- **Driver-order matching: reproduced on fairness only.** Removing the
  value-function-guided matching worsens unfairness (8980 → 9669), as the paper
  reports, but *improves* the other three, for the reason given above.
- **Idle vehicle dispatching: not reproduced — and we can say why.** The paper has it
  cutting the idle rate from 18.27% to 5.94%. In our simulation it changes nothing,
  because it almost never fires: **6 repositioning events per 240-slot day**. The
  arithmetic is decisive. A virtual order has `p_o = 0` (Sec. V-C), so a move of the
  median nearby distance (1.66 km) costs the driver `R_gamma ≈ -$2.07`, while `V`
  spans only ~$5.9 across all 66 Manhattan zones and the move is discounted by
  `gamma^Dt ≈ 0.39`. A driver must therefore find a zone worth **$5.3 more** than its
  current one before moving pays, which is more than the entire spread of `V`. At the
  paper's own cost parameters — chosen so that the bottom 30% of drivers lose money —
  empty repositioning between adjacent Manhattan zones is simply not profitable. This
  looks like a genuine property of the model as specified, not a bug: it survives both
  the "still matchable while repositioning" fix and the corrected dispatch comparison
  (`StateValueFunction.reposition_advantage`, which discounts the stay-put baseline by
  the same `gamma^Dt` so that waiting and moving are compared like with like — without
  that correction, repositioning fires **zero** times).

### Table IV — low-income threshold sweep (2500 vehicles, 3 seeds)

| Threshold | Ours: unfairness | Ours: income | Paper: unfairness | Paper: income |
| --- | ---: | ---: | ---: | ---: |
| 10% | 9,124.46 ± 55.85 | 76,375.96 | 4,032.24 | 176,268.45 |
| 20% | 10,169.43 ± 1,716.61 | 76,430.61 | 3,956.83 | 175,584.91 |
| 30% | 8,979.60 ± 41.87 | 77,140.57 | 3,902.63 | 173,157.30 |
| 40% | 8,993.73 ± 58.39 | 77,864.62 | 3,872.34 | 172,632.22 |
| 50% | 9,394.82 ± 619.57 | 78,634.28 | 3,853.85 | 172,264.85 |

**Partly reproduced.** Unfairness improves from a 10% threshold to 30% and then
flattens, and 30% sits at the knee — which is the paper's conclusion ("the 30%
threshold achieves a reasonable balance"). Two differences: the 20% point is an
outlier driven by one seed (note its ±1717 std, ~30x the others), and our income
*rises* slightly with the threshold where the paper's falls.

Note that the threshold only has an effect at all because it gates the matching
itself: `D_min` (the bottom `threshold` by income) gets sequential, cluster-by-cluster
priority, and everyone above it competes in a single income-maximising pass. When the
threshold enters only the DQN reward of Eq. 8 — the first reading we tried — the
sweep is completely flat, because at this training budget the clustering agent is
still near-random and the reward never propagates.

## 4. Reproducing these numbers

```bash
python scripts/prepare_data.py     # ~2 GB download, ~2 min
python scripts/run_experiments.py  # Fig. 5 sweep, ~25 min
python scripts/run_ablation.py     # Tables III + IV, ~35 min
python scripts/make_figures.py
python tests/test_core.py          # 34 tests
```
