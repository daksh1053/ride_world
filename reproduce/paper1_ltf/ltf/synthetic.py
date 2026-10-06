"""Turn the forecaster's output into request days (paper Sec. 4.2, Fig. 3).

The prediction module outputs "the number of requests that will happen based on each
pair of locations in the next 7 days". Fig. 3 places those predicted requests in the
*action space* of MOMAQL, alongside the historical and current requests. To do that
the hourly OD volumes have to become concrete `(t_r, s_r, d_r)` requests again, which
is what this module does: the predicted counts define a multinomial over OD pairs per
hour, and requests are drawn from it and stamped with times inside that hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, DataConfig


def predicted_to_days(
    predicted: np.ndarray,
    start_time: pd.Timestamp,
    cfg: DataConfig = CONFIG.data,
    sample_rate: float | None = None,
    seed: int = 0,
) -> list[pd.DataFrame]:
    """Materialise `(H, K, K)` predicted hourly counts into per-day request frames.

    Only the peak window is kept, matching the request stream the allocator sees.
    `sample_rate` defaults to the training sampling rate so a synthetic day is the
    same size as a sampled training day.
    """
    rate = cfg.sample_rate if sample_rate is None else sample_rate
    rng = np.random.default_rng(seed)
    n_nodes = predicted.shape[1]
    peak = range(cfg.peak_start_hour, cfg.peak_start_hour + cfg.peak_hours)

    rows: list[pd.DataFrame] = []
    for h in range(predicted.shape[0]):
        stamp = start_time + pd.Timedelta(hours=h)
        if stamp.hour not in peak:
            continue
        counts = np.clip(predicted[h].ravel(), 0.0, None)
        n = int(round(counts.sum() * rate))
        if n <= 0 or counts.sum() <= 0:
            continue
        picks = rng.choice(counts.size, size=n, p=counts / counts.sum())
        s, d = np.unravel_index(picks, (n_nodes, n_nodes))
        minutes = np.sort(rng.uniform(0, 60, size=n))
        rows.append(
            pd.DataFrame(
                {
                    "pickup_time": stamp + pd.to_timedelta(minutes, unit="m"),
                    "s": s.astype(np.int16),
                    "d": d.astype(np.int16),
                }
            )
        )

    if not rows:
        return []
    frame = pd.concat(rows, ignore_index=True)
    days = [g.reset_index(drop=True) for _, g in frame.groupby(frame.pickup_time.dt.normalize())]
    return [d for d in days if len(d)]
