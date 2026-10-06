"""Train the MLP request-prediction module (paper Sec. 4.2, MSE reported in Sec. 5.4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ltf.config import CONFIG, DATA_PROC, RESULTS
from ltf.data import load
from ltf.forecasting import MLPRequestPredictor


def main() -> None:
    bundle = load()
    counts = bundle["hourly_counts"]                     # (T, K, K)
    cfg = CONFIG.data

    # "we use the previous 1 month of data for training" and predict the 7 test days.
    split = int(
        (pd.Timestamp(cfg.test_start_date) - pd.Timestamp(cfg.start_date)) / pd.Timedelta("1h")
    )
    train, test = counts[:split], counts[split - CONFIG.forecast.lookback :]
    print(f"[forecast] train hours={len(train)}  test hours={len(test) - CONFIG.forecast.lookback}")

    model = MLPRequestPredictor(counts.shape[1])
    train_stats = model.fit(train)
    test_mse = model.evaluate(test)
    model.save()

    # Autoregressive forecast of the 7-day horizon, consumed by MOMAQL's action space.
    horizon_hours = 24 * cfg.n_test_days
    predicted = model.forecast_horizon(train, horizon_hours)
    np.savez_compressed(DATA_PROC / "predicted_requests.npz", predicted=predicted)

    out = {
        "train_mse": train_stats["train_mse"],
        "test_mse": test_mse,
        "paper_reported_mse": 94.69,
        "horizon_hours": horizon_hours,
    }
    (RESULTS / "forecast.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
