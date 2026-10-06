"""Time-series request prediction module (paper Sec. 4.2).

"We utilise Multi-Layer Perceptron ... a three-layer MLP ... We first utilise the
pairs of locations (start and destination locations from different requests) as
features, then multiple measurements at time t, (t-1), ..., (t-n) are used to predict
the requests that will happen in the future (the 7 days that we use to test the
model), where each time step is set as 1 hour. The structure of the request
prediction module has *number* of neurons in the hidden layer. By using the chosen
dataset, we use the previous 1 month of data for training and output the number of
requests that will happen based on each pair of locations in the next 7 days."

So one shared regressor is trained over *all* OD pairs, with the location pair itself
as a feature and the `lookback` most recent hourly counts of that pair as the
time-series input:

    x = [ one-hot(s) | one-hot(d) | c_{p,t-1} ... c_{p,t-lookback} | hour-of-day ]
    y = c_{p,t}

The MLP is implemented directly on NumPy (input layer, one hidden ReLU layer, linear
output — the "three-layer MLP" of the paper) so the reproduction needs no
deep-learning dependency. Sec. 5.4 reports an MSE of 94.69 for this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CONFIG, DATA_PROC, ForecastConfig


@dataclass
class MLPRequestPredictor:
    """Three-layer MLP over (location pair, lagged counts) features."""

    n_nodes: int
    cfg: ForecastConfig = field(default_factory=lambda: CONFIG.forecast)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        self.n_pairs = self.n_nodes * self.n_nodes
        # one-hot(s) + one-hot(d) + lags + [sin, cos] of hour-of-day
        self.n_in = 2 * self.n_nodes + self.cfg.lookback + 2
        h = self.cfg.hidden
        self.w1 = rng.normal(0, np.sqrt(2.0 / self.n_in), size=(self.n_in, h))
        self.b1 = np.zeros(h)
        self.w2 = rng.normal(0, np.sqrt(2.0 / h), size=(h, 1))
        self.b2 = np.zeros(1)
        self.y_scale = 1.0
        self.trained = False

    # ------------------------------------------------------------------ features
    def _featurise(self, series: np.ndarray, t_index: np.ndarray, pair_index: np.ndarray,
                   hour_offset: int = 0) -> np.ndarray:
        """Build the design matrix for the given (time, pair) samples.

        `series` is (T, n_pairs) of hourly counts; sample `k` predicts
        `series[t_index[k], pair_index[k]]` from the preceding `lookback` steps.
        """
        lb = self.cfg.lookback
        lags = np.stack([series[t_index - l, pair_index] for l in range(1, lb + 1)], axis=1)
        s, d = np.divmod(pair_index, self.n_nodes)

        x = np.zeros((len(t_index), self.n_in))
        x[np.arange(len(t_index)), s] = 1.0
        x[np.arange(len(t_index)), self.n_nodes + d] = 1.0
        x[:, 2 * self.n_nodes : 2 * self.n_nodes + lb] = lags / self.y_scale
        hour = (t_index + hour_offset) % 24
        x[:, -2] = np.sin(2 * np.pi * hour / 24)
        x[:, -1] = np.cos(2 * np.pi * hour / 24)
        return x

    # ------------------------------------------------------------------- forward
    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        return h, (h @ self.w2 + self.b2).ravel()

    def _predict_raw(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self._forward(x)[1], 0.0) * self.y_scale   # counts are >= 0

    # ------------------------------------------------------------------- fitting
    def fit(self, series3d: np.ndarray, verbose: bool = True) -> dict:
        """`series3d` is (T_hours, n_nodes, n_nodes) of hourly OD request counts."""
        series = series3d.reshape(series3d.shape[0], -1).astype(np.float64)
        self.y_scale = max(series.max(), 1.0)

        rng = np.random.default_rng(self.cfg.seed)
        lb = self.cfg.lookback
        valid_t = np.arange(lb, series.shape[0])

        # Only OD pairs that are ever used are worth learning; the rest are constant 0.
        active = np.flatnonzero(series.sum(axis=0) > 0)
        if active.size == 0:
            self.trained = True
            return {"train_mse": 0.0}

        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        batch = 512

        for epoch in range(1, self.cfg.epochs + 1):
            for _ in range(self.cfg.steps_per_epoch):
                t_idx = rng.choice(valid_t, size=batch)
                p_idx = rng.choice(active, size=batch)
                xb = self._featurise(series, t_idx, p_idx)
                yb = series[t_idx, p_idx] / self.y_scale

                h, out = self._forward(xb)
                err = ((out - yb) / batch)[:, None]

                gw2 = h.T @ err
                gb2 = err.sum(0)
                dh = (err @ self.w2.T) * (h > 0)
                gw1 = xb.T @ dh
                gb1 = dh.sum(0)

                step += 1
                for i, g in enumerate([gw1, gb1, gw2, gb2]):
                    m[i] = beta1 * m[i] + (1 - beta1) * g
                    v[i] = beta2 * v[i] + (1 - beta2) * g * g
                    params[i] -= self.cfg.lr * (m[i] / (1 - beta1 ** step)) / (
                        np.sqrt(v[i] / (1 - beta2 ** step)) + eps
                    )

            if verbose and epoch % 20 == 0:
                print(f"[forecast] epoch {epoch:4d}  train MSE {self.evaluate(series3d):.3f}")

        self.trained = True
        return {"train_mse": self.evaluate(series3d)}

    def evaluate(self, series3d: np.ndarray, max_samples: int = 200_000) -> float:
        """MSE over every (hour, OD pair) of the series (Sec. 5.4 reports 94.69)."""
        series = series3d.reshape(series3d.shape[0], -1).astype(np.float64)
        lb = self.cfg.lookback
        t_all, p_all = np.meshgrid(
            np.arange(lb, series.shape[0]), np.arange(series.shape[1]), indexing="ij"
        )
        t_all, p_all = t_all.ravel(), p_all.ravel()
        if t_all.size > max_samples:
            sel = np.random.default_rng(self.cfg.seed).choice(t_all.size, max_samples, replace=False)
            t_all, p_all = t_all[sel], p_all[sel]

        se = 0.0
        for start in range(0, t_all.size, 100_000):
            t_idx, p_idx = t_all[start : start + 100_000], p_all[start : start + 100_000]
            pred = self._predict_raw(self._featurise(series, t_idx, p_idx))
            se += float(np.sum((pred - series[t_idx, p_idx]) ** 2))
        return se / t_all.size

    def forecast_horizon(self, history3d: np.ndarray, n_hours: int) -> np.ndarray:
        """Roll the model forward `n_hours` steps autoregressively.

        Returns (n_hours, n_nodes, n_nodes) of predicted request counts — the future
        request pattern that MOMAQL folds into its action space.
        """
        series = history3d.reshape(history3d.shape[0], -1).astype(np.float64)
        lb = self.cfg.lookback
        window = series[-lb:].copy()
        pairs = np.arange(self.n_pairs)
        out = []
        for step in range(n_hours):
            padded = np.vstack([window, np.zeros((1, self.n_pairs))])
            t_idx = np.full(self.n_pairs, lb)
            x = self._featurise(padded, t_idx, pairs, hour_offset=len(series) + step - lb)
            nxt = self._predict_raw(x)
            out.append(nxt)
            window = np.vstack([window[1:], nxt[None, :]])
        return np.stack(out).reshape(n_hours, self.n_nodes, self.n_nodes)

    # ------------------------------------------------------------------- storage
    def save(self, path: Path | None = None) -> Path:
        path = path or DATA_PROC / "forecaster.npz"
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, y_scale=self.y_scale)
        return path

    @classmethod
    def load(cls, n_nodes: int, path: Path | None = None) -> "MLPRequestPredictor":
        path = path or DATA_PROC / "forecaster.npz"
        obj = cls(n_nodes)
        z = np.load(path)
        obj.w1, obj.b1, obj.w2, obj.b2 = z["w1"], z["b1"], z["w2"], z["b2"]
        obj.y_scale = float(z["y_scale"])
        obj.trained = True
        return obj
