"""Central configuration for the Long-term Fairness reproduction.

Values quoted from the paper carry a section/equation reference. Values the paper
does not state are marked ASSUMPTION and were chosen to keep the simulation at the
scale implied by Table 1 (20 drivers, mean driver utility ~4.8e3 over seven days).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (DATA_RAW, DATA_PROC, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    """NYC TLC extraction (paper Sec. 5.1)."""

    base_url: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    # 2016-04 is needed because the test horizon ends *on* 01/04/2016 (Sec. 5.2).
    months: tuple[str, ...] = ("2016-03", "2016-04")
    zone_url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    zone_name: str = "taxi_zone_lookup.csv"

    # Sec. 5.1: "dates ranging from 01/03/2016 to 01/04/2016"; `end_date` is
    # exclusive, so it covers the whole of 01/04.
    start_date: str = "2016-03-01"
    end_date: str = "2016-04-02"

    # The node set L is the Manhattan taxi zones (69 of them) — the TLC's own
    # "multiple locations merged together as a node". See ltf/data.py for why the
    # paper's lon/lat bounding box is no longer reproducible from public data.
    borough: str = "Manhattan"

    # Sec. 5.2: "we extracted peak 2-hour data". ASSUMPTION: the evening peak.
    peak_start_hour: int = 18
    peak_hours: int = 2

    # Training/test split (Sec. 5.2).
    test_start_date: str = "2016-03-26"
    n_test_days: int = 7

    # Sec. 5.2: "we used a stratified sampling method with a sampling rate of 0.05
    # for the training data". The sampling is explicitly for *training* only — the
    # test horizon is scored against the full peak demand, which is what leaves the
    # matchers a large candidate pool to select from. Stratified per
    # (date, hour, origin node).
    sample_rate: float = 0.05
    test_sample_rate: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class SimConfig:
    """Simulation of drivers and the assignment process (paper Sec. 3.1)."""

    # Table 1 implies 20 drivers (total utility / mean utility == 20 for every row).
    n_drivers: int = 20

    # c_v, vehicle capacity. ASSUMPTION: homogeneous fleet of 4-seaters.
    capacity: int = 4

    # Requests are collected and assigned in batches. ASSUMPTION: 5-minute batches.
    batch_minutes: int = 5

    # ASSUMPTION (not stated in the paper): a rider is only matched to a driver
    # within this pickup distance. Without such a radius every driver can serve
    # every request, the fleet becomes perfectly substitutable, and *any* method
    # equalises earnings almost exactly — collapsing the efficiency/fairness
    # tension the paper measures. Units are the dataset's trip distance (miles).
    max_pickup_distance: float = 3.0

    # Each driver may take several requests concurrently (Sec. 4.3); a batch is
    # therefore resolved in successive assignment rounds.
    max_rounds_per_batch: int = 64

    seed: int = 0


@dataclass(frozen=True)
class MethodConfig:
    """Objective and learning hyper-parameters (Sec. 5.2)."""

    lam: float = 1.0        # lambda in Eqs. 3 and 7
    omega: float = 0.6      # omega in Eq. 7
    gamma: float = 0.9      # MOMAQL discount factor
    alpha: float = 0.1      # ASSUMPTION: Q-learning rate (not stated).
    epsilon: float = 0.1    # ASSUMPTION: exploration rate during training.
    train_epochs: int = 10  # ASSUMPTION: passes over the training stream.

    # REASSIGN: allowed relative utility loss w.r.t. the max-utility matching.
    reassign_utility_slack: float = 0.2
    reassign_iters: int = 200


@dataclass(frozen=True)
class ForecastConfig:
    """MLP request predictor (Sec. 4.2)."""

    # "multiple measurements at time t, (t-1), ..., (t-n)"
    lookback: int = 24
    hidden: int = 128       # "number of neurons in the hidden layer"
    epochs: int = 100
    steps_per_epoch: int = 50
    lr: float = 1e-3
    seed: int = 0


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)


CONFIG = Config()
