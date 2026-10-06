"""Configuration for the VFDCFMVD reproduction.

Values in Table II of the paper are quoted directly. Anything the paper leaves open
is marked ASSUMPTION here and in `NOTES.md`.
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

MILES_TO_KM = 1.609344


@dataclass(frozen=True)
class DataConfig:
    """NYC TLC order data and the Manhattan taxi-zone map (Sec. VI-A)."""

    base_url: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    month: str = "2016-03"
    zone_url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    zone_name: str = "taxi_zone_lookup.csv"
    borough: str = "Manhattan"

    # Table II: "The time period 18:00 to 22:00", weekdays (Sec. VI-A: "we choose
    # the time period (18:00 to 22:00) on weekdays for the evaluation").
    start_hour: int = 18
    end_hour: int = 22

    # "the average number of order data from 18:00 to 22:00 in these 20 days"
    n_days: int = 20

    # ASSUMPTION: a multiplier on the daily order count. At 1.0 (the literal reading
    # of Sec. VI-A) the raw Manhattan peak demand is ~85k orders, which 1500-3500
    # vehicles cannot come close to serving — every algorithm saturates at a 15-30%
    # service rate, whereas the paper reports ~77% at 2500 vehicles. Scaling demand
    # down puts the simulation in the paper's operating regime; see NOTES.md.
    order_scale: float = 0.35

    seed: int = 0


@dataclass(frozen=True)
class PlatformConfig:
    """Table II — experimental parameter settings."""

    slot_seconds: int = 60                 # "Length of time slot: 60 s"
    n_slots: int = 240                     # 18:00-22:00 at 60 s per slot

    # "Number of vehicles |D|: 1500, 2000, 2500, 3000, 3500"
    vehicle_counts: tuple[int, ...] = (1500, 2000, 2500, 3000, 3500)
    default_vehicles: int = 2500           # the ablation's "medium-scale" setting

    # "Passenger's maximum waiting time t_o^w (min): 3, 4, 5, 6, 7, 8"
    waiting_minutes: tuple[int, ...] = (3, 4, 5, 6, 7, 8)

    # "The average speed of vehicles V_avg: 7.2 mph"
    v_avg_mph: float = 7.2

    # "Unit cost of drivers c_d ($/km): {6, 8, 10} x 2.5/6.8/1.6".
    # Footnote 7: the ratio of the unit basic fare (2.5 $/mile, converted with
    # 1.6 km/mile) to an average fuel consumption of 6.8 L/100km, multiplied by each
    # fuel-consumption value. => c_d in {6,8,10} * (2.5/1.6)/6.8 $/km.
    fuel_consumptions: tuple[float, ...] = (6.0, 8.0, 10.0)
    base_fare_per_mile: float = 2.5
    km_per_mile: float = 1.6
    reference_consumption: float = 6.8

    seed: int = 0

    @property
    def v_avg_kmh(self) -> float:
        return self.v_avg_mph * self.km_per_mile

    @property
    def unit_costs(self) -> tuple[float, ...]:
        ratio = (self.base_fare_per_mile / self.km_per_mile) / self.reference_consumption
        return tuple(f * ratio for f in self.fuel_consumptions)


@dataclass(frozen=True)
class AlgorithmConfig:
    """VFDCFMVD hyper-parameters (Sec. VI-A, "Network Parameters Settings")."""

    # Driver state value function V(s): "learning rate alpha is set to 0.025 and the
    # discount factor gamma to 0.9", stored in a table rather than a network.
    value_lr: float = 0.025
    value_gamma: float = 0.9

    # Driver dynamic clustering DQN: "learning rate alpha_0 to 0.001 and the discount
    # factor gamma_0 to 0.9 ... memory capacity to 2000 ... batch size to 32 ...
    # update the target network every 200 rounds".
    dqn_lr: float = 0.001
    dqn_gamma: float = 0.9
    dqn_memory: int = 2000
    dqn_batch: int = 32
    dqn_target_update: int = 200
    dqn_hidden: int = 64          # "a three-layer small neural network"
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.05
    dqn_epsilon_decay: int = 2000
    dqn_train_episodes: int = 3   # ASSUMPTION: not stated ("maximum training epochs" Max).

    # Action space of the clustering agent: the number of driver clusters.
    # ASSUMPTION: the paper gives no range; WDF is described as the limiting case
    # where the cluster count equals the driver count.
    cluster_choices: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10)

    # Sec. V-A: "we define the bottom 30% of drivers by income in the current time
    # slot as the main group for fairness improvement". Table IV sweeps it.
    low_income_threshold: float = 0.30

    # Idle dispatching searches the R nearest zones (Sec. V-D's N_idle * R term).
    n_nearby_zones: int = 8

    # Number of value-function pre-training passes over the order stream.
    value_train_episodes: int = 3

    seed: int = 0


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)


CONFIG = Config()
