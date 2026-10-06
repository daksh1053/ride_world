"""World-model data: one simulated day -> zone-level observation / action tensors.

The world model of formulation 01 predicts future observations O and events E given
the history and supplied actions A. In the survey's terms ours has a *structured
(feature) substrate*: the predicted future is a per-zone count vector every
STEP_S = 5 minutes, not pixels.

Per step boundary tau_k and zone z (observed by the platform only):

    obs[k, z] =
      new      requests raised in (tau_{k-1}, tau_k]         (event request_created)
      pickups  riders picked up in (tau_{k-1}, tau_k]        (event pickup, origin zone)
      cancels  cancellations + expiries in (tau_{k-1}, tau_k] (by request origin zone)
      idle     idle cabs in z at tau_k                        (context snapshot)
      busy     cabs serving / repositioning in z at tau_k     (context snapshot)
      pending  unmatched requests from z at tau_k             (context snapshot)

    act[k, z] = decisions issued in the step that starts at tau_k (A_t, eq. 6):
      offers    offers sent at the boundary epoch (tau_k, first 30 s) to requests from z
      disp_in   reposition instructions towards z during the step
      disp_out  reposition instructions to cabs in z during the step

Causality (formulation eq. 5): event counts are bucketed by *t_available*, and
events the observation model dropped (`recorded = False`) are excluded. Only the
boundary epoch's offers are used as the action: later offers in the step react to
requests that arrive inside it, and counting them would leak the very arrivals the
model must predict (the survey's "leakage-free" requirement, sec. 5.2). Repositioning
is decided from the state, so the whole step's dispatches are used.

Exogenous context per step: hour of day, weekend, rain intensity, network TTI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STEP_S = 300.0
OBS = ["new", "pickups", "cancels", "idle", "busy", "pending"]
ACT = ["offers", "disp_in", "disp_out"]
N_STEPS = 288                      # one day


def _bin(t, k_max):
    k = np.ceil(np.asarray(t, float) / STEP_S).astype(int)    # event in (tau_{k-1}, tau_k] -> k
    return np.clip(k, 0, k_max)


def build_steps(tb: dict, n_zones: int, weekend: bool) -> dict:
    ev = tb["events"]
    ev = ev[ev.recorded]
    req = tb["requests"].set_index("request_id")
    K = N_STEPS
    obs = np.zeros((K + 1, n_zones, len(OBS)), np.float32)

    def add(df, col, zone):
        m = (zone >= 0) & (df.t_available.to_numpy() <= K * STEP_S)
        np.add.at(obs, (_bin(df.t_available.to_numpy()[m], K), zone[m], OBS.index(col)), 1)

    e = ev[ev.type == "request_created"]
    add(e, "new", e.zone.to_numpy())
    e = ev[ev.type == "pickup"]
    add(e, "pickups", req.origin_zone.reindex(e.request_id).to_numpy().astype(int))
    e = ev[ev.type.isin(["customer_cancelled", "request_expired"])]
    add(e, "cancels", req.origin_zone.reindex(e.request_id).to_numpy().astype(int))

    ctx = tb["context"]
    ctx = ctx[ctx.t <= K * STEP_S]
    k = np.round(ctx.t.to_numpy() / STEP_S).astype(int)
    z = ctx.zone.to_numpy()
    obs[k, z, OBS.index("idle")] = ctx.idle_drivers.to_numpy()
    obs[k, z, OBS.index("busy")] = ctx.busy_drivers.to_numpy()
    obs[k, z, OBS.index("pending")] = ctx.pending.to_numpy()
    g = ctx.groupby(k)
    tti = g.network_tti.first().reindex(range(K + 1)).ffill().bfill().to_numpy()
    rain = g.rain.first().reindex(range(K + 1)).ffill().fillna(0).to_numpy()

    act = np.zeros((K, n_zones, len(ACT)), np.float32)
    off = tb["offers"]
    ks = np.floor(off.t_sent.to_numpy() / STEP_S).astype(int)
    first = (off.t_sent.to_numpy() - ks * STEP_S) < 30.0          # boundary epoch only
    m = first & (ks < K)
    oz = req.origin_zone.reindex(off.request_id).to_numpy().astype(int)
    np.add.at(act, (ks[m], oz[m], ACT.index("offers")), 1)
    d = ev[ev.type == "dispatch_sent"]
    kd = np.floor(d.t_event.to_numpy() / STEP_S).astype(int)
    m = kd < K
    np.add.at(act, (kd[m], d.value.to_numpy()[m].astype(int), ACT.index("disp_in")), 1)
    np.add.at(act, (kd[m], d.zone.to_numpy()[m], ACT.index("disp_out")), 1)

    hour = (np.arange(K + 1) * STEP_S / 3600.0) % 24
    ctx_x = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                      np.full(K + 1, float(weekend)), rain, tti], 1).astype(np.float32)
    return {"obs": obs, "act": act, "ctx": ctx_x}


def zone_static(city, zones: pd.DataFrame, k_nbr: int = 6) -> dict:
    """Per-zone static features and the k nearest zones by typical free-flow time."""
    d = city.data_dir / "traffic"
    w = pd.read_parquet(d / "zone_weights.parquet")
    ff = np.load(d / "zone_tt_freeflow_s.npy").astype(float)
    Z = len(zones)
    off = ff + np.diag(np.full(Z, np.inf))
    nbr = np.argsort(off, axis=1)[:, :min(k_nbr, Z - 1)]
    static = np.stack([np.log(w.home_weight.to_numpy() * Z), np.log(w.activity_weight.to_numpy() * Z),
                       np.log1p(zones.area_km2.to_numpy()), np.log(off.min(1) / 60.0)], 1).astype(np.float32)
    return {"static": static, "nbr": nbr.astype(np.int64)}
