"""Trajectory observables Phi from `formulations/01_all_inputs_formulation.md` (sec. 7).

All are computed from the simulator's *world* tables (the full ledger and event log),
i.e. the ground truth a world model should reproduce:

* driver cohort D* = every driver online at any time in the window;
* u_i  net monetary earnings (eq. 15-17): receipts minus operating cost;
* F_var, F_min, mean (eq. 22) over u_i; hourly v_i = u_i / h_i (eq. 23);
* weighted-time v_i^xi and F_log (eq. 24), with xi_k = the slot's total fare / mean;
* U_dist (eq. 20): trip distance minus pickup distance, per driver;
* customer side per origin zone: completion rate s_z and served wait w_z (eq. 25),
  with denominators and cancelled / expired counts kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def driver_outcomes(tb: dict, slot_s: float = 3600.0) -> pd.DataFrame:
    led = tb["ledger"]
    drv = led[(led.party == "driver") & (led.kind != "online_time")]
    u = drv.groupby("driver_id").amount.sum()
    receipts = drv[drv.amount > 0].groupby("driver_id").amount.sum()
    cost = -drv[drv.kind == "operating_cost"].groupby("driver_id").amount.sum()
    h = led[led.kind == "online_time"].groupby("driver_id").quantity.sum() / 3600
    trips = tb["requests"].query("status == 'completed'").groupby("driver_id").size()
    legs = tb["legs"]
    pick = legs[legs.kind == "pickup"].groupby("request_id").length_m.sum()
    trip = legs[legs.kind.isin(["trip", "trip_to_gateway"])].groupby("request_id").length_m.sum()
    done = tb["requests"].query("status == 'completed'")
    udist = ((trip.reindex(done.request_id).fillna(0).to_numpy()
              - pick.reindex(done.request_id).fillna(0).to_numpy()) / 1000)
    udist = pd.Series(udist, index=done.driver_id.to_numpy()).groupby(level=0).sum()

    cohort = h.index[h > 0]
    out = pd.DataFrame(index=cohort)
    out.index.name = "driver_id"
    out["online_h"] = h
    out["net_earnings"] = u.reindex(cohort).fillna(0)
    out["receipts"] = receipts.reindex(cohort).fillna(0)
    out["operating_cost"] = cost.reindex(cohort).fillna(0)
    out["hourly"] = out.net_earnings / out.online_h
    out["trips"] = trips.reindex(cohort).fillna(0).astype(int)
    out["u_dist_km"] = udist.reindex(cohort).fillna(0)

    # weighted-time earnings (eq. 24): per-slot increments divided by slot weight xi_k
    slot = (drv.t_event // slot_s).astype(int)
    fares = led[(led.party == "driver") & (led.kind == "fare")]
    xi = fares.groupby((fares.t_event // slot_s).astype(int)).amount.sum()
    xi = xi / xi.mean()
    du = drv.groupby([drv.driver_id, slot]).amount.sum().rename("du").reset_index()
    du["xi"] = du.t_event.map(xi).fillna(1.0).clip(lower=1e-3)
    num = (du.du / du.xi).groupby(du.driver_id).sum()
    out["v_xi"] = num.reindex(cohort).fillna(0) / out.online_h
    return out.reset_index()


def fairness(d: pd.DataFrame) -> dict:
    u = d.net_earnings.to_numpy()
    v = d.hourly.to_numpy()
    vx = d.v_xi.to_numpy()
    f_log = float(-np.sum(np.log(vx / vx.max()))) if (vx > 0).all() else float("nan")
    s = np.sort(u)
    gini = float((2 * np.arange(1, len(s) + 1) - len(s) - 1) @ s / (len(s) * s.sum())) if s.sum() > 0 else float("nan")
    return {
        "cohort_size": int(len(d)),
        "mean_net_earnings": float(u.mean()), "F_var": float(u.var()), "F_min": float(u.min()),
        "gini_net_earnings": gini,
        "mean_hourly": float(v.mean()), "var_hourly": float(v.var()), "min_hourly": float(v.min()),
        "p10_hourly": float(np.percentile(v, 10)), "p90_hourly": float(np.percentile(v, 90)),
        "F_log_weighted_time": f_log,
        "F_log_defined": bool((vx > 0).all()),
        "mean_u_dist_km": float(d.u_dist_km.mean()), "var_u_dist_km": float(d.u_dist_km.var()),
    }


def zone_service(tb: dict, n_zones: int) -> pd.DataFrame:
    r = tb["requests"]
    r = r[r.status != "not_raised"]
    g = r.groupby("origin_zone")
    out = pd.DataFrame({
        "requests": g.size(),
        "completed": g.apply(lambda x: (x.status == "completed").sum()),
        "cancelled": g.apply(lambda x: (x.status == "cancelled").sum()),
        "expired": g.apply(lambda x: (x.status == "expired").sum()),
        "unresolved": g.apply(lambda x: x.status.isin(["pending", "offered", "accepted", "pickup", "in_trip"]).sum()),
        "mean_wait_min": g.apply(lambda x: x.wait_s[x.t_pickup.notna()].mean() / 60),
    }).reindex(range(n_zones))
    out.index.name = "zone"
    out["requests"] = out.requests.fillna(0).astype(int)
    out["completion_rate"] = out.completed / out.requests.replace(0, np.nan)   # s_z (NaN if no requests)
    return out.reset_index()


def service_summary(tb: dict) -> dict:
    r = tb["requests"]
    r = r[r.status != "not_raised"]
    served = r.t_pickup.notna()
    off = tb["offers"]
    return {
        "requests": int(len(r)),
        "completion_rate": float((r.status == "completed").mean()),
        "cancel_rate": float((r.status == "cancelled").mean()),
        "expire_rate": float((r.status == "expired").mean()),
        "mean_wait_min": float(r.wait_s[served].mean() / 60),
        "p90_wait_min": float(np.percentile(r.wait_s[served], 90) / 60),
        "mean_fare": float(r.fare.mean()), "mean_distance_km": float(r.distance_km.mean()),
        "intercity_trips": int(((r.external_dest >= 0) & (r.status == "completed")).sum()),
        "offer_acceptance": float((off.response == "accepted").mean()),
        "offers_per_request": float(r.n_offers.mean()),
        "cancel_reasons": r.outcome[r.status == "cancelled"].value_counts().to_dict(),
    }


# ------------------------------------------------------------------ paper metrics

def paper1_metrics(d: pd.DataFrame) -> dict:
    """Kang et al. sec. 3.2-3.3 / 5.3, on the distance utility U_dist over the cohort:
    total utility pi = sum o_v (eq. 1), F = Var(o_v) (eq. 2), F_hat = sigma / mean (eq. 8)."""
    u = d.u_dist_km.to_numpy()
    return {"p1_total_utility_km": float(u.sum()), "p1_F_var": float(u.var()),
            "p1_F_hat": float(u.std() / u.mean()) if u.mean() else float("nan"),
            "p1_min_utility": float(u.min()), "p1_max_utility": float(u.max())}


def paper2_metrics(tb: dict, cohort: np.ndarray, xi_hour: np.ndarray, floor: float = 1e-3) -> dict:
    """Shi et al. sec. III / VI-B with the reproduction's exact definitions:
    F_d = sum_t (u_d^t / xi^t) / sum_t a_d^t with a = busy time (serving/repositioning),
    unfairness = -sum_d log(max(F_d / max F, floor)); idle driver rate = share of drivers
    never active; order service rate = completed / requests; total income = sum u_d."""
    led = tb["ledger"]
    drv = led[(led.party == "driver") & (led.kind != "online_time") & led.driver_id.isin(cohort)]
    h = (drv.t_event // 3600).astype(int).clip(0, 23)
    inc = drv.amount.groupby([drv.driver_id, h]).sum().unstack(fill_value=0.0).reindex(
        index=cohort, columns=range(24), fill_value=0.0)
    weighted = (inc / xi_hour[None, :]).sum(axis=1).to_numpy()
    st = tb["status"].sort_values(["driver_id", "t"])
    busy = np.zeros(len(cohort))
    pos = {d: k for k, d in enumerate(cohort)}
    nxt_t = st.groupby("driver_id").t.shift(-1)
    dur = (nxt_t - st.t).fillna(0.0)
    is_busy = st.status.isin([3, 4, 5, 6])
    for d_, s_ in dur[is_busy].groupby(st.driver_id[is_busy]).sum().items():
        if d_ in pos:
            busy[pos[d_]] = s_ / 3600.0
    f_d = np.divide(weighted, busy, out=np.zeros_like(weighted), where=busy > 0)
    ratio = np.clip(f_d / f_d.max(), floor, 1.0) if f_d.max() > 0 else np.full(len(f_d), np.nan)
    r = tb["requests"]
    r = r[r.status != "not_raised"]
    done = r[r.status == "completed"]
    return {"p2_unfairness": float(-np.log(ratio).sum()),
            "p2_unfairness_per_driver": float(-np.log(ratio).mean()),
            "p2_total_income": float(inc.to_numpy().sum()),
            "p2_idle_driver_rate": float(np.mean(~np.isin(cohort, done.driver_id.unique()))),
            "p2_order_service_rate": float(len(done) / len(r))}
