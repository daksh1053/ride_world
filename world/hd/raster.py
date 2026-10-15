"""Turn a simulated day of a high-detail city into what world model v2 sees.

The city is a G x G grid of square cells over its road network (G = 64: San Francisco
~180 m cells, Pune metro ~550 m). Every TICK_S (5 min) we take a snapshot:

  X  (T, C, G, G)  observed dynamic channels (CHANNELS), what a platform could log:
                   fleet by status and product, requests, pickups, drop-offs,
                   cancellations, declined quotes, surge, GPS speeds of moving cabs,
                   and the platform's own repositioning instructions (the actions)
  S  (CS, G, G)    static channels (STATIC): roads, residents, land uses, airport
  c  (T, K)        context per tick: time of day, weekday, rain (CONTEXT)
  Y  (T, G, G)     hidden truth, never an input: true congestion per cell
                   (travel time / free-flow time on the cell's roads, length-weighted)

Counts are of the window (t - 5 min, t] for flows and at t for snapshots. Products are
always three slots (bike, auto, cab); San Francisco uses only the cab slot, so one
model can read both cities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import traffic as T
from . import sim as _hd_sim  # noqa: F401  (adds the 'break' status)
from ..sim import D_STATUS

TICK_S = 300.0
G = 64
MODES3 = ["bike", "auto", "cab"]
CHANNELS = (
    [f"idle_{m}" for m in MODES3]
    + ["to_pickup", "with_rider", "repositioning", "on_break"]
    + [f"new_requests_{m}" for m in MODES3]
    + ["unmatched", "pickups", "dropoffs", "cancellations", "quote_declines"]
    + [f"surge_{m}" for m in MODES3]
    + ["gps_speed_kmh", "gps_samples", "dispatch_from", "dispatch_to"]
)
STATIC = ["roads_km", "major_roads_km", "residents", "work", "shop", "food_night", "education", "health",
          "hub", "leisure", "airport", "has_road"]
CONTEXT = ["sin_hour", "cos_hour", "sin_week", "cos_week", "weekend", "rain"]
ACTION_CHANNELS = ["dispatch_from", "dispatch_to"]
# added in step 4 (kept in separate files so older stores and models still load):
# repositioning orders counted at their origin cell by the compass octant of their target,
# which ties each "from" to its "to" (octant k covers angles around k x 45 degrees,
# measured from +column towards +row of the grid)
DIR_CHANNELS = [f"dispatch_dir{k}" for k in range(8)]
S_ = {s: i for i, s in enumerate(D_STATUS)}


@dataclass
class Grid:
    x0: float
    y0: float
    cell_m: float
    g: int = G

    @classmethod
    def for_city(cls, nodes: pd.DataFrame, g: int = G) -> "Grid":
        x, y = nodes.x_m.to_numpy(), nodes.y_m.to_numpy()
        span = max(x.max() - x.min(), y.max() - y.min()) * 1.001
        cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
        return cls(cx - span / 2, cy - span / 2, span / g, g)

    def cell(self, x, y):
        """Flat cell index (row-major, row 0 = south)."""
        i = np.clip(((np.asarray(y) - self.y0) / self.cell_m).astype(int), 0, self.g - 1)
        j = np.clip(((np.asarray(x) - self.x0) / self.cell_m).astype(int), 0, self.g - 1)
        return i * self.g + j


class CityRaster:
    """Static layers and lookups of one city; `day(dir)` rasterises one simulated day."""

    def __init__(self, city):
        d = city.data_dir
        self.city = city
        self.nodes = pd.read_parquet(d / "nodes.parquet")
        self.grid = Grid.for_city(self.nodes)
        self.node_cell = self.grid.cell(self.nodes.x_m, self.nodes.y_m)
        self.node_xy = self.nodes[["x_m", "y_m"]].to_numpy()
        zones = pd.read_parquet(d / "zones.parquet")
        pos = pd.Series(np.arange(len(self.nodes)), index=self.nodes.node_id)
        self.zone_centre_cell = self.node_cell[pos.loc[zones.centre_node].to_numpy()]
        # each cell's zone = the zone most of its nodes belong to (for surge)
        nz = self.nodes.zone.to_numpy()
        df = pd.DataFrame({"c": self.node_cell, "z": nz}).value_counts().reset_index()
        df = df.sort_values("count", ascending=False).drop_duplicates("c")
        self.cell_zone = np.full(G * G, -1)
        self.cell_zone[df.c.to_numpy()] = df.z.to_numpy()
        # edges: cell of the midpoint
        e = pd.read_parquet(d / "edges.parquet", columns=["u", "v", "length_m", "road_class"])
        u, v = pos.loc[e.u].to_numpy(), pos.loc[e.v].to_numpy()
        mid = (self.node_xy[u] + self.node_xy[v]) / 2
        self.edge_cell = self.grid.cell(mid[:, 0], mid[:, 1])
        self.edge_len = e.length_m.to_numpy()
        major = e.road_class.isin(["motorway", "trunk", "primary", "secondary"]).to_numpy()
        W = pd.read_parquet(d / "hd" / "node_weights.parquet")
        S = np.zeros((len(STATIC), G * G), np.float32)
        S[0] = np.bincount(self.edge_cell, self.edge_len / 1000, G * G) / 2          # both directions
        S[1] = np.bincount(self.edge_cell[major], self.edge_len[major] / 1000, G * G) / 2
        for k, col in enumerate(["population", "work", "shop", "food_night", "education", "health", "hub",
                                 "leisure", "airport"], start=2):
            S[k] = np.bincount(self.node_cell, W[col].to_numpy(), G * G)
        S[11] = (S[0] > 0)
        # scale each layer to [0, 1] on a log scale so cities are comparable
        for k in range(11):
            S[k] = np.log1p(S[k] / max(np.percentile(S[k][S[k] > 0], 50), 1e-9)) if (S[k] > 0).any() else S[k]
            S[k] /= max(S[k].max(), 1e-9)
        self.S = S.reshape(len(STATIC), G, G)
        self.traffic = T.TrafficModel(city)

    # -------------------------------------------------------------- one day
    def dispatch_dirs(self, day_dir: Path) -> np.ndarray:
        """(T, 8, G, G) uint8: orders issued in each 5-minute step, at their origin cell,
        split by the direction of their target zone (see DIR_CHANNELS)."""
        ev = pd.read_parquet(day_dir / "events.parquet", columns=["t_event", "type", "node", "value"])
        ds = ev[ev.type == "dispatch_sent"]
        nT = int(86400 / TICK_S)
        out = np.zeros((nT, 8, G * G), np.float32)
        if len(ds):
            a = self.node_cell[ds.node.to_numpy()]
            b = self.zone_centre_cell[ds.value.to_numpy().astype(int)]
            ang = np.arctan2(b // G - a // G, b % G - a % G)
            octant = np.round(ang / (np.pi / 4)).astype(int) % 8
            k = np.ceil(ds.t_event.to_numpy() / TICK_S).astype(int) - 1
            ok = (k >= 0) & (k < nT)
            np.add.at(out, (k[ok], octant[ok], a[ok]), 1.0)
        return np.clip(out, 0, 255).astype(np.uint8).reshape(nT, 8, G, G)

    def day(self, day_dir: Path, seed: int):
        tb = {n: pd.read_parquet(day_dir / f"{n}.parquet")
              for n in ("requests", "status", "legs", "events", "surge", "drivers")}
        date = pd.Timestamp(day_dir.name)
        ticks = np.arange(1, int(86400 / TICK_S) + 1) * TICK_S          # snapshot times t
        nT, GG = len(ticks), G * G
        X = np.zeros((nT, len(CHANNELS), GG), np.float32)
        ch = {c: i for i, c in enumerate(CHANNELS)}
        tick_of = lambda t: np.ceil(np.nan_to_num(np.asarray(t, float), nan=-1e9) / TICK_S).astype(int) - 1  # noqa: E731

        def add(name, t, cells, w=None):
            t = np.asarray(t, float)
            k = tick_of(t)
            ok = np.isfinite(t) & (k >= 0) & (k < nT)
            np.add.at(X[:, ch[name]], (k[ok], np.asarray(cells)[ok]), 1.0 if w is None else np.asarray(w)[ok])

        # ---- fleet snapshots: status and position at each tick
        st, legs, drv = tb["status"], tb["legs"], tb["drivers"]
        mode = drv["mode"].map({m: j for j, m in enumerate(MODES3)}).to_numpy()
        st = st.sort_values(["driver_id", "t"], kind="stable")
        key = st.driver_id.to_numpy() * 1e6 + st.t.to_numpy()
        # positions: status-change nodes plus every node passed on a leg
        lt = np.concatenate([x for x in legs.times]) if len(legs) else np.array([])
        ln = np.concatenate([x for x in legs.nodes]) if len(legs) else np.array([], int)
        ld = np.repeat(legs.driver_id.to_numpy(), [len(x) for x in legs.nodes])
        pk = np.concatenate([key, ld * 1e6 + lt])
        pn = np.concatenate([st.node.to_numpy(), ln])
        o = np.argsort(pk, kind="stable")
        pk, pn = pk[o], pn[o]
        nD = len(drv)
        dd = np.repeat(np.arange(nD), nT)
        tt = np.tile(ticks, nD)
        q = dd * 1e6 + tt
        i_s = np.searchsorted(key, q, side="right") - 1
        has = (i_s >= 0) & (np.floor(key[np.maximum(i_s, 0)] / 1e6) == dd)
        status = np.where(has, st.status.to_numpy()[np.maximum(i_s, 0)], S_["offline"])
        i_p = np.searchsorted(pk, q, side="right") - 1
        node = pn[np.maximum(i_p, 0)]
        cell = self.node_cell[node]
        k = np.tile(np.arange(nT), nD)
        for sname, chan in [("pickup", "to_pickup"), ("occupied", "with_rider"), ("repositioning", "repositioning"),
                            ("break", "on_break")]:
            m = status == S_[sname]
            np.add.at(X[:, ch[chan]], (k[m], cell[m]), 1.0)
        for j, mname in enumerate(MODES3):
            m = np.isin(status, (S_["idle"], S_["offered"])) & (mode[dd] == j)     # being offered = still free
            np.add.at(X[:, ch[f"idle_{mname}"]], (k[m], cell[m]), 1.0)
        # ---- requests
        r = tb["requests"]
        r = r[r.status != "not_raised"]
        oc = self.node_cell[r.origin_node.to_numpy()]
        rmode = r["mode"].map({m: j for j, m in enumerate(MODES3)}).to_numpy()
        for j, mname in enumerate(MODES3):
            m = rmode == j
            add(f"new_requests_{mname}", r.t_request.to_numpy()[m], oc[m])
        add("pickups", r.t_pickup, oc)
        add("cancellations", r.t_cancel, oc)
        dec = (r.status == "declined").to_numpy()
        add("quote_declines", r.t_request.to_numpy()[dec], oc[dec])
        inside = (r.dest_node >= 0).to_numpy()
        add("dropoffs", r.t_dropoff.to_numpy()[inside], self.node_cell[r.dest_node.to_numpy()[inside]])
        end = np.fmin(r.t_accept.fillna(np.inf), r.t_cancel.fillna(np.inf)).to_numpy().copy()
        end[dec] = r.t_request.to_numpy()[dec]
        t_req = r.t_request.to_numpy()
        for kk, t in enumerate(ticks):
            m = (t_req <= t) & (end > t)
            if m.any():
                np.add.at(X[kk, ch["unmatched"]], oc[m], 1.0)
        # ---- surge (zone, mode) -> cells, latest value at each tick
        sg = tb["surge"]
        valid = self.cell_zone >= 0
        for kk, t in enumerate(ticks):
            s = sg[sg.t == (np.floor(t / 300.0) * 300.0)]
            for j, mname in enumerate(MODES3):
                sj = s[s["mode"] == mname]
                if len(sj):
                    zmap = np.zeros(self.cell_zone.max() + 2)
                    zmap[sj.zone.to_numpy()] = sj.multiplier.to_numpy() - 1.0
                    X[kk, ch[f"surge_{mname}"], valid] = zmap[self.cell_zone[valid]]
        # ---- GPS: speed of moving cabs per cell (segments between consecutive nodes)
        for lg_n, lg_t in zip(legs.nodes, legs.times):
            if len(lg_n) < 2:
                continue
            a, b = self.node_xy[lg_n[:-1]], self.node_xy[lg_n[1:]]
            dt = np.diff(lg_t)
            ok = dt > 0.5
            if not ok.any():
                continue
            mid = (a[ok] + b[ok]) / 2
            sp = np.sqrt(((b[ok] - a[ok]) ** 2).sum(1)) / dt[ok] * 3.6
            tm = (lg_t[:-1][ok] + lg_t[1:][ok]) / 2
            c = self.grid.cell(mid[:, 0], mid[:, 1])
            add("gps_speed_kmh", tm, c, sp)
            add("gps_samples", tm, c)
        n = X[:, ch["gps_samples"]]
        X[:, ch["gps_speed_kmh"]] = np.where(n > 0, X[:, ch["gps_speed_kmh"]] / np.maximum(n, 1), 0.0)
        # ---- actions: platform repositioning instructions
        ev = tb["events"]
        ds = ev[ev.type == "dispatch_sent"]
        add("dispatch_from", ds.t_event, self.node_cell[ds.node.to_numpy()])
        add("dispatch_to", ds.t_event, self.zone_centre_cell[ds.value.to_numpy().astype(int)])
        # ---- context and hidden congestion
        day_seed = seed * 1000 + date.dayofyear
        cond = self.traffic.sample_day(str(date.date()), np.random.default_rng(day_seed + 7))
        h = ticks / 3600.0
        wk = (date.weekday() * 24 + h) / 168.0
        rain = np.array([cond.rain_at(t) for t in ticks])
        C = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * wk),
                      np.cos(2 * np.pi * wk), np.full(nT, float(date.weekday() >= 5)), rain], 1)
        Y = np.zeros((nT, GG), np.float32)
        wsum = np.bincount(self.edge_cell, self.edge_len, GG)
        for kk, t in enumerate(ticks):
            ratio = self.traffic.edge_times(cond, t - TICK_S / 2) / self.traffic.t0
            Y[kk] = np.bincount(self.edge_cell, self.edge_len * ratio, GG) / np.maximum(wsum, 1e-9)
        Y[:, wsum == 0] = 0
        return (X.reshape(nT, len(CHANNELS), G, G), C.astype(np.float32), Y.reshape(nT, G, G),
                {"date": str(date.date()), "weekday": int(date.weekday()), "rain": cond.summary()["rain"],
                 "n_incidents": int(len(cond.incidents))})


def save_day(path: Path, X, C, Y, meta):
    """Counts as uint8 (clipped), surge / speed / congestion as float16."""
    cont = [CHANNELS.index(c) for c in CHANNELS if c.startswith("surge") or c == "gps_speed_kmh"]
    cnt = [i for i in range(len(CHANNELS)) if i not in cont]
    np.savez_compressed(path, counts=np.clip(X[:, cnt], 0, 255).astype(np.uint8), cont=X[:, cont].astype(np.float16),
                        context=C, congestion=Y.astype(np.float16), count_idx=np.array(cnt), cont_idx=np.array(cont),
                        meta=json.dumps(meta))


def load_day(path: Path):
    z = np.load(path)
    T_, _, g, _ = z["counts"].shape
    X = np.zeros((T_, len(CHANNELS), g, g), np.float32)
    X[:, z["count_idx"]] = z["counts"]
    X[:, z["cont_idx"]] = z["cont"]
    return X, z["context"], z["congestion"].astype(np.float32), json.loads(str(z["meta"]))
