"""World model 3: HTML reports for stages 6, 7 and 8, with map replays that show every action.

    .venv/bin/python scripts/38_wm3_reports.py --collect          # one-off recordings (GPU, ~30 min)
    .venv/bin/python scripts/38_wm3_reports.py --build            # writes the three pages

Each page has a replay of one hour (12:00-13:00, the first hour the platform's new decision maker
is in charge; seed 700 of the stage-5 test seasons) on two maps side by side, with the actions drawn
on the map and listed on the side, then the stage's figures with what they show.
  stage 6   platform's rule  vs  the world model's agent (matching + repositioning), both in the simulator
  stage 7   single offers answered by the simulated people  vs  menus chosen by the trained driver agents
  stage 8   a shock in the simulator  vs  the same shock imagined by the world model (3 shocks)
Recordings that the earlier runs did not keep:
  stage 7   the menus each driver saw (the day is run again with a recorder; data/<city>/wm3_reports/s7_*)
  stage 8   the imagined shocked hours, frame by frame (data/<city>/wm3_reports/s8_imagined.pkl)
"""

from __future__ import annotations

import argparse
import base64
import copy
import gzip
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from world import cities
from world.wm3 import agent as AG
from world.wm3 import drivers as DR
from world.wm3 import features as FT
from world.wm3 import rollout as RO
from world.wm3 import state as S

_spec = importlib.util.spec_from_file_location("ex27", Path(__file__).with_name("27_wm3_rl_execute.py"))
EX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EX)
_quiet = EX._quiet                                   # the pickled warm seasons refer to __main__._quiet
TR = EX.TR
SEED = 700
FRAMES = 120                                          # one hour of 30-s epochs
SHOCKS = ["supply60", "supply30", "price15"]
W3 = cities.OUTPUTS / "wm3"
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
KEYS = ["sf_hd", "pune_hd"]


def root(key):
    return cities.get(key).data_dir / "wm3_rl" / "exec"


def rep_dir(key):
    d = cities.get(key).data_dir / "wm3_reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def base_rec(key):
    return json.loads((root(key) / f"rec_seed{SEED}.json").read_text())


# ---------------------------------------------------------------------------- recordings
def collect_s7(key):
    """Runs the stage-7 arm (platform mbc_pre, menus, trained drivers drv_pre) again for seed 700,
    recording every menu: who saw which rides, with what fare, and what they chose."""
    out = rep_dir(key) / "s7_menus.pkl"
    if out.exists():
        return
    rec = base_rec(key)
    date, t0 = pd.Timestamp(rec["date"]), float(rec["t0"])
    city = cities.get(key)
    gr = S.CityGraph(city)
    rules = RO.CityRules(gr)
    prior = FT.Prior()
    for d in sorted(p for p in (root(key) / f"seed{SEED}").iterdir() if p.is_dir()):
        prior.update_light(d, 5000)
    tabs = TR.tables()
    agent = EX.load_agent("mbc_pre", tabs)
    pol = AG.LiveAgentPolicy(agent, gr, rules, prior, t_start=t0, seed=SEED, mode="agent", match="agent",
                             drivers=EX.load_drivers("drv_pre"))
    rounds = []
    orig = AG.menu_round

    def recording(W, mi, lg, f):
        offers = orig(W, mi, lg, f)
        if W.t >= t0 + FRAMES * S.DT + 60:
            return offers
        menu, _ = DR.menus(mi, lg)
        cols = np.array(sorted(menu), np.int64)
        lm, won = W.last_menu, set(offers)
        for i, c in enumerate(cols):
            d = int(mi["drv"][c])
            reqs = [int(mi["req"][r]) for r in menu[c]]
            n = len(reqs)
            a = int(lm["a"][i])
            rounds.append({"t": float(W.t), "d": d, "req": reqs, "node": [int(W.req[k]["origin"]) for k in reqs],
                           "fare": lm["fare"][i, :n].round(2).tolist(), "pick": (lm["opt"][i, :n, 1] * 10).round(1).tolist(),
                           "trip": (lm["opt"][i, :n, 2] * 30).round(1).tolist(),
                           "net": (lm["opt"][i, :n, 6] * 4 * W.money).round(1).tolist(), "a": a,
                           "won": bool(a > 0 and (d, reqs[a - 1]) in won)})
        return offers

    AG.menu_round = recording
    ta = time.time()
    try:
        state = EX.warm_state(key, SEED, root(key), 3)
        day_dir = rep_dir(key) / "s7_drv_pre" / str(date.date())
        EX.run_day_with(copy.deepcopy(state), date, pol, t0 / 3600 + FRAMES * S.DT / 3600 + 0.1, day_dir)
    finally:
        AG.menu_round = orig
    out.write_bytes(pickle.dumps({"rounds": rounds, "day_dir": str(day_dir)}))
    print(f"  {key} stage 7 recorded: {len(rounds)} menus [{(time.time() - ta) / 60:.1f} min]", flush=True)


def collect_s8(key):
    """The three shocks imagined by the world model from the base run's state at 12:00 (draw 11,
    as in script 36), with positions, riders waiting and every event, frame by frame."""
    out = rep_dir(key) / "s8_imagined.pkl"
    if out.exists():
        return
    rec = base_rec(key)
    shock = json.loads((root(key) / f"shock_seed{SEED}.json").read_text())
    date, t0 = pd.Timestamp(rec["date"]), float(rec["t0"])
    city = cities.get(key)
    gr = S.CityGraph(city)
    rules = RO.CityRules(gr)
    prior = FT.Prior()
    for d in sorted(p for p in (root(key) / f"seed{SEED}").iterdir() if p.is_dir()):
        prior.update_light(d, 5000)
    model = TR.load("wm3_exo")[0].eval()
    Lb = S.DayLog(gr, root(key) / f"seed{SEED}_rule" / str(date.date()))
    e0 = int(t0 // S.DT)
    Ws = []
    for arm in SHOCKS:
        W = AG.AgentWorld(model, rules, Lb, prior, e0, None, seed=11)
        W.setup(None, 11, "rule")
        if arm.startswith("supply"):
            ban = set(shock["arms"][arm]["shock"]["removed"])
            for d in ban:
                if W.st[d] in (RO.IDLE, RO.BREAK):
                    W._offline(d, W.t)
            on_online = W._on_online
            W._on_online = lambda t, d, push, t1, f=on_online, ban=ban: None if d in ban else f(t, d, push, t1)
        else:
            W.surge = np.maximum(W.surge, 1.5)
            sg = W._surge

            def surge(W=W, sg=sg):
                sg()
                W.surge = np.maximum(W.surge, 1.5)
            W._surge = surge
        W.L = None
        Ws.append(W)
    rec_out = {arm: {"pos": [], "st": [], "riders": [], "events": [], "origin": {}} for arm in SHOCKS}
    ptr = [len(W.log) for W in Ws]

    def snap():
        for arm, W, i in zip(SHOCKS, Ws, range(len(Ws))):
            r = rec_out[arm]
            r["pos"].append(W.positions(W.t)[0].astype(np.float32))
            r["st"].append(W.st.astype(np.int8).copy())
            r["riders"].append(np.array([q["origin"] for q in W.req.values() if q["status"] in ("pending", "offered")], np.int32))
            for x in W.log[ptr[i]:]:
                r["events"].append(x)
                if x[1] == "offer_sent" and x[3] in W.req:
                    r["origin"][x[3]] = int(W.req[x[3]]["origin"])
            ptr[i] = len(W.log)

    snap()
    ta = time.time()
    chunk = {"sf_hd": 8, "pune_hd": 4}[key]
    for _ in range(FRAMES):
        AG.rollout_many(Ws, 1, chunk)
        snap()
    for arm in SHOCKS:
        r = rec_out[arm]
        r["pos"], r["st"] = np.stack(r["pos"]), np.stack(r["st"])
    out.write_bytes(pickle.dumps({"t0": t0, "date": str(date.date()), "arms": rec_out}))
    print(f"  {key} stage 8 imagined [{(time.time() - ta) / 60:.1f} min]", flush=True)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- page data
CUR = {"sf_hd": ("$", 2), "pune_hd": ("\u20b9", 0)}
RESP = {"accepted": 0, "rejected": 1, "timeout": 2}


def gzb(a):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 6)).decode()


class Geo:
    """Road graph packed for the page (positions in 2-m units from the south-west corner)."""

    def __init__(self, key):
        self.g = g = S.CityGraph(cities.get(key))
        self.x0, self.y0 = float(g.x.min()), float(g.y.min())
        z = g.zone
        self.znode = np.zeros(int(z.max()) + 1, np.int64)
        for k in range(len(self.znode)):
            idx = np.flatnonzero(z == k)
            if len(idx):
                cx, cy = g.x[idx].mean(), g.y[idx].mean()
                self.znode[k] = idx[np.argmin((g.x[idx] - cx) ** 2 + (g.y[idx] - cy) ** 2)]

    def q(self, v, o):
        return np.clip(np.round((v - o) / 2.0), 0, 65534).astype(np.uint16)

    def pack(self):
        g = self.g
        cls = pd.Series(g.road_class).map({"motorway": 3, "trunk": 3, "primary": 2, "secondary": 1}).fillna(0).to_numpy(np.uint8)
        return {"nodes": {"x": gzb(self.q(g.x, self.x0)), "y": gzb(self.q(g.y, self.y0))},
                "edges": {"u": gzb(g.eu.astype(np.uint32)), "v": gzb(g.ev.astype(np.uint32)), "cls": gzb(cls)}}


def frames(geo, xy, st, riders, dids):
    """xy (ne, nD, 2) metres, st (ne, nD) -> packed arrays for the columns dids."""
    ne = len(st)
    x = np.full((len(dids), ne), 65535, np.uint16)
    y = x.copy()
    s = np.zeros((len(dids), ne), np.int8)
    for k in range(ne):
        on = st[k][dids] != 0
        x[on, k] = geo.q(xy[k][dids[on], 0], geo.x0)
        y[on, k] = geo.q(xy[k][dids[on], 1], geo.y0)
        s[:, k] = st[k][dids]
    off = np.r_[0, np.cumsum([len(r) for r in riders])].astype(np.int32)
    rn = np.concatenate(riders).astype(np.int32) if off[-1] else np.zeros(0, np.int32)
    return {"n": int(len(dids)), "ne": int(ne), "x": gzb(x), "y": gzb(y), "st": gzb(s), "ro": gzb(off), "rn": gzb(rn),
            "dids": [int(d) for d in dids]}


def cum(es, ne):
    """Cumulative count per frame of events at frames es."""
    es = np.asarray(es, int)
    es = es[(es >= 0) & (es < ne)]
    return np.cumsum(np.bincount(es, minlength=ne)).astype(int).tolist()


def real_run(geo, day_dir, t0, name, sub):
    """A simulator day (saved tables) from t0 for FRAMES epochs: positions, riders and actions."""
    L = S.DayLog(geo.g, day_dir)
    e0, ne = int(t0 // S.DT), FRAMES + 1
    xy = [L.xy(e0 + k) for k in range(ne)]
    st = [L.status[:, e0 + k].astype(np.int8) for k in range(ne)]
    riders = []
    for k in range(ne):
        R = L.open_requests(e0 + k)
        riders.append(R.origin[R.status <= 1].to_numpy())
    dids = np.flatnonzero(np.any([s != 0 for s in st], 0))
    run = {"name": name, "sub": sub, "w0": t0, **frames(geo, xy, st, riders, dids)}
    fr = lambda t: np.floor((np.asarray(t, float) - t0) / S.DT).astype(int)  # noqa: E731
    o = pd.read_parquet(day_dir / "offers.parquet")
    rq = pd.read_parquet(day_dir / "requests.parquet").set_index("request_id")
    o = o[(o.t_sent >= t0) & (fr(o.t_sent) < ne)].sort_values("t_sent")
    run["off"] = {"e": fr(o.t_sent).tolist(), "d": o.driver_id.astype(int).tolist(),
                  "node": rq.origin_node.reindex(o.request_id).astype(int).tolist(),
                  "r": o.response.map(RESP).fillna(3).astype(int).tolist(), "eta": np.round(o.eta_est_s / 6).astype(int).tolist()}
    ev = L.ev[(L.ev.t_event >= t0) & (fr(L.ev.t_event) < ne)]
    ds = ev[ev.type == "dispatch_sent"].sort_values("t_event")
    ans = ev[ev.type.isin(["dispatch_accepted", "dispatch_declined"])]
    ok = []
    for _, r in ds.iterrows():
        a = ans[(ans.driver_id == r.driver_id) & (ans.t_event >= r.t_event) & (ans.t_event <= r.t_event + 180)]
        ok.append(2 if not len(a) else int(a.iloc[0].type == "dispatch_accepted"))
    run["disp"] = {"e": fr(ds.t_event).tolist(), "d": ds.driver_id.astype(int).tolist(), "node": ds.node.astype(int).tolist(),
                   "z": ds.zone.astype(int).tolist(), "ok": ok}
    dq = ev[ev.type == "quote_declined"].sort_values("t_event")
    run["dec"] = {"e": fr(dq.t_event).tolist(), "node": dq.node.astype(int).tolist()}
    E = lambda typ: fr(ev.t_event[ev.type.isin(typ)])  # noqa: E731
    run["kpi"] = {"pickups": cum(E(["pickup"]), ne), "cancels": cum(E(["customer_cancelled"]), ne),
                  "rejected": cum(E(["offer_rejected", "offer_timeout"]), ne), "declined": cum(E(["quote_declined"]), ne),
                  "moves": cum(E(["dispatch_sent"]), ne), "waiting": [int(len(r)) for r in riders],
                  "online": [int((s != 0).sum()) for s in st]}
    return run, L


def imagined_run(geo, arec, t0, name, sub):
    ne = FRAMES + 1
    st = list(arec["st"])
    xy = list(arec["pos"])
    dids = np.flatnonzero((arec["st"] != 0).any(0))
    run = {"name": name, "sub": sub, "w0": t0, **frames(geo, xy, st, arec["riders"], dids)}
    fr = lambda t: int(np.floor((t - t0) / S.DT))  # noqa: E731
    evs = arec["events"]
    resp = {}
    for x in evs:
        if x[1] in ("offer_accepted", "offer_rejected", "offer_timeout"):
            resp.setdefault((x[2], x[3]), ["offer_accepted", "offer_rejected", "offer_timeout"].index(x[1]))
    off = {"e": [], "d": [], "node": [], "r": [], "eta": []}
    disp = {"e": [], "d": [], "node": [], "z": [], "ok": []}
    dec = {"e": [], "node": []}
    dans = {}
    for x in evs:
        if x[1] in ("dispatch_accepted", "dispatch_declined"):
            dans.setdefault((x[2], x[4]), int(x[1] == "dispatch_accepted"))
    kp = {k: [] for k in ["pickups", "cancels", "rejected", "declined", "moves"]}
    for x in evs:
        e = min(max(fr(x[0]), 0), ne - 1)
        if x[1] == "offer_sent" and x[3] in arec["origin"]:
            off["e"].append(e); off["d"].append(int(x[2])); off["node"].append(arec["origin"][x[3]])
            off["r"].append(resp.get((x[2], x[3]), 3)); off["eta"].append(int(round(x[5] / 6)))
        elif x[1] == "dispatch_sent":
            disp["e"].append(e); disp["d"].append(int(x[2])); disp["node"].append(int(geo.znode[x[4]])); disp["z"].append(int(x[4]))
            disp["ok"].append(dans.get((x[2], x[4]), 2))
        elif x[1] == "quote_declined":
            dec["e"].append(e); dec["node"].append(int(geo.znode[x[4]]))
        k = {"pickup": "pickups", "customer_cancelled": "cancels", "offer_rejected": "rejected", "offer_timeout": "rejected",
             "quote_declined": "declined", "dispatch_sent": "moves"}.get(x[1])
        if k:
            kp[k].append(e)
    o = np.argsort(off["e"], kind="stable")
    run["off"] = {k: [v[i] for i in o] for k, v in off.items()}
    run["disp"], run["dec"] = disp, dec
    run["kpi"] = {k: cum(v, ne) for k, v in kp.items()}
    run["kpi"].update({"waiting": [int(len(r)) for r in arec["riders"]], "online": [int((s != 0).sum()) for s in st]})
    return run


def menus_pack(rounds, t0):
    ne = FRAMES + 1
    rs = sorted((r for r in rounds if t0 <= r["t"] < t0 + ne * S.DT), key=lambda r: r["t"])
    m = {"e": [], "d": [], "k": [], "node": [], "fare": [], "pick": [], "trip": [], "net": [], "a": [], "won": []}
    for r in rs:
        k = len(r["node"])
        m["e"].append(int((r["t"] - t0) // S.DT)); m["d"].append(r["d"]); m["k"].append(k); m["a"].append(r["a"]); m["won"].append(int(r["won"]))
        for f in ("node", "fare", "pick", "trip", "net"):
            m[f] += list(r[f]) + [0] * (4 - k)
    return m


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def page(data, title, intro, body, path):
    tpl = (Path(__file__).resolve().parents[1] / "world" / "wm3" / "report.html").read_text()
    html = (tpl.replace("/*TITLE*/", title).replace("/*INTRO*/", intro).replace("/*BODY*/", body)
            .replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    path.write_text(html)
    print(f"  wrote {path} ({len(html) / 1e6:.1f} MB)")


def build_all(keys):
    import importlib.util as iu
    spec = iu.spec_from_file_location("t38", Path(__file__).with_name("38_wm3_reports_text.py"))
    T = iu.module_from_spec(spec)
    spec.loader.exec_module(T)
    geos = {k: Geo(k) for k in keys}
    for stage in (6, 7, 8):
        cs = []
        for key in keys:
            geo, rec = geos[key], base_rec(key)
            date, t0 = rec["date"], float(rec["t0"])
            pairs = []
            if stage == 6:
                A, _ = real_run(geo, root(key) / f"seed{SEED}_rule" / date, t0, "Platform's rule", "closest driver first; rule moves idle drivers to busy areas")
                B, _ = real_run(geo, root(key) / f"seed{SEED}_mbc_pre+m" / date, t0, "World-model agent", "the agent picks the driver for every rider and orders moves")
                pairs.append({"label": "rule vs agent", "A": A, "B": B, "note": T.REPLAY_NOTE[6]})
            elif stage == 7:
                A, _ = real_run(geo, root(key) / f"seed{SEED}_mbc_pre+m" / date, t0, "Single offers", "one ride at a time; simulated people accept or reject")
                mz = pickle.loads((rep_dir(key) / "s7_menus.pkl").read_bytes())
                B, _ = real_run(geo, Path(mz["day_dir"]), t0, "Menus + driver agents", "up to 4 rides with fares; trained drivers choose")
                B["menus"] = menus_pack(mz["rounds"], t0)
                B.pop("off")
                pairs.append({"label": "single offers vs menus", "A": A, "B": B, "note": T.REPLAY_NOTE[7]})
            else:
                im = pickle.loads((rep_dir(key) / "s8_imagined.pkl").read_bytes())
                shock = json.loads((root(key) / f"shock_seed{SEED}.json").read_text())
                Lb = S.DayLog(geo.g, root(key) / f"seed{SEED}_rule" / date)
                xy0 = Lb.xy(int(t0 // S.DT))
                for arm in SHOCKS:
                    A, _ = real_run(geo, root(key) / f"seed{SEED}_{arm}" / date, t0, "Simulator (what really happens)", T.SHOCK_SUB[arm])
                    B = imagined_run(geo, im["arms"][arm], t0, "World model (imagined)", "from the same 12:00 state, nothing from the real future")
                    if arm.startswith("supply"):
                        out = np.array(shock["arms"][arm]["shock"]["removed"], int)
                        rm = {"x": geo.q(xy0[out, 0], geo.x0).tolist(), "y": geo.q(xy0[out, 1], geo.y0).tolist(), "until": 20}
                        for r in (A, B):
                            r["rm"] = rm
                            r["banner"] = f"12:00 shock: {len(out)} idle drivers ({arm[-2:]}%) logged off and stay off"
                            r["bannerUntil"] = 20
                    else:
                        for r in (A, B):
                            r["banner"] = "Prices held at x1.5 or more everywhere from 12:00"
                    pairs.append({"label": T.SHOCK_LABEL[arm], "A": A, "B": B, "note": T.REPLAY_NOTE[8] + T.SHOCK_LOOK[arm]})
            cur, dec = CUR[key]
            cs.append({"key": key, "name": CITY_NAME[key], "cur": cur, "dec": dec, **geo.pack(), "pairs": pairs})
            print(f"  stage {stage} {key} packed", flush=True)
        data = {"dt": S.DT, "unit": 2.0, "cities": cs, **T.PAGE[stage]["replay"]}
        intro, body = T.content(stage, png)
        page(data, T.PAGE[stage]["title"], intro, body, W3 / f"stage{stage}" / f"stage{stage}_report.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--city", choices=KEYS)
    args = ap.parse_args()
    keys = [args.city] if args.city else KEYS
    if args.collect:
        for key in keys:
            collect_s8(key)
            collect_s7(key)
    if args.build:
        build_all(keys)


if __name__ == "__main__":
    main()
