"""World model 3, stage 2d: the model's one-step predictions on the street map.

    .venv/bin/python scripts/21_wm3_pred_page.py --tag wm3_base

For one test day per city, 07:00-09:00, every 30-s epoch: the stage-1 viewer (every driver,
rider and road) plus what world model 3 predicts at that moment from the real state:
  - each driver's chances of each own decision in the next 30 s (and of accepting an offer /
    following an order when one is made), colourable on the map as the chance within 10 min
  - each open request's chance that the rider cancels / boards
  - predicted new requests over the next 10 min at every road node (zone rate x the learned
    origin-node distribution), with the requests that really came as squares
Click a driver or rider for its predicted probabilities next to what actually happened.
Writes outputs/wm3/stage2/<city>_predictions.html.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from world import cities
from world.wm3 import data as WD
from world.wm3 import features as FT
from world.wm3 import state as S


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TR = _load("tr3", "18_wm3_train.py")
V = _load("v16", "16_wm3_world.py")
RL = _load("r20", "20_wm3_rollout.py")
DEV = TR.DEV
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()  # noqa: E731


@torch.no_grad()
def predictions(model, key, L, g, prior, ids, rq, e0, e1):
    city = cities.get(key)
    zones = pd.read_parquet(city.data_dir / "zones.parquet")
    Fz = FT.Featurizer(L, prior, WD.CITY_KEYS.index(key), zones, np.load(city.data_dir / "traffic" / "zone_tt_s.npy"))
    tab = model.tables[key]
    ne = e1 - e0 + 1
    pos = pd.Series(np.arange(len(ids)), index=ids)
    rpos = pd.Series(np.arange(len(rq)), index=rq)
    ev = np.zeros((len(ids), ne, len(FT.EV)), np.uint16)
    lab = np.full((len(ids), ne), -1, np.int8)
    offer = np.full((len(ids), ne), 65535, np.uint16)
    order = np.full((len(ids), ne), 65535, np.uint16)
    req_rows, dem = [], []
    for k, e in enumerate(range(e0, e1 + 1)):
        s = Fz.snapshot(e)
        snap = {"key": key, **s}
        b = WD.collate([snap], DEV)
        with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
            out = model(b)
        j = pos.reindex(s["D_id"]).to_numpy()
        ok = ~np.isnan(j)
        j = j[ok].astype(int)
        pe = out["ev"][0].float().softmax(-1).cpu().numpy()[ok]
        ev[j, k] = np.round(pe * 65535)
        lab[j, k] = s["D_ev"][ok]
        po = out["offer"][0].float().softmax(-1)[:, 0].cpu().numpy()[ok]
        pr = out["order"][0].float().softmax(-1)[:, 0].cpu().numpy()[ok]
        has_o, has_r = s["D"][ok, FT.D_COLS.index("offer")] > 0.5, s["D"][ok, FT.D_COLS.index("order")] > 0.5
        offer[j[has_o], k] = np.round(po[has_o] * 65534)
        order[j[has_r], k] = np.round(pr[has_r] * 65534)
        rj = rpos.reindex(s["R_id"]).to_numpy()
        pc = torch.sigmoid(out["cancel"][0].float()).cpu().numpy()
        pb = torch.sigmoid(out["board"][0].float()).cpu().numpy()
        for i in np.flatnonzero(~np.isnan(rj)):
            req_rows.append((k, int(rj[i]), int(round(pc[i] * 65535)), int(round(pb[i] * 65535))))
        # expected new requests per node in the next 10 min (20 steps at the current rate)
        lam = out["rate"][0].float().clamp(-12, 6).exp()                         # (Z, 3)
        hZ = out["h"][2][0].float()
        node_val = torch.zeros(tab.N, device=hZ.device)
        for m in range(3):
            if lam[:, m].sum() < 1e-4:
                continue
            zt = torch.arange(tab.Z, device=hZ.device)
            hZd = out.get("hZd", out["h"][2])[0].float()
            ol, cand, *_ = model.new_request_logits(key, hZd[:tab.Z], zt, torch.full_like(zt, m))
            p = ol.float().softmax(-1) * lam[:tab.Z, m:m + 1] * 20
            node_val.index_add_(0, cand.clamp(min=0).flatten(), (p * (cand >= 0)).flatten())
        v = node_val.cpu().numpy()
        nn = np.flatnonzero(v >= 0.02)
        dem.append((nn.astype(np.uint32), v[nn].astype(np.float32)))
    R = np.array(req_rows, np.int64).reshape(-1, 4)
    return {"ev": b64(ev), "labels_ev": b64(lab), "offer": b64(offer), "order": b64(order),
            "req_e": b64(R[:, 0].astype(np.int16)), "req_j": b64(R[:, 1].astype(np.int32)),
            "req_cancel": b64(R[:, 2].astype(np.uint16)), "req_board": b64(R[:, 3].astype(np.uint16)),
            "dem_off": b64(np.r_[0, np.cumsum([len(a) for a, _ in dem])].astype(np.int32)),
            "dem_node": b64(np.concatenate([a for a, _ in dem])), "dem_val": b64(np.concatenate([b for _, b in dem])),
            "ev_names": FT.EV}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wm3_base")
    args = ap.parse_args()
    model, _ = TR.load(args.tag)
    model.eval()
    V.W0, V.W1 = 7 * 3600, 9 * 3600
    out = TR.OUT
    for key in WD.CITY_KEYS:
        city = cities.get(key)
        g = S.CityGraph(city)
        meta = json.loads((city.data_dir / WD.SUBDIR / "meta.json").read_text())
        sp = meta["split"]
        test = [d for d in meta["days"] if sp["test_from"] <= d <= sp["test_until"] and pd.Timestamp(d).weekday() < 5]
        day_dir = next((city.data_dir / "hd_runs").glob(f"wm2_seed*/{test[2]}"))
        L = S.DayLog(g, day_dir)
        prior = RL.season_prior(day_dir)
        data = V.viewer_data(key, L, g)
        ids = np.frombuffer(base64.b64decode(data["drivers"]["ids"]), np.int32)
        rq = np.frombuffer(base64.b64decode(data["requests"]["id"]), np.int32)
        e0, e1 = int(V.W0 / S.DT), int(V.W1 / S.DT)
        data["pred"] = predictions(model, key, L, g, prior, ids, rq, e0, e1)
        tpl = (Path(__file__).resolve().parents[1] / "world" / "wm3" / "viewer.html").read_text()
        html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
        html = html.replace("<title>World State Viewer</title>", "<title>Model Predictions Viewer</title>")
        f = out / f"{key}_predictions.html"
        f.write_text(html)
        print(f"  {key} {test[2]}: {f} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
