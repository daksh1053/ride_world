"""Stage 2a fix: rebuild the NEW (new request) arrays of the day files in float32.

The first build stored them as float16, which cannot hold node ids above 2048 exactly. The
records are recomputed from the requests table exactly as features.Featurizer does.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from world import cities
from world.wm3 import state as S


def fix(key, f):
    city = cities.get(key)
    nodes = pd.read_parquet(city.data_dir / "nodes.parquet")
    x, y = nodes.x_m.to_numpy(), nodes.y_m.to_numpy()
    z = dict(np.load(f))
    if z["NEW"].dtype == np.float32:
        return f.name, "already"
    day_dir = next((city.data_dir / "hd_runs").glob(f"wm2_seed*/{f.stem}"))
    q = pd.read_parquet(day_dir / "requests.parquet").sort_values("request_id")
    mode = q["mode"].map({m: i for i, m in enumerate(S.MODES)}).to_numpy()
    ep = np.ceil(q.t_request.to_numpy() / S.DT).astype(int) - 1
    o, d = q.origin_node.to_numpy(), q.dest_node.to_numpy()
    km = np.where(d >= 0, np.hypot(x[np.maximum(d, 0)] - x[o], y[np.maximum(d, 0)] - y[o]) / 1000, -1)
    money = float(z["money"])
    rows, off = [], [0]
    for e in z["epoch"]:
        i = np.flatnonzero(ep == e)
        rows.append(np.column_stack([q.origin_zone.to_numpy()[i], o[i], q.dest_zone.to_numpy()[i], d[i], q.external_dest.to_numpy()[i],
                                     mode[i], (q.status == "declined").to_numpy()[i], q.quote.to_numpy()[i] / money,
                                     q.surge.to_numpy()[i], q.eta_shown_s.to_numpy()[i] / 60, km[i]]).astype(np.float32))
        off.append(off[-1] + len(i))
    old = z["NEW"]
    z["NEW"], z["NEW_off"] = np.concatenate(rows), np.array(off, np.int64)
    same = len(old) == len(z["NEW"]) and np.allclose(old[:, 0], z["NEW"][:, 0])
    np.savez(f, **z)
    return f.name, f"{len(z['NEW'])} requests, counts match {same}"


def main():
    jobs = [(k, f) for k in ("sf_hd", "pune_hd") for f in sorted((cities.get(k).data_dir / "wm3").glob("20*.npz"))]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(fix, *zip(*jobs)))
    print(len(res), "files;", sum("True" in r for _, r in res), "with matching counts;", [r for r in res if "False" in r[1]][:5])


if __name__ == "__main__":
    main()
