"""World model 3, stage 2a: the training data, one file per simulated day.

    .venv/bin/python scripts/17_wm3_dataset.py              # all seasons of both cities, in parallel

For every day of the step-2 seasons (data/<city>/hd_runs/wm2_seed*/<date>/), every 6th decision
epoch (every 3 minutes; the first epoch shifts with the day so all phases are covered) becomes
a snapshot: inputs and next-30-s targets for every online driver, offline driver, open request
and zone (world/wm3/features.py). Days of a season are processed in order so that each driver's
records from earlier days (Prior) are available, as a platform's database would hold them.
Writes data/<city>/wm3/<date>.npz and data/<city>/wm3/meta.json.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from world import cities
from world.wm3 import data as WD
from world.wm3 import features as FT
from world.wm3 import state as S

STRIDE = 6
CITIES = {"sf_hd": 0, "pune_hd": 1}


def run_season(key, season_dir):
    city = cities.get(key)
    g = S.CityGraph(city)
    zones = pd.read_parquet(city.data_dir / "zones.parquet")
    ztt = np.load(city.data_dir / "traffic" / "zone_tt_s.npy")
    out = city.data_dir / WD.SUBDIR
    out.mkdir(exist_ok=True)
    prior = FT.Prior()
    days = sorted(p for p in Path(season_dir).iterdir() if p.is_dir() and p.name.startswith("20"))
    log = []
    for i, d in enumerate(days):
        t0 = time.time()
        L = S.DayLog(g, d)
        f = out / f"{d.name}.npz"
        if not f.exists():
            F = FT.Featurizer(L, prior, CITIES[key], zones, ztt)
            ep = list(range(i % STRIDE, S.N_EPOCH, STRIDE))
            P = FT.pack([F.snapshot(e) for e in ep], ep)
            P["money"] = np.float32(F.money)
            np.savez(f, **P)
            log.append(f"{key} {d.parent.name} {d.name} {len(ep)} snapshots {f.stat().st_size / 1e6:.0f} MB {time.time() - t0:.0f}s")
        prior.update(L)
    return log


def main():
    jobs = [(k, s) for k in CITIES for s in sorted((cities.get(k).data_dir / "hd_runs").glob("wm2_seed*"))]
    with ProcessPoolExecutor(len(jobs)) as ex:
        for log in ex.map(run_season, *zip(*jobs)):
            print("\n".join(log[-3:]), flush=True)
    for k in CITIES:
        city = cities.get(k)
        meta = {"D_COLS": FT.D_COLS, "F_COLS": FT.F_COLS, "R_COLS": FT.R_COLS, "Z_COLS": FT.Z_COLS, "G_COLS": FT.G_COLS,
                "NEW_COLS": FT.NEW_COLS, "EV": FT.EV, "OFFER_RESP": FT.OFFER_RESP, "stride": STRIDE, "city_id": CITIES[k],
                "split": json.loads((city.data_dir / "wm2" / "split.json").read_text()),
                "days": sorted(p.stem for p in (city.data_dir / WD.SUBDIR).glob("20*.npz"))}
        (city.data_dir / WD.SUBDIR / "meta.json").write_text(json.dumps(meta, indent=1))
        print(k, len(meta["days"]), "days")


if __name__ == "__main__":
    main()
