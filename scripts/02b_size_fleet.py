"""Stage 2b: size each generated city's fleet so the baseline dispatcher completes
about TARGET of requests on a typical weekday.

    .venv/bin/python scripts/02b_size_fleet.py --city boulder annarbor ...

The demand level per city is an assumption scaled from population (world/cities.py);
fleet size is then chosen the way a platform would: the smallest fleet that serves
the target share. Writes data/<city>/fleet.json, which cities.get() applies.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world import cities, metrics
from world.policies import POLICIES
from world.setup import CityData, build_world
from world.sim import SimConfig

TARGET = 0.88
DATE = "2025-03-05"          # a Wednesday, dry in both hemispheres' climates used here


def completion(data, n):
    data.city = replace(data.city, market=replace(data.city.market, n_drivers=n))
    w = build_world(data, SimConfig(date=DATE, seed=5), POLICIES["nearest"](), log=lambda *a: None)
    w.run(log=lambda *a: None)
    return metrics.service_summary(w.tables())["completion_rate"]


def size(key):
    city = cities.CITIES[key]                # the template value, ignoring any old fleet.json
    data = CityData(city)
    n = city.market.n_drivers
    tried = {}
    for _ in range(6):
        tried[n] = completion(data, n)
        print(f"  {key}: {n} drivers -> {tried[n]:.3f}", flush=True)
        ok = [k for k, v in tried.items() if v >= TARGET]
        bad = [k for k, v in tried.items() if v < TARGET]
        if ok and bad and min(ok) <= 1.08 * max(bad):
            break
        if not ok and len(tried) >= 3 and max(tried.values()) - min(tried.values()) < 0.02:
            break                                   # plateau: more drivers do not help
        if not ok:
            n = int(n * (1.5 if tried[n] < TARGET - 0.1 else 1.2))
        elif not bad:
            n = max(20, int(n * 0.8))
        else:
            n = int((min(ok) + max(bad)) / 2)
    # Completion can plateau below TARGET (customers who give up on long ETAs or change
    # their mind cannot be served by more cabs): then take the smallest fleet within
    # 1 point of the best seen, instead of buying drivers for nothing.
    goal = min(TARGET, max(tried.values()) - 0.01)
    n = min(k for k, v in tried.items() if v >= goal)
    (city.data_dir / "fleet.json").write_text(json.dumps({"n_drivers": n, "target": TARGET,
                                                          "tried": tried, "date": DATE}, indent=2))
    print(f"== {key}: fleet {n} (completion {tried[n]:.3f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", required=True)
    for k in ap.parse_args().city:
        size(k)
