"""World model 3: batches of snapshots from the per-day files of scripts/17_wm3_dataset.py.

A batch is B snapshots (any days, either city). Each entity type is padded to the largest count
in the batch; masks mark real rows. Static per-city tables (node positions and zones, zone
node lists, gateways) live in CityTables.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from .. import cities
from . import features as FT

CITY_KEYS = ["sf_hd", "pune_hd"]
# dataset folder under data/<city>/ (wm3v2: legs cut short restored, pre-surge quotes; see STEPS.md)
SUBDIR = os.environ.get("WM3_DATA", "wm3v2")
I_X, I_Y = FT.D_COLS.index("x"), FT.D_COLS.index("y")


class CityTables:
    """Per-city static tables on the device: nodes (position, zone), zones' node lists, gateways."""

    def __init__(self, key, device):
        city = cities.get(key)
        nodes = pd.read_parquet(city.data_dir / "nodes.parquet")
        zones = pd.read_parquet(city.data_dir / "zones.parquet")
        cx, cy = float(np.median(nodes.x_m)), float(np.median(nodes.y_m))
        self.key, self.N, self.Z = key, len(nodes), len(zones)
        self.node_zone = torch.tensor(nodes.zone.to_numpy(), device=device)
        self.node_xy = torch.tensor(np.c_[(nodes.x_m - cx) / FT.SCALE, (nodes.y_m - cy) / FT.SCALE], dtype=torch.float32, device=device)
        # nodes of each zone, padded (-1)
        groups = [np.flatnonzero(nodes.zone.to_numpy() == z) for z in range(self.Z)]
        m = max(len(x) for x in groups)
        zn = np.full((self.Z, m), -1, np.int64)
        for z, x in enumerate(groups):
            zn[z, :len(x)] = x
        self.zone_nodes = torch.tensor(zn, device=device)
        self.zone_xy = torch.tensor(np.c_[(zones.x_m - cx) / FT.SCALE, (zones.y_m - cy) / FT.SCALE], dtype=torch.float32, device=device)
        self.n_ext = len(city.market.external)
        # which nodes ever serve as an origin / destination (the simulator never uses freeways):
        # learned through the node bias, so no mask is imposed here


class DayPool:
    """Keeps `n_days` day files in memory and draws random snapshots; one day is swapped out
    every `swap` draws, so over training every day of the split is used."""

    def __init__(self, split="train", n_days=10, swap=40, seed=0, keys=CITY_KEYS, days=None):
        self.rng = np.random.default_rng(seed)
        self.files = []
        for k in keys:
            c = cities.get(k)
            meta = json.loads((c.data_dir / SUBDIR / "meta.json").read_text())
            sp = meta["split"]
            for d in meta["days"]:
                ok = {"train": d < sp["val_from"], "val": sp["val_from"] <= d <= sp["val_until"],
                      "test": sp["test_from"] <= d <= sp["test_until"]}[split]
                if ok and (days is None or d in days):
                    self.files.append((k, c.data_dir / SUBDIR / f"{d}.npz"))
        self.n_days, self.swap, self.draws = min(n_days, len(self.files)), swap, 0
        self.pool = [self._load(i) for i in self.rng.choice(len(self.files), self.n_days, replace=False)]

    def _load(self, i):
        k, f = self.files[i]
        z = np.load(f)
        return {"key": k, "date": f.stem, **{n: z[n] for n in z.files}}

    def sample(self):
        self.draws += 1
        if self.draws % self.swap == 0 and len(self.files) > self.n_days:
            self.pool[self.rng.integers(self.n_days)] = self._load(self.rng.integers(len(self.files)))
        day = self.pool[self.rng.integers(self.n_days)]
        return day, int(self.rng.integers(len(day["epoch"])))


def snapshot(day, i):
    """Snapshot i of a loaded day as a dict of numpy arrays."""
    out = {"key": day["key"], "date": day["date"], "epoch": int(day["epoch"][i]), "G": day["G"][i], "Z": day["Z"][i],
           "Z_new": day["Z_new"][i], "money": float(day["money"])}
    for k in ("D", "D_id", "D_ev", "D_repo", "D_offer", "D_order", "F", "F_id", "F_on", "R", "R_id", "R_cancel", "R_board", "NEW"):
        o = day[f"{k}_off"]
        out[k] = day[k][o[i]: o[i + 1]]
    return out


def collate(snaps, device):
    """Pad a list of snapshots into tensors."""
    B = len(snaps)

    def pad(key, dtype=torch.float32, fill=0):
        n = max(max(len(s[key]) for s in snaps), 1)
        shp = snaps[0][key].shape[1:]
        a = np.full((B, n) + shp, fill, dtype=np.float32 if dtype == torch.float32 else np.int64)
        m = np.zeros((B, n), bool)
        for b, s in enumerate(snaps):
            a[b, :len(s[key])] = s[key]
            m[b, :len(s[key])] = True
        return torch.tensor(a, dtype=dtype, device=device), torch.tensor(m, device=device)

    t = {}
    t["D"], t["D_mask"] = pad("D")
    for k in ("D_ev", "D_repo", "D_offer", "D_order"):
        t[k] = pad(k, torch.long, -1 if k != "D_ev" else 0)[0]
    t["F"], t["F_mask"] = pad("F")
    t["F_on"] = pad("F_on", torch.long)[0]
    t["R"], t["R_mask"] = pad("R")
    t["R_cancel"] = pad("R_cancel", torch.long)[0]
    t["R_board"] = pad("R_board", torch.long)[0]
    t["Z"], t["Z_mask"] = pad("Z")
    t["Z_new"] = pad("Z_new", torch.long)[0]
    t["NEW"], t["NEW_mask"] = pad("NEW")
    t["G"] = torch.tensor(np.stack([s["G"] for s in snaps]), dtype=torch.float32, device=device)
    t["city"] = [s["key"] for s in snaps]
    return t
