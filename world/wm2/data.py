"""World model v2 data: the per-day rasters (world/hd/raster.py) packed into one memory-
mapped byte array per city, and batches of consecutive frames for training.

Store layout (data/<city>/wm2/store/):
    X.u8      (D, T, C, G, G)  uint8   counts as they are; surge as (m - 1) x 100;
                                       GPS speed in km/h
    ctx.npy   (D, T, K)        float32 context per step
    cong.u8   (D, T, G, G)     uint8   hidden congestion as (ratio - 1) x 100 (probes only)
    static.npy (CS, G, G)      float32
    days.json  dates, shapes

A training sample is K + 1 consecutive frames. Input frame k holds the observation at
step k, the platform's actions during (k, k+1] and the context of k+1; the target is the
observation at k+1. All K positions are trained at once (causal in time).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..hd import raster as R

CH = list(R.CHANNELS)
ACT = [CH.index(c) for c in R.ACTION_CHANNELS]
OBS = [i for i in range(len(CH)) if i not in ACT]                    # 20 observed channels
SURGE = [CH.index(f"surge_{m}") for m in R.MODES3]
SPEED, SAMPLES = CH.index("gps_speed_kmh"), CH.index("gps_samples")
COUNT = [i for i in OBS if i not in SURGE and i != SPEED]             # 16 Poisson targets
COUNT_NAMES = [CH[i] for i in COUNT]
N_DIRS = len(R.DIR_CHANNELS)
DIRS = False          # step 4: append the 8 dispatch-direction channels (store/dirs.npy) to every frame


def build_dirs(city, log=print) -> Path:
    """Step 4 backfill: the dispatch-direction channels of every stored day, recomputed from
    the saved event tables (data/<city>/hd_runs/wm2_seed*/<date>/events.parquet)."""
    s = city.data_dir / "wm2" / "store"
    info = json.loads((s / "days.json").read_text())
    D, T, _, G, _ = info["shape"]
    cr = R.CityRaster(city)
    out = np.lib.format.open_memmap(s / "dirs.npy", "w+", np.uint8, (D, T, N_DIRS, G, G))
    runs = city.data_dir / "hd_runs"
    for i, date in enumerate(info["dates"]):
        day_dir = next(runs.glob(f"wm2_seed*/{date}"))
        out[i] = cr.dispatch_dirs(day_dir)[:T]
    out.flush()
    log(f"  dirs {city.key}: {D} days")
    return s / "dirs.npy"


def build_store(city, log=print) -> Path:
    d = city.data_dir / "wm2"
    out = d / "store"
    out.mkdir(exist_ok=True)
    files = sorted(p for p in d.glob("20*.npz"))
    X0, C0, Y0, _ = R.load_day(files[0])
    T, C, G, _ = X0.shape
    D = len(files)
    X = np.lib.format.open_memmap(out / "X.npy", "w+", np.uint8, (D, T, C, G, G))
    cong = np.lib.format.open_memmap(out / "cong.npy", "w+", np.uint8, (D, T, G, G))
    ctx = np.zeros((D, T, C0.shape[1]), np.float32)
    dates, metas = [], []
    for i, f in enumerate(files):
        x, c, y, meta = R.load_day(f)
        x[:, SURGE] *= 100
        X[i] = np.clip(np.round(x), 0, 255).astype(np.uint8)
        cong[i] = np.clip(np.round((y - 1) * 100), 0, 255).astype(np.uint8)
        ctx[i] = c
        dates.append(f.stem)
        metas.append(meta)
    X.flush(); cong.flush()
    np.save(out / "ctx.npy", ctx)
    np.save(out / "static.npy", np.load(d / "static.npz")["S"].astype(np.float32))
    (out / "days.json").write_text(json.dumps({"dates": dates, "meta": metas, "shape": [D, T, C, G, G]}))
    log(f"  store {city.key}: {D} days, {X.nbytes / 1e9:.2f} GB")
    return out


class CityStore:
    def __init__(self, city, key_id: int):
        s = city.data_dir / "wm2" / "store"
        self.key, self.id = city.key, key_id
        self.root, self.store_dir = city.data_dir / "wm2", s
        self.X = np.load(s / "X.npy", mmap_mode="r")
        self.cong = np.load(s / "cong.npy", mmap_mode="r")
        self.ctx = np.load(s / "ctx.npy")
        self.S = np.load(s / "static.npy")
        info = json.loads((s / "days.json").read_text())
        self.dates, self.meta = info["dates"], info["meta"]
        self.D, self.T = self.X.shape[:2]
        self.dirs = np.load(s / "dirs.npy", mmap_mode="r") if (s / "dirs.npy").exists() else None

    def act_peak(self):
        return _act_peak(self)

    def split(self, val=0.1, test=0.15):
        """Chronological: train on the first days, validate, test on the last. Since step 4
        the boundaries are fixed dates (data/<city>/wm2/split.json) so that days added later
        never move earlier days between the sets."""
        f = self.root / "split.json"
        if f.exists():
            sp = json.loads(f.read_text())
            dts = np.array(self.dates)
            idx = np.arange(self.D)
            return (idx[dts < sp["val_from"]], idx[(dts >= sp["val_from"]) & (dts <= sp["val_until"])],
                    idx[(dts >= sp["test_from"]) & (dts <= sp["test_until"])])
        n_te, n_va = max(1, round(self.D * test)), max(1, round(self.D * val))
        idx = np.arange(self.D)
        return idx[: self.D - n_te - n_va], idx[self.D - n_te - n_va: self.D - n_te], idx[self.D - n_te:]


def _act_peak(store):
    """(D, T) largest number of orders issued from one 4 x 4 block of cells in each 5-minute
    step, cached in the store folder (step 4d)."""
    f = store.store_dir / "act_peak.npy"
    if f.exists() and np.load(f, mmap_mode="r").shape[0] == store.D:
        return np.load(f)
    ch = CH.index("dispatch_from")
    pk = np.zeros((store.D, store.T), np.uint16)
    for d in range(store.D):
        a = np.asarray(store.X[d, :, ch], np.uint16)
        pk[d] = a.reshape(store.T, R.G // 4, 4, R.G // 4, 4).sum((2, 4)).max((1, 2))
    np.save(f, pk)
    return pk


class Batcher:
    """Random windows from several cities' stores; returns uint8 tensors on the device."""

    def __init__(self, stores, days: dict, K: int, device, seed=0, dirs=None, act_focus=0.0, focus_min=8):
        """`act_focus` (step 4d): share of windows drawn so that one of their target frames holds
        a large repositioning order (>= `focus_min` orders from one 4 x 4 block in 5 minutes)."""
        self.stores, self.days, self.K, self.device = stores, days, K, device
        self.use_dirs = DIRS if dirs is None else dirs
        self.rng = np.random.default_rng(seed)
        self.S = {s.id: torch.tensor(s.S, device=device) for s in stores}
        self.act_focus = act_focus
        if act_focus:
            self.focus = {}
            for s in stores:
                pk = s.act_peak()
                d, t = np.nonzero(pk >= focus_min)
                keep = np.isin(d, days.get(s.key, [])) & (t >= K) & (t < s.T)
                self.focus[s.key] = np.stack([d[keep], t[keep]], 1)

    def sample(self, B):
        out = []
        for _ in range(B):
            s = self.stores[self.rng.integers(len(self.stores))]
            if self.act_focus and len(self.focus[s.key]) and self.rng.random() < self.act_focus:
                d, ts = self.focus[s.key][self.rng.integers(len(self.focus[s.key]))]
                t = int(np.clip(ts - 1 - self.rng.integers(self.K), 0, s.T - self.K - 1))   # ts is a target frame
            else:
                d = self.rng.choice(self.days[s.key])
                t = self.rng.integers(0, s.T - self.K)
            out.append((s, d, t))
        return self.gather(out)

    def gather(self, items):
        K = self.K
        x = np.stack([s.X[d, t: t + K + 1] for s, d, t in items])            # (B, K+1, C, G, G)
        if self.use_dirs:                                                      # + 8 direction channels
            x = np.concatenate([x, np.stack([s.dirs[d, t: t + K + 1] for s, d, t in items])], 2)
        c = np.stack([s.ctx[d, t: t + K + 1] for s, d, t in items])
        y = np.stack([s.cong[d, t: t + K + 1] for s, d, t in items])
        S = torch.stack([self.S[s.id] for s, _, _ in items])
        dev = self.device
        return (torch.from_numpy(x).to(dev, non_blocking=True), torch.from_numpy(c).to(dev),
                S, torch.from_numpy(y).to(dev), [(s.id, int(d), int(t)) for s, d, t in items])


def make_io(x_u8, ctx, S):
    """uint8 frames -> model inputs and targets.

    x_u8 (B, K+1, C, G, G); returns inp (B, K, Cin, G, G), ctx_next (B, K, 6), and the
    target frames (B, K, C, G, G) as float (raw units: counts, surge x100, km/h)."""
    x = x_u8.float()
    B, K1, C, G, _ = x.shape
    obs = x[:, :-1][:, :, OBS]
    feats = obs.clone()
    j_surge = [OBS.index(i) for i in SURGE]
    j_speed = OBS.index(SPEED)
    cnt = [OBS.index(i) for i in COUNT]
    feats[:, :, cnt] = torch.log1p(feats[:, :, cnt])
    feats[:, :, j_surge] = feats[:, :, j_surge] / 100.0
    feats[:, :, j_speed] = feats[:, :, j_speed] / 60.0
    act_idx = ACT + list(range(len(CH), C))                                  # + direction channels if present
    act = torch.log1p(x[:, 1:][:, :, act_idx])                               # actions during (k, k+1]
    stat = S[:, None].expand(B, K1 - 1, *S.shape[1:])
    inp = torch.cat([feats, act, stat], 2)
    return inp, ctx[:, 1:], x[:, 1:]
