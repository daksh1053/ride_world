"""Step 6: twin runs (scripts/15_pairs.py) as training data for the effect of an order.

A pair is one base morning and one counterfactual morning with an extra order, identical until
the order. Each is cut to the K frames before the first frame that differs (frame a) and the H
frames from it, so a rollout from the shared history sees each run's own actions.
Seeds ending in 9 are held out for validation.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from . import data as WD

FOLDER = "wm2_pairs"


class PairSet:
    def __init__(self, stores, K: int, H: int, device, dirs=True, which="train", seed=0):
        self.K, self.H, self.device = K, H, device
        self.rng = np.random.default_rng(seed)
        self.items, self.S = [], {s.id: torch.tensor(s.S, device=device) for s in stores}
        nch = len(WD.CH) + (WD.N_DIRS if dirs else 0)
        for s in stores:
            d = s.root.parent / FOLDER
            for rf in sorted(d.glob("rec_seed*.json")):
                rec = json.loads(rf.read_text())
                if (rec["seed"] % 10 == 9) != (which == "val"):
                    continue
                zb = np.load(d / f"seed{rec['seed']}_base.npz")
                xb, cb = zb["X"], zb["ctx"]
                for arm in rec["arms"][1:]:
                    zc = np.load(d / f"seed{rec['seed']}_{arm['arm']}.npz")
                    xc, cc = zc["X"], zc["ctx"]
                    diff = (xb[:, WD.ACT] != xc[:, WD.ACT]).reshape(len(xb), -1).any(1)
                    if not diff.any() or arm.get("n_moved", 0) == 0:
                        continue
                    a = int(np.argmax(diff))
                    lo, hi = a - K, a + H
                    if lo < 0 or hi > len(xb):
                        continue
                    self.items.append({"sid": s.id, "key": s.key, "seed": rec["seed"], **arm,
                                       "xb": xb[lo:hi, :nch], "xc": xc[lo:hi, :nch], "cb": cb[lo:hi], "cc": cc[lo:hi]})

    def __len__(self):
        return len(self.items)

    def gather(self, idx):
        """(2B, K+H, C, G, G) uint8 frames: the B base runs, then their B counterfactual twins."""
        its = [self.items[i] for i in idx]
        x = np.stack([it["xb"] for it in its] + [it["xc"] for it in its])
        c = np.stack([it["cb"] for it in its] + [it["cc"] for it in its])
        S = torch.stack([self.S[it["sid"]] for it in its] * 2)
        dev = self.device
        return torch.from_numpy(x).to(dev), torch.from_numpy(c).to(dev), S

    def sample(self, B):
        return self.gather(self.rng.integers(len(self.items), size=B))


def blocks(t, b=4):
    """Sum (..., G, G) over b x b blocks."""
    G = t.shape[-1]
    return t.reshape(*t.shape[:-2], G // b, b, G // b, b).sum((-3, -1))


def coupled_poisson(mb, mc):
    """Poisson counts for two runs with common random numbers: a shared draw for what both
    expect and separate draws for the excess, so the sampled difference is no noisier than
    the predicted difference requires."""
    m = torch.minimum(mb, mc)
    common = torch.poisson(m)
    return common + torch.poisson(mb - m), common + torch.poisson(mc - m)
