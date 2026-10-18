"""World model 3: the learned event model p_theta(E_{t+1} | X_t, A_t) of eq. 12.

Every online driver, open request and zone of a snapshot is a token, plus one global token
(time, weekday, rain, congestion). A transformer lets every token read every other one, with a
learned distance penalty per attention head (some heads look only nearby, some city-wide), so a
driver's decision can depend on the riders, drivers and roads around it. Heads:

  driver      own decision (EV, 8 classes); zone chosen when repositioning on its own (attention
              over the zone tokens); answer to an offer (3); answer to a repositioning order (2)
  offline     comes online in the next 30 s (an MLP on its record + the global token; offline
              drivers are many and only read the city as a whole)
  request     rider cancels; rider boards (driver waiting)
  zone        new requests per mode in the next 30 s (Poisson); for each new request: its origin
              node (learned node table + the zone's state), destination zone or gateway (attention
              over zones) and node, and whether the rider declines the quote

All heads are likelihoods, so the training loss is the negative log-likelihood of eq. 29
restricted to the stochastic events; deterministic rules (F) are not scored.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import features as FT

I_X, I_Y = FT.D_COLS.index("x"), FT.D_COLS.index("y")
NEW = {c: i for i, c in enumerate(FT.NEW_COLS)}


def fourier(xy, n=12):
    """(…, 2) positions in 10 km units -> (…, 4n) sin/cos at scales from ~60 m to ~40 km."""
    f = torch.exp(torch.linspace(math.log(0.15), math.log(100.0), n, device=xy.device))
    a = xy[..., None] * f * 2 * math.pi / 10
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


def bumps(x, lo, hi, step):
    """(…) values -> (…, k) Gaussian bumps centred lo..hi every step (lets a head be sharply peaked)."""
    c = torch.arange(lo, hi + 1e-6, step, device=x.device, dtype=x.dtype)
    return torch.exp(-0.5 * ((x[..., None] - c) / step) ** 2)


DI = {c: i for i, c in enumerate(FT.D_COLS)}
FI = {c: i for i, c in enumerate(FT.F_COLS)}


def time_feats_D(D):
    """Online drivers: hours from usual shift start, hours online, time in the current status."""
    return torch.cat([bumps(D[..., DI["shift_offset_h"]], -6, 14, 0.5), bumps(D[..., DI["online_h"]], 0, 14, 0.5),
                      bumps(D[..., DI["in_status_log"]], 0, 6, 0.5)], -1)


def time_feats_F(F):
    """Offline drivers: hours from usual shift start, hours online today, hours since going offline."""
    return torch.cat([bumps(F[..., FI["shift_offset_h"]] * 12, -6, 9, 0.25), bumps(F[..., FI["online_h_today"]] * 8, 0, 14, 0.5),
                      bumps(F[..., FI["hours_since_off"]] * 24, 0, 16, 0.5)], -1)


# zone columns that do not depend on where drivers are (demand in the simulator is exogenous;
# a demand head that sees supply learns "few idle drivers -> more requests", which is true when
# watching but wrong when the platform moves drivers)
SUPPLY_Z = {"pending", "idle_bike", "idle_auto", "idle_cab", "busy", "repo_to", "surge_bike", "surge_auto", "surge_cab"}
EXO_Z = [i for i, c in enumerate(FT.Z_COLS) if c not in SUPPLY_Z]

N_TF_D = 41 + 29 + 13
N_TF_F = 61 + 29 + 33


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bias):
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.o(a.transpose(1, 2).reshape(B, N, d))
        return x + self.ff(self.n2(x))


class WorldModel3(nn.Module):
    def __init__(self, tables: dict, d=128, layers=4, heads=4, node_dim=16, tf=False, exo=False):
        super().__init__()
        self.d, self.heads, self.tf, self.exo = d, heads, tf, exo       # tf: time bumps into the driver heads
        if exo:                                                           # exo: demand heads see no supply
            self.emb_Zx = mlp(len(EXO_Z) + 48 + len(FT.G_COLS), 2 * d, d)
            self.blocks_x = nn.ModuleList([Block(d, heads) for _ in range(2)])
            self.dist_wx = nn.Parameter(torch.log(torch.expm1(torch.tensor([0.02, 0.3, 1.0, 3.0]).repeat(heads // 4 + 1)[:heads])).repeat(2, 1))
            self.out_nx = nn.LayerNorm(d)
        nf = 48                                                         # fourier features of (x, y)
        self.emb_D = mlp(len(FT.D_COLS) + nf, 2 * d, d)
        self.emb_R = mlp(len(FT.R_COLS) + nf, 2 * d, d)
        self.emb_Z = mlp(len(FT.Z_COLS) + nf + len(FT.G_COLS), 2 * d, d)
        self.emb_G = mlp(len(FT.G_COLS), 2 * d, d)
        self.type_emb = nn.Parameter(torch.zeros(4, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        # distance penalty per head and layer (per km): from ~0 (city-wide) to strongly local
        init = torch.tensor([0.02, 0.3, 1.0, 3.0]).repeat(heads // 4 + 1)[:heads]
        self.dist_w = nn.Parameter(torch.log(torch.expm1(init)).repeat(layers, 1))
        self.out_n = nn.LayerNorm(d)
        self.h_ev = mlp(d + (N_TF_D if tf else 0), d, len(FT.EV))
        self.h_offer = mlp(d, d, 3)
        self.h_order = mlp(d, d, 2)
        self.q_repo, self.k_repo = nn.Linear(d, d), nn.Linear(d, d)
        self.h_online = mlp(len(FT.F_COLS) + nf + d + (N_TF_F if tf else 0), 2 * d, 1, depth=3)
        self.h_cancel = mlp(d, d, 1)
        self.h_board = mlp(d, d, 1)
        self.h_rate = mlp(d, d, 3)
        self.mode_emb = nn.Embedding(3, d)
        # per-city node tables for where requests start and end
        self.node_b = nn.ParameterDict({k: nn.Parameter(torch.zeros(t.N)) for k, t in tables.items()})
        self.node_e = nn.ParameterDict({k: nn.Parameter(torch.randn(t.N, node_dim) * 0.02) for k, t in tables.items()})
        self.node_b2 = nn.ParameterDict({k: nn.Parameter(torch.zeros(t.N)) for k, t in tables.items()})
        self.node_e2 = nn.ParameterDict({k: nn.Parameter(torch.randn(t.N, node_dim) * 0.02) for k, t in tables.items()})
        self.ext_e = nn.ParameterDict({k: nn.Parameter(torch.randn(t.n_ext, d) * 0.02) for k, t in tables.items()})
        self.q_origin, self.q_dnode = nn.Linear(2 * d, node_dim), nn.Linear(2 * d, node_dim)
        self.q_dest, self.k_dest = nn.Linear(2 * d, d), nn.Linear(d, d)
        self.dest_dist = nn.Parameter(torch.tensor(0.5))
        self.h_decl = mlp(2 * d + 6, d, 1)
        self.tables = tables

    # ------------------------------------------------------------------ encoder
    def encode(self, b):
        """Token states after the transformer: (drivers, requests, zones, global) parts."""
        D, R, Z, G = b["D"], b["R"], b["Z"], b["G"]
        B = D.shape[0]
        dxy, rxy, zxy = D[..., [I_X, I_Y]], R[..., [4, 5]], Z[..., [0, 1]]
        hD = self.emb_D(torch.cat([D, fourier(dxy)], -1)) + self.type_emb[0]
        hR = self.emb_R(torch.cat([R, fourier(rxy)], -1)) + self.type_emb[1]
        hZ = self.emb_Z(torch.cat([Z, fourier(zxy), G[:, None].expand(-1, Z.shape[1], -1)], -1)) + self.type_emb[2]
        hG = self.emb_G(G)[:, None] + self.type_emb[3]
        x = torch.cat([hD, hR, hZ, hG], 1)
        xy = torch.cat([dxy, rxy, zxy, torch.zeros(B, 1, 2, device=D.device)], 1)
        mask = torch.cat([b["D_mask"], b["R_mask"], b["Z_mask"], torch.ones(B, 1, dtype=torch.bool, device=D.device)], 1)
        dist = torch.cdist(xy, xy) * 10.0                                   # km
        dist[:, -1, :] = 0
        dist[:, :, -1] = 0                                                  # the global token has no place
        pad = torch.zeros_like(dist).masked_fill(~mask[:, None, :], float("-inf"))
        for li, blk in enumerate(self.blocks):
            w = Fn.softplus(self.dist_w[li])
            bias = pad[:, None] - w[None, :, None, None] * dist[:, None]
            x = blk(x, bias)
        x = self.out_n(x)
        nD, nR, nZ = D.shape[1], R.shape[1], Z.shape[1]
        return x[:, :nD], x[:, nD:nD + nR], x[:, nD + nR:nD + nR + nZ], x[:, -1]

    def encode_demand(self, b):
        """Zone states for the demand heads from exogenous inputs only (place, time, weather,
        congestion, recent requests), zones attending to zones."""
        Z, G = b["Z"], b["G"]
        zxy = Z[..., [0, 1]]
        x = self.emb_Zx(torch.cat([Z[..., EXO_Z], fourier(zxy), G[:, None].expand(-1, Z.shape[1], -1)], -1))
        dist = torch.cdist(zxy, zxy) * 10.0
        pad = torch.zeros_like(dist).masked_fill(~b["Z_mask"][:, None, :], float("-inf"))
        for li, blk in enumerate(self.blocks_x):
            w = Fn.softplus(self.dist_wx[li])
            x = blk(x, pad[:, None] - w[None, :, None, None] * dist[:, None])
        return self.out_nx(x)

    # ------------------------------------------------------------------ heads
    def forward(self, b):
        hD, hR, hZ, hG = self.encode(b)
        out = {"ev": self.h_ev(torch.cat([hD, time_feats_D(b["D"]).to(hD.dtype)], -1) if self.tf else hD), "offer": self.h_offer(hD), "order": self.h_order(hD),
               "repo": torch.einsum("bnd,bzd->bnz", self.q_repo(hD), self.k_repo(hZ)) / math.sqrt(self.d),
               "cancel": self.h_cancel(hR)[..., 0], "board": self.h_board(hR)[..., 0],
               "rate": None}
        hZd = self.encode_demand(b) if self.exo else hZ
        out["rate"] = self.h_rate(hZd)
        out["hZd"] = hZd
        out["repo"] = out["repo"].masked_fill(~b["Z_mask"][:, None, :], float("-inf"))
        F = b["F"]
        fxy = F[..., [5, 6]]
        parts = [F, fourier(fxy), hG[:, None].expand(-1, F.shape[1], -1)] + ([time_feats_F(F)] if self.tf else [])
        out["online"] = self.h_online(torch.cat([x.to(hG.dtype) for x in parts], -1))[..., 0]
        out["h"] = (hD, hR, hZ, hG)
        return out

    def new_request_logits(self, key, hZ_b, zone, mode, origin=None, dest_zone=None):
        """For new requests in one snapshot (zone, mode: (n,) long), logits of
        origin node within the zone, destination (zones then gateways) and destination node
        within the destination zone. Returns (origin_logits (n, maxnodes), candidates,
        dest_logits (n, Z + n_ext), dnode_logits, dcand)."""
        t = self.tables[key]
        hz = hZ_b[zone]
        ctx = torch.cat([hz, self.mode_emb(mode)], -1)
        cand = t.zone_nodes[zone]                                            # (n, m)
        cn = cand.clamp(min=0)
        ol = self.node_b[key][cn] + torch.einsum("nmk,nk->nm", self.node_e[key][cn], self.q_origin(ctx))
        ol = ol.masked_fill(cand < 0, float("-inf"))
        # destination: attention over zone tokens plus gateway embeddings, with a distance term
        keys = torch.cat([self.k_dest(hZ_b), self.k_dest(self.ext_e[key])], 0)
        dl = self.q_dest(ctx) @ keys.T / math.sqrt(self.d)
        if origin is not None:
            oxy = t.node_xy[origin]
            dd = torch.cdist(oxy, t.zone_xy) * 10
            dl[:, :t.Z] = dl[:, :t.Z] - Fn.softplus(self.dest_dist) * torch.log1p(dd)
        dn = dnl = None
        if dest_zone is not None:
            dz = dest_zone.clamp(min=0)
            ctx2 = torch.cat([hZ_b[dz], self.mode_emb(mode)], -1)
            dn = t.zone_nodes[dz]
            dnc = dn.clamp(min=0)
            dnl = self.node_b2[key][dnc] + torch.einsum("nmk,nk->nm", self.node_e2[key][dnc], self.q_dnode(ctx2))
            dnl = dnl.masked_fill(dn < 0, float("-inf"))
        return ol, cand, dl, dnl, dn

    def decline_logit(self, hZ_b, zone, mode, feats):
        """feats (n, 6): quote, surge, shown ETA (min), trip km, external, 1."""
        return self.h_decl(torch.cat([hZ_b[zone], self.mode_emb(mode), feats], -1))[..., 0]


def losses(model, b, out):
    """Mean negative log-likelihood per head (each over its own instances)."""
    L = {}
    m = b["D_mask"]
    L["driver_event"] = Fn.cross_entropy(out["ev"][m], b["D_ev"][m])
    o = b["D_offer"] >= 0
    L["offer_answer"] = Fn.cross_entropy(out["offer"][o], b["D_offer"][o]) if o.any() else None
    o = b["D_order"] >= 0
    L["order_answer"] = Fn.cross_entropy(out["order"][o], b["D_order"][o]) if o.any() else None
    o = b["D_repo"] >= 0
    L["own_reposition_zone"] = Fn.cross_entropy(out["repo"][o], b["D_repo"][o]) if o.any() else None
    m = b["F_mask"]
    L["comes_online"] = Fn.binary_cross_entropy_with_logits(out["online"][m], b["F_on"][m].float())
    m = b["R_mask"]
    L["rider_cancels"] = Fn.binary_cross_entropy_with_logits(out["cancel"][m], b["R_cancel"][m].float()) if m.any() else None
    m = b["R_mask"] & (b["R"][..., 3] > 0.5)                                # driver waiting at the pickup
    L["rider_boards"] = Fn.binary_cross_entropy_with_logits(out["board"][m], b["R_board"][m].float()) if m.any() else None
    m = b["Z_mask"]
    lr = out["rate"].clamp(-12, 6)
    y = b["Z_new"].float()
    L["new_requests"] = ((lr.exp() - y * lr + torch.lgamma(y + 1))[m]).sum(-1).mean()
    # each new request: origin node, destination, destination node, quote declined
    hZ = out["h"][2]
    hZd = out.get("hZd", hZ)                                           # demand states (exogenous if exo)
    lo, ld, ldn, ldc, n = 0.0, 0.0, 0.0, 0.0, 0
    for bi, key in enumerate(b["city"]):
        nw = b["NEW"][bi][b["NEW_mask"][bi]]
        if not len(nw):
            continue
        zone, mode = nw[:, NEW["zone"]].long(), nw[:, NEW["mode"]].long()
        origin = nw[:, NEW["origin"]].long()
        ext = nw[:, NEW["external"]].long()
        dz = nw[:, NEW["dest_zone"]].long()
        dnode = nw[:, NEW["dest_node"]].long()
        Zb = int(b["Z_mask"][bi].sum())
        ol, cand, dl, dnl, dn = model.new_request_logits(key, hZd[bi, :Zb], zone, mode, origin, dz)
        lo = lo + Fn.cross_entropy(ol, (cand == origin[:, None]).float().argmax(1), reduction="sum")
        dtarget = torch.where(ext >= 0, Zb + ext.clamp(min=0), dz.clamp(min=0))
        ld = ld + Fn.cross_entropy(dl, dtarget, reduction="sum")
        inside = ext < 0
        if inside.any():
            ldn = ldn + Fn.cross_entropy(dnl[inside], (dn[inside] == dnode[inside, None]).float().argmax(1), reduction="sum")
        # quote: the pre-surge quote (features.py)
        feats = torch.stack([nw[:, NEW["quote"]], nw[:, NEW["surge"]] - 1, nw[:, NEW["eta_min"]] / 10,
                             nw[:, NEW["trip_km"]].clamp(min=0) / 10, (ext >= 0).float(), torch.ones_like(zone, dtype=torch.float32)], 1)
        ldc = ldc + Fn.binary_cross_entropy_with_logits(model.decline_logit(hZ[bi, :Zb], zone, mode, feats),
                                                        nw[:, NEW["declined"]], reduction="sum")
        n += len(nw)
    if n:
        L["request_origin_node"] = lo / n
        L["request_destination"] = ld / n
        L["request_dest_node"] = ldn / n
        L["quote_declined"] = ldc / n
    return {k: v for k, v in L.items() if v is not None}
