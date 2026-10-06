"""Static PNG rendering shared by every stage.

Colour follows the dataviz reference palette: road hierarchy and magnitudes are
ordinal/sequential, so they use one blue ramp (light = minor, dark = major) plus line
width, never a categorical rainbow.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

# road class -> (colour, linewidth, z-order); minor roads recede to a hairline grey.
ROAD_STYLE = {
    "motorway": ("#0d366b", 1.6, 9),
    "trunk": ("#184f95", 1.3, 8),
    "primary": ("#256abf", 1.0, 7),
    "secondary": ("#3987e5", 0.8, 6),
    "tertiary": ("#6da7ec", 0.6, 5),
    "residential": ("#c3c2b7", 0.3, 3),
    "unclassified": ("#c3c2b7", 0.3, 3),
    "living_street": ("#d6d5ce", 0.25, 2),
    "service": ("#d6d5ce", 0.25, 2),
    "other": ("#d6d5ce", 0.25, 2),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": HAIRLINE, "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def segments(nodes: pd.DataFrame, edges: pd.DataFrame, x="lon", y="lat") -> np.ndarray:
    """Straight u->v segments. Curved OSM geometries are not needed at city scale."""
    pos = nodes.set_index("node_id")[[x, y]]
    a = pos.loc[edges.u].to_numpy()
    b = pos.loc[edges.v].to_numpy()
    return np.stack([a, b], axis=1)


def map_axes(ax, nodes: pd.DataFrame, pad=0.01):
    ax.set_aspect(1 / np.cos(np.radians(nodes.lat.mean())))
    ax.set_xlim(nodes.lon.min() - pad, nodes.lon.max() + pad)
    ax.set_ylim(nodes.lat.min() - pad, nodes.lat.max() + pad)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_boundary(ax, boundary):
    if boundary is None:
        return
    boundary.to_crs("EPSG:4326").boundary.plot(ax=ax, color=MUTED, lw=0.8, ls="--", zorder=1)


def draw_roads(ax, nodes, edges, by_class=True, values=None, cmap=SEQ, vmin=None, vmax=None,
               lw=0.5):
    """Draw edges either styled by road class or coloured by a numeric edge value."""
    segs = segments(nodes, edges)
    if by_class and values is None:
        for cls, (col, w, z) in sorted(ROAD_STYLE.items(), key=lambda kv: kv[1][2]):
            m = (edges.road_class == cls).to_numpy()
            if m.any():
                ax.add_collection(LineCollection(segs[m], colors=col, linewidths=w, zorder=z,
                                                 capstyle="round"))
        return None
    order = np.argsort(values)  # draw high values on top
    lc = LineCollection(segs[order], array=np.asarray(values)[order], cmap=cmap, linewidths=lw,
                        capstyle="round")
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    return lc


def road_legend(ax, edges):
    from matplotlib.lines import Line2D
    km = edges.groupby("road_class").length_m.sum() / 1000
    shown = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service"]
    handles = [Line2D([], [], color=ROAD_STYLE[c][0], lw=max(ROAD_STYLE[c][1] * 2, 1.2),
                      label=f"{c}  ({km.get(c, 0):,.0f} km)") for c in shown if km.get(c, 0) > 0]
    leg = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8,
                    title="road class", title_fontsize=8)
    leg.get_frame().set_edgecolor(HAIRLINE)
    return leg


def colorbar(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label(label, color=INK_2)
    cb.ax.tick_params(colors=MUTED, length=0)
    return cb


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(path.parents[2])}")
