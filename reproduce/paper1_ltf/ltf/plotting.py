"""Shared figure styling.

The categorical palette below was validated with the data-viz colour checks
(lightness band, chroma floor, all-pairs CVD separation, normal-vision floor,
contrast). CVD separation lands in the 6-8 band, which is admissible only with a
secondary encoding, so every series also carries a distinct marker shape, a legend,
and a written-out table view (`results/*.csv`).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d8d7d2"

# Validated 5-slot categorical palette.
PALETTE = ["#2a78d6", "#e34948", "#1baf7a", "#4a3aa7", "#eda100"]
MARKERS = ["o", "X", "^", "v", "s"]

# Fixed slot per method, so a series never changes colour between figures.
SERIES_STYLE = {
    "LAF":                       (PALETTE[0], MARKERS[0]),
    "Greedy":                    (PALETTE[1], MARKERS[1]),
    "REASSIGN":                  (PALETTE[2], MARKERS[2]),
    "Balance Ride-Pooling":      (PALETTE[3], MARKERS[3]),
    "Proposed Method":           (PALETTE[4], MARKERS[4]),
    "Our Method":                (PALETTE[0], MARKERS[0]),
    "Our Method w/o Prediction": (PALETTE[1], MARKERS[1]),
    "Our Method w/o Fairness":   (PALETTE[4], MARKERS[4]),
}


def new_axes(figsize=(7.8, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    return fig, ax


def finish(fig, ax, title: str, xlabel: str, ylabel: str, path, legend: bool = True):
    ax.set_title(title, color=INK, fontsize=10.5, pad=10, loc="left", wrap=True)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if legend:
        leg = ax.legend(frameon=True, fontsize=8.5, loc="upper left",
                        bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        leg.get_frame().set_facecolor(SURFACE)
        leg.get_frame().set_edgecolor(GRID)
        for text in leg.get_texts():
            text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[fig] wrote {path}")
