"""Generate the README figures from measured results.

Reads results/*.csv (committed, so figures are reproducible without re-training)
and writes light + dark PNGs to docs/figures/. The README uses <picture> so
GitHub serves whichever matches the reader's theme.

    PYTHONPATH=. python scripts/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"

# Palette. Categorical slots 1-3, which are the three that validate on the
# all-pairs check (scatter needs all-pairs, not just adjacent).
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
        grid="#e1e0d9", baseline="#c3c2b7",
        series=("#2a78d6", "#eb6834", "#1baf7a"),
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", baseline="#383835",
        series=("#3987e5", "#d95926", "#199e70"),
    ),
}

FONT = ["system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]


def load_training() -> list[dict]:
    with open(RESULTS / "nmnist_training.csv") as f:
        return list(csv.DictReader(f))


def load_summary() -> list[dict]:
    with open(RESULTS / "summary.csv") as f:
        return list(csv.DictReader(f))


def style_axes(ax, t: dict, *, ygrid: bool = True) -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["baseline"])
        ax.spines[side].set_linewidth(1.0)
    if ygrid:
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(t["secondary"])


def new_fig(t: dict, w: float, h: float):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    return fig, ax


def save(fig, name: str, mode: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}-{mode}.png"
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Figure 1 — validation accuracy over training. The BNTT story.
# ---------------------------------------------------------------------------
def fig_training_curves(mode: str) -> None:
    t = THEMES[mode]
    rows = load_training()
    fig, ax = new_fig(t, 7.2, 4.2)

    series = [
        ("CNN", [r for r in rows if r["run"] == "bntt" and r["model"] == "cnn"], t["series"][0]),
        ("SNN + BNTT", [r for r in rows if r["run"] == "bntt" and r["model"] == "snn"], t["series"][1]),
        ("SNN, plain BatchNorm", [r for r in rows if r["run"] == "plain" and r["model"] == "snn"], t["series"][2]),
    ]

    for label, data, color in series:
        xs = [int(r["epoch"]) for r in data]
        ys = [float(r["val_acc"]) for r in data]
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label,
                marker="o", markersize=4, markeredgecolor=t["surface"], markeredgewidth=1.2)
        # Direct label at the line end. Required relief for the light-mode aqua,
        # and it means identity never depends on colour alone.
        ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", color=color, fontsize=10,
                    fontweight="bold", va="center")

    ax.set_xlabel("epoch", color=t["secondary"], fontsize=10)
    ax.set_ylabel("validation accuracy (%)", color=t["secondary"], fontsize=10)
    ax.set_title("Plain BatchNorm costs the SNN 29 accuracy points",
                 color=t["ink"], fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.set_xlim(0.5, 17.2)
    ax.set_ylim(30, 102)
    style_axes(ax, t)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t["secondary"])
    save(fig, "training-curves", mode)


# ---------------------------------------------------------------------------
# Figure 2 — the headline trade-off: accuracy against estimated energy.
# ---------------------------------------------------------------------------
def fig_accuracy_vs_energy(mode: str) -> None:
    t = THEMES[mode]
    rows = load_summary()
    fig, ax = new_fig(t, 6.4, 4.4)

    order = ["cnn", "snn_bntt", "snn_plain"]
    labels = {"cnn": "CNN", "snn_bntt": "SNN + BNTT", "snn_plain": "SNN, plain BN"}
    offsets = {"cnn": (-10, -20), "snn_bntt": (10, 10), "snn_plain": (10, 8)}

    for i, key in enumerate(order):
        r = next(x for x in rows if x["model"] == key)
        x, y = float(r["energy_uj"]), float(r["accuracy"])
        color = t["series"][i]
        ax.scatter([x], [y], s=190, color=color, zorder=3,
                   edgecolor=t["surface"], linewidth=2, label=labels[key])
        ax.annotate(f"{labels[key]}\n{y:.2f}%  ·  {x:.2f} µJ", (x, y),
                    xytext=offsets[key], textcoords="offset points",
                    color=color, fontsize=9.5, fontweight="bold",
                    va="center", linespacing=1.4)

    ax.set_xlabel("estimated energy per sample (µJ) - lower is better", color=t["secondary"], fontsize=10)
    ax.set_ylabel("accuracy (%) - higher is better", color=t["secondary"], fontsize=10)
    ax.set_title("Accuracy vs estimated inference energy",
                 color=t["ink"], fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.set_xlim(3.5, 18.5)
    ax.set_ylim(58, 104)
    style_axes(ax, t)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)

    ax.annotate("2.30 points of accuracy\nfor 2.5× less energy",
                xy=(6.25, 95.55), xytext=(11.2, 78),
                color=t["muted"], fontsize=9, ha="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=t["muted"],
                                linewidth=1, connectionstyle="arc3,rad=-0.25"))

    leg = ax.legend(loc="lower left", frameon=False, fontsize=9.5, scatterpoints=1)
    for txt in leg.get_texts():
        txt.set_color(t["secondary"])
    save(fig, "accuracy-vs-energy", mode)


# ---------------------------------------------------------------------------
# Figure 3 — where the energy goes. Two panels, never a dual axis: operations
# and energy are different scales and different units.
# ---------------------------------------------------------------------------
def fig_energy_breakdown(mode: str) -> None:
    t = THEMES[mode]
    rows = load_summary()
    cnn = next(r for r in rows if r["model"] == "cnn")
    snn = next(r for r in rows if r["model"] == "snn_bntt")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.8), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    panels = [
        (ax1, "operations per sample (millions)",
         [float(cnn["operations_m"]), float(snn["operations_m"])],
         ["{:.2f} M MAC", "{:.2f} M SynOp"]),
        (ax2, "estimated energy per sample (µJ)",
         [float(cnn["energy_uj"]), float(snn["energy_uj"])],
         ["{:.2f} µJ", "{:.2f} µJ"]),
    ]
    names = ["CNN", "SNN + BNTT"]
    colors = [t["series"][0], t["series"][1]]

    for ax, title, values, fmts in panels:
        bars = ax.bar(names, values, color=colors, width=0.5, zorder=3)
        for rect, val, fmt in zip(bars, values, fmts):
            ax.annotate(fmt.format(val), (rect.get_x() + rect.get_width() / 2, val),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", color=t["secondary"], fontsize=9.5, fontweight="bold")
        ax.set_title(title, color=t["ink"], fontsize=10.5, fontweight="bold", loc="left", pad=10)
        ax.set_ylim(0, max(values) * 1.28)
        style_axes(ax, t)

    fig.suptitle("The SNN does MORE operations and still uses less energy",
                 color=t["ink"], fontsize=13, fontweight="bold", x=0.005, ha="left", y=1.06)
    fig.text(0.005, -0.06,
             "Each spike triggers an accumulate (0.9 pJ), not a multiply-accumulate (4.6 pJ).",
             color=t["muted"], fontsize=9, ha="left")
    fig.tight_layout()
    save(fig, "energy-breakdown", mode)


# ---------------------------------------------------------------------------
# Figure 4 — spike density over training. The lever on energy.
# ---------------------------------------------------------------------------
def fig_firing_rate(mode: str) -> None:
    t = THEMES[mode]
    rows = load_training()
    fig, ax = new_fig(t, 7.2, 3.6)

    series = [
        ("SNN + BNTT", [r for r in rows if r["run"] == "bntt" and r["model"] == "snn"], t["series"][1]),
        ("SNN, plain BatchNorm", [r for r in rows if r["run"] == "plain" and r["model"] == "snn"], t["series"][2]),
    ]
    for label, data, color in series:
        pts = [(int(r["epoch"]), float(r["val_fire"])) for r in data if r["val_fire"]]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label,
                marker="o", markersize=4, markeredgecolor=t["surface"], markeredgewidth=1.2)
        ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", color=color, fontsize=10,
                    fontweight="bold", va="center")

    ax.axhspan(2, 20, color=t["series"][0], alpha=0.07, zorder=0)
    ax.annotate("target band for a strong energy win", (1.2, 5.5),
                color=t["muted"], fontsize=8.5)

    ax.set_xlabel("epoch", color=t["secondary"], fontsize=10)
    ax.set_ylabel("mean firing rate (%)", color=t["secondary"], fontsize=10)
    ax.set_title("Spike density stays around 21% — the remaining headroom",
                 color=t["ink"], fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.set_xlim(0.5, 17.2)
    ax.set_ylim(0, 32)
    style_axes(ax, t)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t["secondary"])
    save(fig, "firing-rate", mode)


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT
    for mode in ("light", "dark"):
        print(f"{mode}:")
        fig_training_curves(mode)
        fig_accuracy_vs_energy(mode)
        fig_energy_breakdown(mode)
        fig_firing_rate(mode)


if __name__ == "__main__":
    main()
