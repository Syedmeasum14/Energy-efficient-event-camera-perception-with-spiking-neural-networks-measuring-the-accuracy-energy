"""Qualitative figures: what event data looks like, and what the SNN predicts.

Separate from make_figures.py because these need the dataset and a trained
checkpoint, whereas make_figures.py regenerates from committed CSVs alone.

    PYTHONPATH=. python scripts/make_qualitative_figures.py

Requires: data/nmnist downloaded, runs/nmnist_bntt/snn_best.pt present.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.datasets import load_nmnist  # noqa: E402
from src.models.classifier import SNNClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "docs" / "figures"
CKPT = ROOT / "runs" / "nmnist_bntt" / "snn_best.pt"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e",
                  muted="#898781", on="#2a78d6", off="#eb6834",
                  ramp_mid="#86b6ef", ramp_dark="#0d366b",
                  good="#0ca30c", bad="#d03b3b"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7",
                 muted="#898781", on="#3987e5", off="#d95926",
                 ramp_mid="#256abf", ramp_dark="#cde2fb",
                 good="#0ca30c", bad="#d03b3b"),
}
NUM_STEPS = 10


def save(fig, name: str, mode: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}-{mode}.png"
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render_polarity(ax, spikes_2ch: np.ndarray, t: dict) -> None:
    """Draw a (2, H, W) polarity tensor: ON one colour, OFF another.

    Only meaningful for a SINGLE timestep, where the tensor is sparse. Summing
    timesteps first and thresholding produces a solid blob -- the saccade sweeps
    events across the whole digit, so almost every pixel fires at some point.
    Use render_density for accumulated views.
    """
    h, w = spikes_2ch.shape[1:]
    rgb = np.empty((h, w, 3))
    rgb[:] = matplotlib.colors.to_rgb(t["surface"])
    rgb[spikes_2ch[0] > 0] = matplotlib.colors.to_rgb(t["off"])
    rgb[spikes_2ch[1] > 0] = matplotlib.colors.to_rgb(t["on"])
    ax.imshow(rgb, interpolation="nearest")
    _bare(ax)


def render_density(ax, counts: np.ndarray, t: dict, frame: bool = False) -> None:
    """Accumulated event COUNT on a sequential one-hue ramp.

    Count carries the structure: edges fire repeatedly as the camera moves,
    flat regions barely fire. Encoding magnitude rather than presence is what
    makes the subject legible -- a binary threshold on the accumulated sum
    fills the whole shape into a solid blob.

    Normalised to the 99th percentile rather than the max: a single hot pixel
    would otherwise compress everything else into the palest step, which is
    what made sparse N-CARS crops look almost blank.

    `frame` draws a hairline around the full canvas. N-CARS crops are variable
    size and sit in the top-left of a padded 120x100 canvas, so without a frame
    the panels appear to float at different sizes.
    """
    total = counts.sum(0)
    peak = np.percentile(total, 99.0) if total.max() > 0 else 0.0
    if peak <= 0:
        peak = total.max() or 1.0
    norm = np.clip(total / peak, 0.0, 1.0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "events", [t["surface"], t["ramp_mid"], t["ramp_dark"]])
    ax.imshow(norm, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    _bare(ax)
    if frame:
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(t["muted"])
            side.set_linewidth(0.6)
            side.set_alpha(0.35)


# ---------------------------------------------------------------------------
# What the raw event stream turns into
# ---------------------------------------------------------------------------
def fig_representations(mode: str, sample_idx: int = 3) -> None:
    t = THEMES[mode]
    _, test_spikes, _ = load_nmnist(str(ROOT / "data" / "nmnist"),
                                    representation="snn_spikes",
                                    num_steps=NUM_STEPS, subset=400)
    spikes, label = test_spikes[sample_idx]

    steps_to_show = [0, 2, 4, 6, 8]
    fig, axes = plt.subplots(1, len(steps_to_show) + 1,
                             figsize=(2.0 * (len(steps_to_show) + 1), 2.5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    # Accumulated view first: what a frame camera would roughly see.
    render_density(axes[0], spikes.numpy().sum(0), t)
    axes[0].set_title("all 10 steps\naccumulated", color=t["secondary"],
                      fontsize=9, pad=8, linespacing=1.4)

    for ax, step in zip(axes[1:], steps_to_show):
        render_polarity(ax, spikes[step].numpy(), t)
        ax.set_title(f"t = {step}", color=t["muted"], fontsize=9, pad=8)

    fig.suptitle(
        f"One N-MNIST sample (digit {label}) as a binary spike tensor",
        color=t["ink"], fontsize=13, fontweight="bold", x=0.005, ha="left", y=1.10)
    fig.text(0.005, -0.04,
             "Each timestep is mostly empty — that sparsity is what the SNN exploits. "
             "Blue = brightness up (ON), orange = brightness down (OFF).",
             color=t["muted"], fontsize=9, ha="left")
    fig.tight_layout()
    save(fig, "event-representations", mode)


# ---------------------------------------------------------------------------
# What the trained SNN actually predicts
# ---------------------------------------------------------------------------
def fig_predictions(mode: str, n: int = 12) -> None:
    t = THEMES[mode]
    _, test_spikes, _ = load_nmnist(str(ROOT / "data" / "nmnist"),
                                    representation="snn_spikes",
                                    num_steps=NUM_STEPS, subset=2000)

    model = SNNClassifier(in_channels=2, num_classes=10, width=16,
                          num_blocks=3, num_steps=NUM_STEPS)
    model.load_state_dict(torch.load(CKPT, weights_only=False, map_location="cpu")["model"])
    model.eval()

    cols, rows_n = 6, 2
    fig, axes = plt.subplots(rows_n, cols, figsize=(1.55 * cols, 1.95 * rows_n), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    correct = 0
    for i, ax in enumerate(axes.flat):
        spikes, label = test_spikes[i]
        with torch.no_grad():
            logits = model(spikes.unsqueeze(1))  # (T, 1, C, H, W)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(probs.argmax())
            conf = float(probs[pred])

        render_density(ax, spikes.numpy().sum(0), t)
        ok = pred == label
        correct += ok
        colour = t["good"] if ok else t["bad"]
        mark = "correct" if ok else "wrong"
        ax.set_title(f"pred {pred}  ({conf * 100:.0f}%)", color=colour,
                     fontsize=9.5, fontweight="bold", pad=6)
        ax.set_xlabel(f"true {label} · {mark}", color=t["muted"], fontsize=8.5, labelpad=4)

    fig.suptitle(f"SNN predictions on held-out N-MNIST — {correct}/{len(axes.flat)} correct here",
                 color=t["ink"], fontsize=13, fontweight="bold", x=0.005, ha="left", y=1.04)
    fig.text(0.005, -0.03,
             "Events accumulated over all 10 timesteps for display; the model consumes them "
             "step by step. Full held-out accuracy: 95.55%.",
             color=t["muted"], fontsize=9, ha="left")
    fig.tight_layout()
    save(fig, "predictions", mode)


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]
    if not CKPT.exists():
        raise SystemExit(f"missing checkpoint {CKPT} — run scripts/run_nmnist.py first")
    for mode in ("light", "dark"):
        print(f"{mode}:")
        fig_representations(mode)
        fig_predictions(mode)
        if NCARS_CKPT.exists():
            fig_ncars_predictions(mode)
        else:
            print(f"  skipping N-CARS figure: {NCARS_CKPT} not found")




# ---------------------------------------------------------------------------
# Rung 2: what the SNN predicts on real automotive event data
# ---------------------------------------------------------------------------
NCARS_CKPT = ROOT / "runs" / "ncars_lambda_1.0" / "snn_best.pt"
NCARS_ROOT = ROOT / "data" / "ncars"
NCARS_CLASSES = {0: "background", 1: "car"}


def fig_ncars_predictions(mode: str, n: int = 12) -> None:
    """Held-out N-CARS crops with the sparse SNN's prediction and ground truth.

    Uses the lambda=1.0 checkpoint -- the sparse one the README headlines
    (88.52% at 31 uJ), not the unpenalised baseline.
    """
    from src.data.datasets import load_ncars

    t = THEMES[mode]
    _, test_ds, _ = load_ncars(str(NCARS_ROOT), representation="snn_spikes",
                               num_steps=NUM_STEPS, subset=600)

    model = SNNClassifier(in_channels=2, num_classes=2, width=16,
                          num_blocks=4, threshold=0.5, num_steps=NUM_STEPS)
    model.load_state_dict(
        torch.load(NCARS_CKPT, weights_only=False, map_location="cpu")["model"])
    model.eval()

    # Pick a balanced set so the figure is not accidentally all one class.
    wanted = {0: n // 2, 1: n // 2}
    picks: list[int] = []
    for i in range(len(test_ds)):
        label = test_ds[i][1]
        if wanted.get(label, 0) > 0:
            picks.append(i)
            wanted[label] -= 1
        if not any(wanted.values()):
            break

    cols, rows_n = 6, 2
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.1 * cols, 2.3 * rows_n), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    correct = 0
    for idx, ax in zip(picks, axes.flat):
        spikes, label = test_ds[idx]
        with torch.no_grad():
            probs = torch.softmax(model(spikes.unsqueeze(1)), dim=1)[0]
            pred = int(probs.argmax())
            conf = float(probs[pred])

        render_density(ax, spikes.numpy().sum(0), t, frame=True)
        ok = pred == label
        correct += ok
        ax.set_title(f"{NCARS_CLASSES[pred]}  ({conf * 100:.0f}%)",
                     color=t["good"] if ok else t["bad"],
                     fontsize=9.5, fontweight="bold", pad=6)
        ax.set_xlabel(f"true: {NCARS_CLASSES[label]}", color=t["muted"],
                      fontsize=8.5, labelpad=4)

    fig.suptitle(
        f"N-CARS: spiking network predictions on held-out driving data "
        f"({correct}/{len(picks)} correct here)",
        color=t["ink"], fontsize=13, fontweight="bold", x=0.005, ha="left", y=1.04)
    fig.text(0.005, -0.03,
             "Recorded from an event camera behind a car windshield. Events accumulated "
             "over 10 timesteps for display; the model consumes them step by step. "
             "Full held-out accuracy: 88.52% at 31 uJ/sample.",
             color=t["muted"], fontsize=9, ha="left")
    fig.tight_layout()
    save(fig, "ncars-predictions", mode)


if __name__ == "__main__":
    main()
