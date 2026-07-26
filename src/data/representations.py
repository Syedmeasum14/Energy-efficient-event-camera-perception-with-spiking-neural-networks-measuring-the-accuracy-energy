"""Event stream -> tensor representations.

BACKGROUND (read this before the code)
--------------------------------------
An event camera does not produce frames. Each pixel independently fires an
"event" whenever the log-intensity it sees changes by more than a threshold C:

    | log I(x, y, t) - log I(x, y, t_last_event) |  >  C

An event is a 4-tuple:

    x, y : pixel coordinates          (uint16)
    t    : timestamp, microseconds    (int64)  <-- ~1 us resolution
    p    : polarity, +1 or -1         (int8)   <-- brightness up or down

So the raw data is a *sparse, asynchronous point cloud in space-time*, not a
dense array. Two consequences drive this whole project:

1. There is nothing to feed a CNN. A CNN needs a dense (C, H, W) tensor, so we
   must *choose* how to collapse the time axis. That choice is lossy, and it is
   the single biggest design decision in event-based vision. Hence this file.

2. An SNN does NOT need that collapse. It consumes events as spikes over
   discrete timesteps, which is why it can be cheaper: it only does work where
   events actually occurred. The `to_snn_input` function preserves the time
   axis rather than destroying it.

The representations below are ordered from most lossy to least.
"""

from __future__ import annotations

import numpy as np
import torch


def events_to_histogram(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
) -> torch.Tensor:
    """Simplest representation: count events per pixel, split by polarity.

    Output shape: (2, H, W) — channel 0 = negative events, channel 1 = positive.

    Time is discarded ENTIRELY. Every event in the window is treated as
    simultaneous. This is fast, and it is a surprisingly strong baseline for
    detection, because object *shape* survives even when motion timing does not.

    Its failure mode: two objects crossing the same pixels at different moments
    within the window become indistinguishable, and fast motion smears.
    """
    # Fold polarity into a channel index: p=-1 -> 0, p=+1 -> 1.
    # DSEC stores polarity as {0, 1} already; normalise both conventions.
    pol_idx = (p > 0).astype(np.int64)

    hist = torch.zeros(2, height, width, dtype=torch.float32)

    # Flatten (channel, y, x) into a single index so we can use bincount, which
    # is dramatically faster than a Python loop over millions of events.
    flat_idx = pol_idx * (height * width) + y.astype(np.int64) * width + x.astype(np.int64)
    counts = np.bincount(flat_idx, minlength=2 * height * width)

    return torch.from_numpy(counts).float().view(2, height, width)


def events_to_voxel_grid(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    num_bins: int = 5,
) -> torch.Tensor:
    """The standard representation in the literature (Zhu et al., 2019).

    Output shape: (num_bins, H, W).

    Time is normalised to [0, num_bins - 1] and each event's polarity is
    *linearly interpolated* into its two neighbouring temporal bins. That
    interpolation is the important part: it keeps sub-bin timing information
    that a hard assignment would throw away, and it makes the representation
    differentiable with respect to time.

    Think of it as a short video with `num_bins` frames, where pixel values are
    signed accumulated polarity rather than intensity.

    num_bins is a real trade-off: more bins = finer temporal detail but sparser,
    noisier individual bins, and a linearly larger input tensor.
    """
    voxel = torch.zeros(num_bins, height, width, dtype=torch.float32)

    if len(t) == 0:
        return voxel

    # Normalise timestamps to [0, num_bins - 1].
    t = t.astype(np.float64)
    t_min, t_max = t[0], t[-1]
    if t_max == t_min:
        # Degenerate window (all events same timestamp) — fall back to bin 0.
        t_norm = np.zeros_like(t)
    else:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1)

    # Map polarity to {-1, +1} regardless of input convention.
    pol = np.where(p > 0, 1.0, -1.0)

    t_lo = np.floor(t_norm).astype(np.int64)
    t_hi = t_lo + 1
    # Weight splits between the two bins by how close the event is to each.
    w_hi = t_norm - t_lo
    w_lo = 1.0 - w_hi

    xi = x.astype(np.int64)
    yi = y.astype(np.int64)

    flat = voxel.view(-1)
    hw = height * width

    for bin_idx, weight in ((t_lo, w_lo), (t_hi, w_hi)):
        # Drop contributions that fall outside the grid (t_hi can hit num_bins).
        valid = (bin_idx >= 0) & (bin_idx < num_bins)
        idx = bin_idx[valid] * hw + yi[valid] * width + xi[valid]
        vals = torch.from_numpy(pol[valid] * weight[valid]).float()
        # index_add_ accumulates rather than overwrites — essential, since many
        # events land on the same (bin, y, x).
        flat.index_add_(0, torch.from_numpy(idx), vals)

    return voxel


def events_to_time_surface(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    tau: float = 50_000.0,
) -> torch.Tensor:
    """Exponentially-decayed map of the most recent event time per pixel.

    Output shape: (2, H, W), one channel per polarity.

    Each pixel holds exp(-(t_end - t_last) / tau), so a pixel that just fired is
    near 1.0 and one that fired long ago decays toward 0. `tau` is in the same
    units as `t` (microseconds for DSEC), so tau=50_000 is a 50 ms memory.

    This is the cheapest way to keep "what happened most recently" and it is
    what a lot of classical event-vision work (HOTS, HATS) builds on. Unlike the
    voxel grid it keeps only the LAST event per pixel — density information is
    lost, recency is preserved. The opposite trade to the histogram.
    """
    surface = torch.zeros(2, height, width, dtype=torch.float32)

    if len(t) == 0:
        return surface

    t_end = float(t[-1])
    pol_idx = (p > 0).astype(np.int64)

    # Events arrive time-sorted, so writing in order means the LAST write per
    # pixel wins — which is exactly the "most recent event" semantics we want.
    decayed = np.exp(-(t_end - t.astype(np.float64)) / tau)
    surface[pol_idx, y.astype(np.int64), x.astype(np.int64)] = torch.from_numpy(decayed).float()

    return surface


def events_to_snn_input(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    num_steps: int = 10,
) -> torch.Tensor:
    """Binary spike tensor for the SNN. THIS is the one that matters for model C.

    Output shape: (num_steps, 2, H, W), values in {0, 1}.

    Note the difference from the voxel grid: we do NOT interpolate and we do NOT
    accumulate magnitudes. A pixel either spiked in a given timestep or it did
    not. That binarisation is what makes the energy argument valid — downstream,
    a spike triggers an *accumulate* (~0.9 pJ), not a *multiply-accumulate*
    (~4.6 pJ). If we let values be real-valued counts we would silently be back
    to multiplications and the energy claim would be false.

    The leading time axis is preserved and is iterated over by the LIF layers,
    which carry membrane potential across steps. num_steps is the SNN's
    equivalent of "temporal resolution" and directly scales both compute and
    the difficulty of credit assignment through time.
    """
    spikes = torch.zeros(num_steps, 2, height, width, dtype=torch.float32)

    if len(t) == 0:
        return spikes

    t = t.astype(np.float64)
    t_min, t_max = t[0], t[-1]
    if t_max == t_min:
        step_idx = np.zeros(len(t), dtype=np.int64)
    else:
        step_idx = np.floor((t - t_min) / (t_max - t_min) * num_steps).astype(np.int64)
        # The final event maps exactly to num_steps; clamp it into the last bin.
        step_idx = np.clip(step_idx, 0, num_steps - 1)

    pol_idx = (p > 0).astype(np.int64)

    # Direct assignment (not accumulation) enforces the binary constraint:
    # multiple events on the same pixel in the same step still yield one spike.
    spikes[step_idx, pol_idx, y.astype(np.int64), x.astype(np.int64)] = 1.0

    return spikes


# Registry so configs can select a representation by name.
REPRESENTATIONS = {
    "histogram": events_to_histogram,
    "voxel_grid": events_to_voxel_grid,
    "time_surface": events_to_time_surface,
    "snn_spikes": events_to_snn_input,
}
