"""Sanity checks for event representations.

These are deliberately property-based rather than golden-value: they assert the
invariants each representation is *supposed* to have, which is what actually
catches bugs when you tweak binning later.
"""

import numpy as np
import pytest
import torch

from src.data.representations import (
    events_to_histogram,
    events_to_snn_input,
    events_to_time_surface,
    events_to_voxel_grid,
)

H, W = 16, 24


@pytest.fixture
def events():
    rng = np.random.default_rng(0)
    n = 500
    t = np.sort(rng.integers(0, 100_000, size=n)).astype(np.int64)
    x = rng.integers(0, W, size=n).astype(np.uint16)
    y = rng.integers(0, H, size=n).astype(np.uint16)
    p = rng.integers(0, 2, size=n).astype(np.int8)
    return x, y, t, p


def test_histogram_conserves_event_count(events):
    x, y, t, p = events
    hist = events_to_histogram(x, y, p, H, W)
    assert hist.shape == (2, H, W)
    # Every event must land in exactly one bin.
    assert hist.sum().item() == pytest.approx(len(x))


def test_voxel_grid_conserves_total_polarity(events):
    x, y, t, p = events
    voxel = events_to_voxel_grid(x, y, t, p, H, W, num_bins=5)
    assert voxel.shape == (5, H, W)
    # Linear interpolation splits each event's polarity across two bins with
    # weights summing to 1, so total signed polarity is conserved.
    expected = np.where(p > 0, 1.0, -1.0).sum()
    assert voxel.sum().item() == pytest.approx(expected, abs=1e-3)


def test_time_surface_is_bounded_and_recent_is_hot(events):
    x, y, t, p = events
    surf = events_to_time_surface(x, y, t, p, H, W, tau=50_000.0)
    assert surf.shape == (2, H, W)
    assert surf.min().item() >= 0.0
    assert surf.max().item() <= 1.0
    # The last event is at t_end, so its pixel must decay by exactly 0.
    last_c = int(p[-1] > 0)
    assert surf[last_c, int(y[-1]), int(x[-1])].item() == pytest.approx(1.0)


def test_snn_input_is_strictly_binary(events):
    x, y, t, p = events
    spikes = events_to_snn_input(x, y, t, p, H, W, num_steps=10)
    assert spikes.shape == (10, 2, H, W)
    # This is the invariant the entire energy argument rests on.
    assert torch.all((spikes == 0) | (spikes == 1))
    assert spikes.sum() > 0


def test_snn_input_never_exceeds_step_range(events):
    """Regression guard: the final event must not index out of bounds."""
    x, y, t, p = events
    for steps in (1, 2, 7, 10, 33):
        spikes = events_to_snn_input(x, y, t, p, H, W, num_steps=steps)
        assert spikes.shape[0] == steps


def test_empty_stream_returns_zeros():
    empty = np.array([], dtype=np.int64)
    ep = np.array([], dtype=np.int8)
    assert events_to_histogram(empty, empty, ep, H, W).sum() == 0
    assert events_to_voxel_grid(empty, empty, empty, ep, H, W).sum() == 0
    assert events_to_time_surface(empty, empty, empty, ep, H, W).sum() == 0
    assert events_to_snn_input(empty, empty, empty, ep, H, W).sum() == 0


def test_single_timestamp_does_not_divide_by_zero():
    """All events simultaneous — t_max == t_min. Must not produce NaN."""
    n = 20
    t = np.full(n, 1234, dtype=np.int64)
    x = np.zeros(n, dtype=np.uint16)
    y = np.zeros(n, dtype=np.uint16)
    p = np.ones(n, dtype=np.int8)
    voxel = events_to_voxel_grid(x, y, t, p, H, W, num_bins=5)
    spikes = events_to_snn_input(x, y, t, p, H, W, num_steps=10)
    assert not torch.isnan(voxel).any()
    assert not torch.isnan(spikes).any()
