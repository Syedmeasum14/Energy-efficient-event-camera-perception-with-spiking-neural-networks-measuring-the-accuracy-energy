"""Tests for energy accounting.

The MAC formulas are checked against hand-computed values. If these drift, every
headline number in the project is wrong -- and wrong in a way no training curve
would ever reveal.
"""

import pytest
import torch
import torch.nn as nn

from src.engine.energy import (
    PJ_PER_MAC,
    PJ_PER_SOP,
    EnergyCounter,
    compare,
)
from src.models.lif import LIF, reset_all


def test_linear_macs_match_hand_count():
    """Linear(10 -> 5): each of 5 outputs is a dot product over 10 inputs = 50."""
    model = nn.Linear(10, 5, bias=False)
    with EnergyCounter(model, mode="dense") as c:
        model(torch.ones(1, 10))
    assert c.report().total_macs == 50


def test_conv2d_macs_match_hand_count():
    """Conv2d(3 -> 8, k=3) on 32x32 with padding=1 keeps 32x32 spatially.
    MACs = (8 * 32 * 32 outputs) * (3 in_channels * 3 * 3 kernel) = 8192 * 27."""
    model = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    with EnergyCounter(model, mode="dense") as c:
        model(torch.ones(1, 3, 32, 32))
    assert c.report().total_macs == 8 * 32 * 32 * 3 * 3 * 3


def test_macs_are_per_sample_not_per_batch():
    """Batch size must not change the per-sample cost."""
    model = nn.Linear(10, 5, bias=False)
    counts = []
    for batch in (1, 4, 16):
        with EnergyCounter(model, mode="dense") as c:
            model(torch.ones(batch, 10))
        counts.append(c.report().total_macs)
    assert len(set(counts)) == 1, f"per-sample MACs varied with batch: {counts}"


def test_grouped_conv_costs_less():
    """Groups divide the input channels each filter sees."""
    plain = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
    grouped = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)
    x = torch.ones(1, 8, 16, 16)

    with EnergyCounter(plain, mode="dense") as c:
        plain(x)
    plain_macs = c.report().total_macs
    with EnergyCounter(grouped, mode="dense") as c:
        grouped(x)
    grouped_macs = c.report().total_macs

    assert grouped_macs == plain_macs // 8


def test_energy_uses_correct_constants():
    model = nn.Linear(10, 5, bias=False)
    with EnergyCounter(model, mode="dense") as c:
        model(torch.ones(1, 10))
    assert c.report().energy_pj == pytest.approx(50 * PJ_PER_MAC)


# --------------------------------------------------------------------------
# Sparsity: the core of the SNN energy claim
# --------------------------------------------------------------------------

def test_synops_scale_with_input_sparsity():
    """THE key property. Half the input spikes must cost half the SynOps."""
    model = nn.Linear(100, 10, bias=False)

    dense_in = torch.ones(1, 100)
    half_in = torch.zeros(1, 100)
    half_in[0, :50] = 1.0

    with EnergyCounter(model, mode="spiking") as c:
        model(dense_in)
    full = c.report().total_synops

    with EnergyCounter(model, mode="spiking") as c:
        model(half_in)
    half = c.report().total_synops

    assert half == pytest.approx(full * 0.5, rel=1e-4)


def test_zero_input_costs_nothing_when_spiking():
    """No spikes means no work -- the whole point of event-driven compute."""
    model = nn.Linear(100, 10, bias=False)
    with EnergyCounter(model, mode="spiking") as c:
        model(torch.zeros(1, 100))
    assert c.report().total_synops == 0.0


def test_dense_mode_ignores_sparsity():
    """A dense accelerator pays full price even for an all-zero input."""
    model = nn.Linear(100, 10, bias=False)
    with EnergyCounter(model, mode="dense") as c:
        model(torch.zeros(1, 100))
    assert c.report().total_synops == 1000


def test_synops_accumulate_over_timesteps():
    model = nn.Linear(100, 10, bias=False)
    x = torch.ones(1, 100)
    with EnergyCounter(model, mode="spiking") as c:
        for _ in range(5):
            model(x)
    assert c.report(timesteps=5).total_synops == pytest.approx(5 * 1000)


def test_sparse_snn_beats_dense_cnn_on_energy():
    """End-to-end sanity: at 10% firing rate over 4 steps, the SNN should win.

    4 steps x 10% density = 0.4x the operations, at 0.9/4.6 the cost per op,
    so roughly 0.4 * 0.196 ~= 0.08x the energy -- about 13x cheaper.
    """
    torch.manual_seed(0)
    model = nn.Linear(1000, 100, bias=False)

    dense_x = torch.ones(1, 1000)
    with EnergyCounter(model, mode="dense") as c:
        model(dense_x)
    dense_report = c.report()

    sparse_x = (torch.rand(1, 1000) < 0.1).float()
    with EnergyCounter(model, mode="spiking") as c:
        for _ in range(4):
            model(sparse_x)
    spike_report = c.report(timesteps=4)

    assert spike_report.energy_uj < dense_report.energy_uj
    ratio = dense_report.energy_uj / spike_report.energy_uj
    assert 8.0 < ratio < 20.0, f"unexpected energy ratio {ratio:.1f}x"
    assert "less estimated energy" in compare(dense_report, spike_report)


def test_counter_works_with_real_lif_network():
    """Integration: hooks must survive a stateful multi-timestep SNN."""
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(64, 32, bias=False), LIF(beta=0.9, threshold=1.0),
        nn.Linear(32, 8, bias=False), LIF(beta=0.9, threshold=1.0),
    )
    x = (torch.rand(1, 64) < 0.2).float()

    reset_all(net)
    with EnergyCounter(net, mode="spiking") as c:
        for _ in range(6):
            net(x)

    report = c.report(timesteps=6)
    assert len(report.layers) == 2, "should hook both Linear layers, not the LIFs"
    assert report.total_synops > 0
    assert 0.0 <= report.mean_density <= 1.0
    assert report.energy_pj == pytest.approx(report.total_synops * PJ_PER_SOP)


def test_hooks_are_removed_on_exit():
    """Leaked hooks would silently corrupt every later measurement."""
    model = nn.Linear(10, 5)
    counter = EnergyCounter(model, mode="dense")
    with counter:
        assert len(counter._handles) == 1
    assert counter._handles == []


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        EnergyCounter(nn.Linear(2, 2), mode="quantum")
