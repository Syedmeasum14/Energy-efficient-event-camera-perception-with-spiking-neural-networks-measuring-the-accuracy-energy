"""Tests for the paired CNN / SNN classifiers.

The parameter-parity test is the important one: it enforces the experimental
design. If the two models ever diverge in capacity, every energy number the
project reports becomes uninterpretable.
"""

import pytest
import torch

from src.engine.energy import EnergyCounter
from src.models.classifier import (
    CNNClassifier,
    SNNClassifier,
    build_model,
    count_parameters,
)
from src.models.lif import LIF, collect_firing_rates

B, C, H, W = 4, 2, 32, 32
T = 6


def test_cnn_output_shape():
    model = CNNClassifier(in_channels=C, num_classes=10)
    assert model(torch.randn(B, C, H, W)).shape == (B, 10)


def test_snn_output_shape():
    model = SNNClassifier(in_channels=C, num_classes=10)
    spikes = (torch.rand(T, B, C, H, W) < 0.1).float()
    assert model(spikes).shape == (B, 10)


def test_cnn_and_snn_have_identical_parameter_counts():
    """THE experimental-design test. Any capacity difference invalidates the
    energy comparison, and this is the first thing a reviewer checks."""
    cnn = CNNClassifier(in_channels=C, num_classes=10, width=16, num_blocks=3)
    snn = SNNClassifier(
        in_channels=C, num_classes=10, width=16, num_blocks=3, learn_beta=False
    )
    assert count_parameters(cnn) == count_parameters(snn)


def test_bntt_keeps_separate_stats_per_timestep():
    """BNTT must use a different BatchNorm at each timestep, and reset its
    counter between samples."""
    from src.models.classifier import BNTT
    from src.models.lif import reset_all as _reset

    bntt = BNTT(num_features=4, num_steps=3)
    assert len(bntt.bns) == 3
    assert bntt.step == 0
    for expected in (1, 2, 3):
        bntt(torch.randn(2, 4, 8, 8))
        assert bntt.step == expected
    _reset(bntt)
    assert bntt.step == 0


def test_bntt_running_stats_differ_across_timesteps():
    """The whole point: each timestep's BN must learn its own distribution."""
    from src.models.classifier import BNTT

    bntt = BNTT(num_features=4, num_steps=3).train()
    bntt(torch.randn(8, 4, 8, 8) * 0.1)   # sparse-ish early step
    bntt(torch.randn(8, 4, 8, 8) * 5.0)   # dense later step
    bntt(torch.randn(8, 4, 8, 8) * 5.0)
    assert not torch.allclose(bntt.bns[0].running_var, bntt.bns[1].running_var)


def test_snn_with_bntt_is_stable_between_train_and_eval():
    """REGRESSION: plain BatchNorm cost 28 accuracy points on N-MNIST because
    eval-mode running stats did not match any timestep's distribution. With
    BNTT, train-mode and eval-mode outputs must stay close."""
    torch.manual_seed(0)
    model = SNNClassifier(in_channels=C, num_classes=10, num_blocks=2, num_steps=T)
    spikes = (torch.rand(T, 16, C, H, W) < 0.15).float()

    # Populate running statistics.
    model.train()
    for _ in range(20):
        model((torch.rand(T, 16, C, H, W) < 0.15).float())

    with torch.no_grad():
        model.train()
        train_out = model(spikes)
        model.eval()
        eval_out = model(spikes)

    # Not identical (batch vs running stats), but the same ballpark.
    diff = (train_out - eval_out).abs().mean().item()
    scale = train_out.abs().mean().item() + 1e-8
    assert diff / scale < 0.5, f"train/eval divergence {diff / scale:.2f} too large"


def test_snn_rejects_wrong_timestep_count_when_bntt_enabled():
    model = SNNClassifier(in_channels=C, num_classes=10, num_blocks=2, num_steps=T)
    with pytest.raises(ValueError, match="was built for num_steps"):
        model((torch.rand(T + 3, 2, C, H, W) < 0.1).float())


def test_learned_beta_adds_exactly_one_param_per_lif():
    """learn_beta is the only legitimate source of extra parameters, and it must
    be accounted for explicitly rather than discovered later."""
    plain = SNNClassifier(num_blocks=3, learn_beta=False)
    learned = SNNClassifier(num_blocks=3, learn_beta=True)
    assert count_parameters(learned) - count_parameters(plain) == 3


def test_snn_is_stateless_between_calls():
    """Two identical inputs must give identical outputs. If membrane state
    leaked across samples, results would silently depend on batch ordering."""
    torch.manual_seed(0)
    model = SNNClassifier(in_channels=C, num_classes=10).eval()
    spikes = (torch.rand(T, B, C, H, W) < 0.1).float()
    with torch.no_grad():
        assert torch.allclose(model(spikes), model(spikes))


def test_snn_output_depends_on_spike_timing():
    """If reversing the time axis changed nothing, the model would not be using
    temporal information at all and the SNN would be pointless."""
    torch.manual_seed(0)
    model = SNNClassifier(in_channels=C, num_classes=10).eval()
    spikes = (torch.rand(T, B, C, H, W) < 0.15).float()
    with torch.no_grad():
        forward = model(spikes)
        reversed_ = model(torch.flip(spikes, dims=[0]))
    assert not torch.allclose(forward, reversed_, atol=1e-4)


def test_snn_gradients_reach_the_first_layer():
    """Surrogate gradients must survive backprop through every LIF layer and
    through time. Vanishing here is the classic deep-SNN failure."""
    model = SNNClassifier(in_channels=C, num_classes=10)
    spikes = (torch.rand(T, B, C, H, W) < 0.15).float()
    model(spikes).sum().backward()

    first_conv = model.blocks[0].conv
    assert first_conv.weight.grad is not None
    assert first_conv.weight.grad.abs().sum() > 0, "gradient died before layer 1"


def test_snn_fires_at_a_healthy_rate():
    """A silent network reports fake energy savings. BatchNorm plus a 0.5
    threshold should keep every layer alive on sparse input."""
    torch.manual_seed(0)
    model = SNNClassifier(in_channels=C, num_classes=10, threshold=0.5)
    spikes = (torch.rand(T, B, C, H, W) < 0.05).float()
    model(spikes)

    rates = collect_firing_rates(model)
    assert len(rates) == 3
    for name, rate in rates.items():
        assert rate > 0.0, f"layer {name} is dead (0% firing)"
        assert rate < 0.9, f"layer {name} fires at {rate:.1%} -- not sparse"


def test_snn_uses_less_energy_than_cnn():
    """End-to-end integration of everything built so far."""
    torch.manual_seed(0)
    cnn = CNNClassifier(in_channels=C, num_classes=10)
    snn = SNNClassifier(in_channels=C, num_classes=10)

    voxel = torch.randn(1, C, H, W)
    spikes = (torch.rand(T, 1, C, H, W) < 0.05).float()

    with EnergyCounter(cnn, mode="dense") as counter:
        cnn(voxel)
    dense = counter.report()

    with EnergyCounter(snn, mode="spiking") as counter:
        snn(spikes)
    spiking = counter.report(timesteps=T)

    assert spiking.energy_uj < dense.energy_uj
    assert dense.total_macs > 0 and spiking.total_synops > 0


def test_energy_counter_ignores_lif_layers():
    """LIF layers carry no weights; only conv/linear should be counted."""
    snn = SNNClassifier(in_channels=C, num_classes=10, num_blocks=3)
    spikes = (torch.rand(T, 1, C, H, W) < 0.1).float()
    with EnergyCounter(snn, mode="spiking") as counter:
        snn(spikes)
    # 3 convs + 1 linear head, no LIFs.
    assert len(counter.report().layers) == 4


def test_build_model_factory():
    assert isinstance(build_model("cnn", num_classes=2), CNNClassifier)
    assert isinstance(build_model("snn", num_classes=2), SNNClassifier)
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model("transformer")


@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_depth_is_configurable(num_blocks):
    model = SNNClassifier(in_channels=C, num_classes=10, num_blocks=num_blocks)
    spikes = (torch.rand(T, B, C, 64, 64) < 0.1).float()
    assert model(spikes).shape == (B, 10)
    assert len([m for m in model.modules() if isinstance(m, LIF)]) == num_blocks
