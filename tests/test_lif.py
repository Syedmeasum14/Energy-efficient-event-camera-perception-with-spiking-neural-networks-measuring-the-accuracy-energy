"""Tests for the LIF neuron and surrogate gradient.

Each test here guards a specific failure mode that is otherwise very hard to
diagnose, because a broken SNN and a merely hard-to-train SNN look identical
from the loss curve.
"""

import math

import pytest
import torch

from src.models.lif import LIF, collect_firing_rates, reset_all, spike


# --------------------------------------------------------------------------
# The surrogate gradient. If these fail, nothing downstream can learn.
# --------------------------------------------------------------------------

def test_forward_is_strictly_binary():
    """Spikes must be exactly 0 or 1 -- the energy argument depends on it."""
    v = torch.randn(1000, requires_grad=True)
    s = spike(v)
    assert torch.all((s == 0.0) | (s == 1.0))


def test_forward_matches_heaviside():
    v = torch.tensor([-1.0, -0.001, 0.0, 0.001, 1.0], requires_grad=True)
    s = spike(v)
    assert torch.equal(s, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_gradient_is_nonzero():
    """THE test. A true Heaviside has zero gradient everywhere, which kills
    learning. The whole point of the surrogate is that this is not zero."""
    v = torch.randn(500, requires_grad=True)
    spike(v).sum().backward()
    assert v.grad is not None
    assert v.grad.abs().sum() > 0


def test_gradient_peaks_at_threshold():
    """Neurons near threshold should receive the largest gradient -- they are
    the ones whose weights actually determine whether a spike happens."""
    v = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    spike(v).sum().backward()
    g = v.grad
    assert g[2] == g.max()               # peak at 0 (== threshold)
    assert g[1] > g[0] and g[3] > g[4]   # decays away from it


def test_gradient_is_symmetric():
    v = torch.tensor([-1.5, -0.3, 0.3, 1.5], requires_grad=True)
    spike(v).sum().backward()
    assert v.grad[0] == pytest.approx(v.grad[3].item(), rel=1e-6)
    assert v.grad[1] == pytest.approx(v.grad[2].item(), rel=1e-6)


@pytest.mark.parametrize("kind", ["atan", "fast_sigmoid"])
def test_both_surrogates_produce_finite_gradients(kind):
    # Note: build the leaf tensor first, then set requires_grad. Doing
    # `torch.randn(..., requires_grad=True) * 5` makes v a *non-leaf* tensor,
    # whose .grad is never populated.
    v = (torch.randn(200) * 5).requires_grad_(True)
    spike(v, kind=kind).sum().backward()
    assert torch.isfinite(v.grad).all()
    assert v.grad.abs().sum() > 0


def test_atan_gradient_matches_closed_form():
    """Guard against a typo in the derivative -- silent and devastating."""
    alpha = 2.0
    v = torch.tensor([0.0, 0.25, -0.5], requires_grad=True)
    spike(v, alpha=alpha, kind="atan").sum().backward()
    expected = alpha / (1.0 + (math.pi * alpha * v.detach()) ** 2)
    assert torch.allclose(v.grad, expected, atol=1e-6)


def test_unknown_surrogate_raises():
    v = torch.randn(4, requires_grad=True)
    with pytest.raises(ValueError, match="unknown surrogate"):
        spike(v, kind="nonsense").sum().backward()


# --------------------------------------------------------------------------
# Membrane dynamics
# --------------------------------------------------------------------------

def test_membrane_integrates_over_time():
    """Sub-threshold input must accumulate until it eventually fires."""
    lif = LIF(beta=0.9, threshold=1.0)
    lif.reset()
    current = torch.tensor([0.3])
    fired = [lif(current).item() for _ in range(10)]
    # 0.3 alone never crosses 1.0, but accumulated it must.
    assert sum(fired) > 0, "neuron never fired despite sustained input"


def test_membrane_leaks():
    """After input stops, V must decay toward zero -- that is the 'leaky' part."""
    lif = LIF(beta=0.5, threshold=100.0)  # threshold high so it never fires
    lif.reset()
    lif(torch.tensor([1.0]))
    v_start = lif.v.clone()
    for _ in range(5):
        lif(torch.zeros(1))
    assert lif.v.item() < v_start.item()
    assert lif.v.item() == pytest.approx(v_start.item() * 0.5**5, rel=1e-5)


def test_strong_input_fires_immediately():
    lif = LIF(beta=0.9, threshold=1.0)
    lif.reset()
    assert lif(torch.tensor([5.0])).item() == 1.0


def test_subtract_reset_keeps_excess_charge():
    lif = LIF(beta=1.0 - 1e-9, threshold=1.0, reset_mode="subtract")
    lif.reset()
    lif(torch.tensor([2.5]))
    # Fired once, subtracted 1.0, so ~1.5 of charge should remain.
    assert lif.v.item() == pytest.approx(1.5, abs=1e-6)


def test_zero_reset_discards_charge():
    lif = LIF(beta=0.9, threshold=1.0, reset_mode="zero")
    lif.reset()
    lif(torch.tensor([2.5]))
    assert lif.v.item() == pytest.approx(0.0, abs=1e-6)


def test_reset_clears_state_between_samples():
    """Without this, membrane potential leaks across unrelated samples and
    batches become silently correlated."""
    lif = LIF(beta=0.9, threshold=10.0)
    lif.reset()
    lif(torch.tensor([5.0]))
    assert lif.v.item() > 0
    lif.reset()
    assert lif.v is None


# --------------------------------------------------------------------------
# Parameterisation and diagnostics
# --------------------------------------------------------------------------

def test_beta_stays_in_range_when_learned():
    """A learned beta drifting above 1.0 makes the membrane explode. The
    sigmoid parameterisation must make that impossible."""
    lif = LIF(beta=0.95, learn_beta=True)
    assert isinstance(lif.beta_logit, torch.nn.Parameter)
    assert lif.beta.item() == pytest.approx(0.95, abs=1e-5)

    # Shove the raw parameter to absurd values; beta must stay in (0, 1).
    with torch.no_grad():
        lif.beta_logit.fill_(50.0)
    assert 0.0 < lif.beta.item() < 1.0
    with torch.no_grad():
        lif.beta_logit.fill_(-50.0)
    assert 0.0 < lif.beta.item() < 1.0


def test_learned_beta_receives_gradient():
    lif = LIF(beta=0.9, threshold=0.5, learn_beta=True)
    lif.reset()
    out = sum(lif(torch.tensor([0.4])) for _ in range(5))
    out.backward()
    assert lif.beta_logit.grad is not None
    assert lif.beta_logit.grad.abs().item() > 0


def test_invalid_beta_rejected():
    with pytest.raises(ValueError, match="beta must be"):
        LIF(beta=1.0)
    with pytest.raises(ValueError, match="beta must be"):
        LIF(beta=-0.1)


def test_invalid_reset_mode_rejected():
    with pytest.raises(ValueError, match="reset_mode"):
        LIF(reset_mode="explode")


def test_firing_rate_bounds():
    """Firing rate is the key diagnostic; it must be a sane fraction."""
    lif = LIF(beta=0.9, threshold=1.0)

    # Never fires -> rate 0.
    lif.reset()
    for _ in range(10):
        lif(torch.zeros(100))
    assert lif.firing_rate() == 0.0

    # Always fires -> rate 1.
    lif.reset()
    for _ in range(10):
        lif(torch.full((100,), 100.0))
    assert lif.firing_rate() == pytest.approx(1.0)


def test_reset_all_and_collect_firing_rates_walk_nested_modules():
    net = torch.nn.Sequential(
        torch.nn.Linear(4, 8), LIF(beta=0.9, threshold=0.1),
        torch.nn.Linear(8, 4), LIF(beta=0.9, threshold=0.1),
    )
    reset_all(net)
    for _ in range(3):
        net(torch.randn(2, 4))

    rates = collect_firing_rates(net)
    assert len(rates) == 2
    assert all(0.0 <= r <= 1.0 for r in rates.values())

    reset_all(net)
    assert all(m.v is None for m in net.modules() if isinstance(m, LIF))


# --------------------------------------------------------------------------
# End-to-end: can a tiny SNN actually learn?
# --------------------------------------------------------------------------

def test_tiny_snn_learns_a_trivial_task():
    """The integration test that matters. If the surrogate gradient is wired up
    correctly, this loss must go down. If it is broken, it will not move."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(10, 20), LIF(beta=0.9, threshold=0.5))
    opt = torch.optim.Adam(net.parameters(), lr=0.05)

    x = torch.randn(16, 10)
    target = torch.ones(16, 20)  # ask every neuron to spike every step

    first_loss, last_loss = None, None
    for _ in range(40):
        reset_all(net)
        out = sum(net(x) for _ in range(4)) / 4.0
        loss = torch.nn.functional.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

    assert last_loss < first_loss * 0.5, (
        f"SNN failed to learn: {first_loss:.4f} -> {last_loss:.4f}"
    )


# --------------------------------------------------------------------------
# Differentiable firing-rate penalty (the Rung 2 sparsity sweep depends on it)
# --------------------------------------------------------------------------

def test_firing_rate_loss_is_differentiable():
    """THE point of this function. The old detached version reported a
    plausible number but produced no gradient, so lambda did nothing."""
    from src.models.lif import firing_rate_loss

    net = torch.nn.Sequential(torch.nn.Linear(8, 16), LIF(beta=0.9, threshold=0.2))
    reset_all(net)
    x = torch.randn(4, 8)
    for _ in range(5):
        net(x)

    loss = firing_rate_loss(net)
    assert loss.requires_grad, "penalty is detached -- lambda would have no effect"
    loss.backward()
    grad = net[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_firing_rate_loss_matches_reported_rate():
    """The differentiable value must agree with the detached diagnostic,
    otherwise the logged firing rate is not what is being optimised."""
    from src.models.lif import collect_firing_rates, firing_rate_loss

    net = torch.nn.Sequential(torch.nn.Linear(8, 16), LIF(beta=0.9, threshold=0.2))
    reset_all(net)
    x = torch.randn(4, 8)
    for _ in range(6):
        net(x)

    reported = sum(collect_firing_rates(net).values()) / 1
    assert firing_rate_loss(net).item() == pytest.approx(reported, abs=1e-5)


def test_firing_rate_loss_zero_without_lif():
    from src.models.lif import firing_rate_loss

    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    assert firing_rate_loss(net).item() == 0.0


def test_penalty_actually_reduces_firing():
    """End-to-end: training with a large lambda must produce a sparser network
    than training without it. This is the mechanism the Pareto sweep rides on."""
    from src.models.lif import collect_firing_rates

    def train(lam):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(20, 40), LIF(beta=0.9, threshold=0.3))
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        x = torch.randn(16, 20)
        for _ in range(30):
            reset_all(net)
            out = sum(net(x) for _ in range(4)) / 4.0
            loss = torch.nn.functional.mse_loss(out, torch.ones_like(out))
            if lam:
                from src.models.lif import firing_rate_loss
                loss = loss + lam * firing_rate_loss(net)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return sum(collect_firing_rates(net).values())

    assert train(5.0) < train(0.0), "sparsity penalty had no effect on firing"


def test_reset_clears_differentiable_accumulator():
    """A stale graph across samples would leak memory and corrupt gradients."""
    from src.models.lif import firing_rate_loss

    lif = LIF(beta=0.9, threshold=0.1)
    lif.reset()
    for _ in range(3):
        lif(torch.randn(2, 5))
    assert lif.rate_sum is not None
    lif.reset()
    assert lif.rate_sum is None
    assert firing_rate_loss(lif).item() == 0.0
