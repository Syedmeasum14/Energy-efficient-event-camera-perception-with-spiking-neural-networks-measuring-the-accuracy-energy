"""Validate the from-scratch LIF against snntorch's reference implementation.

WHY THIS FILE EXISTS
--------------------
`src/models/lif.py` was written by hand so the dynamics are inspectable. That is
only worth anything if the hand-written version is actually *correct*. snntorch
is the community-standard implementation; agreeing with it to numerical
precision is the strongest cheap evidence available that the membrane update,
the reset rule and the surrogate gradient are all right.

If these tests fail after an edit to lif.py, trust snntorch and fix lif.py.
"""

import pytest
import torch

snntorch = pytest.importorskip("snntorch", reason="snntorch not installed")
from snntorch import surrogate  # noqa: E402

from src.models.lif import LIF, spike  # noqa: E402


BETA = 0.9
THRESHOLD = 1.0


def test_membrane_dynamics_match_snntorch():
    """Same input, same beta, same threshold -> same spikes and same membrane.

    snntorch's Leaky uses the soft ("subtract") reset by default, matching our
    reset_mode='subtract'.
    """
    ours = LIF(beta=BETA, threshold=THRESHOLD, reset_mode="subtract")
    theirs = snntorch.Leaky(
        beta=BETA, threshold=THRESHOLD, reset_mechanism="subtract", reset_delay=False
    )

    ours.reset()
    mem = theirs.init_leaky()

    torch.manual_seed(0)
    currents = torch.randn(20, 8) * 0.6

    for t in range(currents.shape[0]):
        s_ours = ours(currents[t])
        s_theirs, mem = theirs(currents[t], mem)

        assert torch.equal(s_ours, s_theirs), f"spikes diverged at timestep {t}"
        assert torch.allclose(ours.v, mem, atol=1e-6), f"membrane diverged at step {t}"


def test_zero_reset_matches_snntorch():
    ours = LIF(beta=BETA, threshold=THRESHOLD, reset_mode="zero")
    theirs = snntorch.Leaky(
        beta=BETA, threshold=THRESHOLD, reset_mechanism="zero", reset_delay=False
    )

    ours.reset()
    mem = theirs.init_leaky()

    torch.manual_seed(1)
    currents = torch.randn(20, 8) * 0.8

    for t in range(currents.shape[0]):
        s_ours = ours(currents[t])
        s_theirs, mem = theirs(currents[t], mem)
        assert torch.equal(s_ours, s_theirs), f"spikes diverged at timestep {t}"
        assert torch.allclose(ours.v, mem, atol=1e-6), f"membrane diverged at step {t}"


def test_atan_surrogate_gradient_matches_snntorch():
    """Our atan surrogate must agree with snntorch's, up to their alpha
    convention. snntorch parameterises atan_surrogate with alpha=2.0 as

        grad = alpha / 2 / (1 + (pi/2 * alpha * x)^2)

    while ours is

        grad = alpha / (1 + (pi * alpha * x)^2)

    These coincide when their alpha = 2 * our alpha. Verifying the mapping is
    the point: it confirms both are the derivative of the same arctan surrogate,
    just scaled differently.
    """
    x = torch.linspace(-2, 2, 41)

    ours_in = x.clone().requires_grad_(True)
    spike(ours_in, alpha=1.0, kind="atan").sum().backward()

    theirs_fn = surrogate.atan(alpha=2.0)
    theirs_in = x.clone().requires_grad_(True)
    theirs_fn(theirs_in).sum().backward()

    assert torch.allclose(ours_in.grad, theirs_in.grad, atol=1e-5)


def test_forward_pass_is_binary_in_both():
    x = torch.randn(100, requires_grad=True)
    ours = spike(x)
    theirs = surrogate.atan(alpha=2.0)(x)
    assert torch.all((ours == 0) | (ours == 1))
    assert torch.all((theirs == 0) | (theirs == 1))
    assert torch.equal(ours, theirs)


def test_learned_beta_still_matches_at_init():
    """learn_beta changes the parameterisation, not the dynamics."""
    ours = LIF(beta=BETA, threshold=THRESHOLD, learn_beta=True)
    theirs = snntorch.Leaky(
        beta=BETA, threshold=THRESHOLD, reset_mechanism="subtract", reset_delay=False
    )

    ours.reset()
    mem = theirs.init_leaky()

    torch.manual_seed(2)
    currents = torch.randn(10, 4) * 0.7

    for t in range(currents.shape[0]):
        s_ours = ours(currents[t])
        s_theirs, mem = theirs(currents[t], mem)
        assert torch.equal(s_ours, s_theirs)
        assert torch.allclose(ours.v, mem, atol=1e-5)
