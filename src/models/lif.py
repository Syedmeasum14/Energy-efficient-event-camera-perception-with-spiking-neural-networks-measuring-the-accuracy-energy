"""Leaky Integrate-and-Fire neurons with surrogate gradients, from scratch.

WHY WRITE THIS BY HAND
----------------------
`snntorch` already provides all of it. We implement it once anyway, because
when a deep SNN stops learning in Rung 3 the cause is almost always one of
three things -- the membrane decayed too fast, the neurons stopped firing, or
the surrogate gradient vanished -- and none of those are debuggable if the
neuron is a black box.

THE MODEL
---------
A biological neuron accumulates incoming charge on its membrane. The charge
leaks away over time. When the membrane potential crosses a threshold, the
neuron emits a spike and resets. That is the whole idea. Discretised:

    V[t] = beta * V[t-1]  +  I[t]  -  reset
    S[t] = 1 if V[t] >= threshold else 0

where
    V     membrane potential (the neuron's internal state, carried across time)
    I[t]  input current at timestep t -- for us, the output of a conv layer
    beta  decay/leak factor in [0, 1). beta=0.95 keeps 95% of V each step.
    S[t]  the output spike: exactly 0 or 1, never anything in between

Note what `beta` buys: it is the *only* reason the neuron has memory. With
beta=0 the neuron forgets everything each step and the SNN degenerates into a
stateless network applied frame-by-frame. With beta close to 1 it integrates
over long windows but responds sluggishly.

THE PROBLEM: SPIKES ARE NOT DIFFERENTIABLE
------------------------------------------
S[t] is a Heaviside step function of V. Its derivative is zero everywhere,
and undefined (infinite) exactly at the threshold:

        S                            dS/dV
        |     ______                 |     |
      1 |    |                       |     |  <- infinite spike at 0
        |    |                       |     |
      0 |____|                     0 |_____|_____   <- zero everywhere else
             V=thresh                      V=thresh

Backpropagate through that and every gradient in the network becomes zero. The
network cannot learn. This is *the* central obstacle in training SNNs.

THE FIX: SURROGATE GRADIENTS
----------------------------
Use the true step function on the forward pass -- so spikes stay strictly
binary, which is what the energy argument depends on -- but substitute a smooth
approximation of its derivative on the backward pass. The forward and backward
passes deliberately disagree. That is not a bug; it is the technique
(Neftci et al., 2019).

Below, `SpikeFunction` does exactly this: `forward` returns a hard step,
`backward` returns the derivative of a smooth stand-in.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class StatefulModule(nn.Module):
    """Base for modules carrying per-sample state that must be cleared.

    Anything holding state across timesteps -- membrane potential in a LIF, the
    timestep index in a BNTT -- must be reset before each new sample, or state
    leaks between unrelated samples. Subclassing this makes `reset_all()` find
    the module automatically, so a new stateful layer cannot be forgotten.
    """

    def reset(self) -> None:  # pragma: no cover - overridden by subclasses
        raise NotImplementedError


class SpikeFunction(torch.autograd.Function):
    """Heaviside step forward, smooth surrogate derivative backward.

    Two surrogates are provided. Both are bell-shaped curves centred on the
    threshold, encoding the intuition: "a neuron sitting near its threshold is
    the one whose weights most affect whether it fires, so give it the biggest
    gradient. A neuron far above or below barely cares."
    """

    @staticmethod
    def forward(ctx, v_shifted: torch.Tensor, alpha: float, kind: str) -> torch.Tensor:
        # v_shifted is (V - threshold), so the decision boundary sits at 0.
        ctx.save_for_backward(v_shifted)
        ctx.alpha = alpha
        ctx.kind = kind
        # The hard step. Output is exactly 0.0 or 1.0 -- this is what makes the
        # downstream SynOps count (and therefore the energy claim) legitimate.
        return (v_shifted >= 0).to(v_shifted.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_shifted,) = ctx.saved_tensors
        alpha = ctx.alpha

        if ctx.kind == "atan":
            # Surrogate:  S ~= (1/pi) * arctan(pi * alpha * x) + 1/2
            # Derivative: dS/dx = alpha / (1 + (pi * alpha * x)^2)
            # Heavy tails: neurons far from threshold still get a small gradient,
            # which makes this the more forgiving choice for deep nets.
            grad = alpha / (1.0 + (math.pi * alpha * v_shifted) ** 2)
        elif ctx.kind == "fast_sigmoid":
            # Derivative of x / (1 + alpha*|x|), from Zenke & Ganguli (2018).
            # Cheaper, but the tails vanish faster.
            grad = alpha / (1.0 + alpha * v_shifted.abs()) ** 2
        else:
            raise ValueError(f"unknown surrogate: {ctx.kind!r}")

        # Chain rule. alpha and kind are non-tensor args, so they get None.
        return grad_output * grad, None, None


def spike(v_shifted: torch.Tensor, alpha: float = 2.0, kind: str = "atan") -> torch.Tensor:
    """Convenience wrapper around SpikeFunction.apply."""
    return SpikeFunction.apply(v_shifted, alpha, kind)


class LIF(StatefulModule):
    """A layer of Leaky Integrate-and-Fire neurons.

    Stateful across timesteps. Call `reset()` at the start of every sample,
    otherwise membrane potential leaks between unrelated samples and your
    batches become silently correlated -- a bug that shows up as suspiciously
    good training accuracy and terrible test accuracy.

    Usage inside a network:

        lif = LIF(beta=0.95)
        lif.reset()
        for t in range(num_steps):
            current = conv(x[t])       # input current for this timestep
            spikes  = lif(current)     # membrane updates, maybe fires
    """

    def __init__(
        self,
        beta: float = 0.95,
        threshold: float = 1.0,
        learn_beta: bool = False,
        reset_mode: str = "subtract",
        surrogate: str = "atan",
        alpha: float = 2.0,
    ):
        super().__init__()

        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if reset_mode not in ("subtract", "zero"):
            raise ValueError(f"reset_mode must be 'subtract' or 'zero', got {reset_mode!r}")

        # beta can be learned. Letting the network choose its own time constant
        # per layer usually helps, at the cost of one more thing that can go
        # wrong. We store it raw and sigmoid it so it stays in (0, 1) no matter
        # what the optimizer does -- an unconstrained beta drifting above 1.0
        # makes the membrane explode.
        if learn_beta:
            beta_logit = math.log(beta / (1.0 - beta))  # inverse sigmoid
            self.beta_logit = nn.Parameter(torch.tensor(beta_logit))
        else:
            self.register_buffer("beta_logit", torch.tensor(math.log(beta / (1.0 - beta))))

        self.threshold = threshold
        self.reset_mode = reset_mode
        self.surrogate = surrogate
        self.alpha = alpha

        # Membrane potential. None until the first forward pass, because we do
        # not know the tensor shape until we see input.
        self.v: torch.Tensor | None = None

        # Diagnostics: how many spikes this layer emitted since the last reset.
        # Consumed by the energy counter, and the first thing to check when a
        # network will not train.
        self.spike_count: float = 0.0
        self.step_count: int = 0

    # Clamp bound on the raw logit. sigmoid(8) = 0.99966, sigmoid(-8) = 0.00034.
    # Without this, a logit the optimizer pushes past ~17 saturates sigmoid to
    # *exactly* 1.0 in float32 -- beta = 1.0 means a neuron that never leaks, so
    # membrane potential integrates without bound and the layer blows up. The
    # clamp keeps beta strictly inside (0, 1) while leaving gradients intact
    # across the entire useful range.
    _LOGIT_CLAMP = 8.0

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_logit.clamp(-self._LOGIT_CLAMP, self._LOGIT_CLAMP))

    def reset(self) -> None:
        """Clear membrane state and diagnostics. Call before every sample."""
        self.v = None
        self.spike_count = 0.0
        self.step_count = 0

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """Advance one timestep. `current` is the input I[t]."""
        if self.v is None:
            self.v = torch.zeros_like(current)

        # 1. Leak and integrate:  V = beta*V + I
        self.v = self.beta * self.v + current

        # 2. Fire where V crossed threshold. Subtracting the threshold first
        #    puts the decision boundary at 0, which is what SpikeFunction wants.
        s = spike(self.v - self.threshold, self.alpha, self.surrogate)

        # 3. Reset. Applied IMMEDIATELY, within the same timestep that fired.
        #
        #    This is a convention choice, and it is worth knowing that snntorch
        #    defaults the other way (`reset_delay=True`), applying the reset at
        #    the start of the *next* step to mimic the one-cycle latency of real
        #    neuromorphic hardware. Both are used in the literature; they give
        #    identical spike trains but membrane traces offset by one threshold.
        #    We reset immediately because it is simpler to reason about when
        #    debugging. tests/test_snntorch_equivalence.py pins this by
        #    comparing against snntorch with reset_delay=False.
        if self.reset_mode == "subtract":
            # "Soft" reset: subtract the threshold, keeping any excess charge.
            # Preserves information about *how far* over threshold the neuron
            # went, which matters for strong inputs. Usual choice for deep SNNs.
            #
            # s.detach() is deliberate: without it, the reset path contributes a
            # second, spurious route for gradients to flow back through the
            # spike, which destabilises training. Standard practice.
            self.v = self.v - s.detach() * self.threshold
        else:
            # "Hard" reset: dump the membrane to zero. More biologically
            # faithful, discards the excess.
            self.v = self.v * (1.0 - s.detach())

        # 4. Diagnostics.
        self.spike_count += float(s.detach().sum().item())
        self.step_count += 1

        return s

    def firing_rate(self) -> float:
        """Mean spikes per neuron per timestep since the last reset.

        The single most useful number when debugging an SNN:
          ~0.0        the network is dead. beta too low, threshold too high, or
                      the surrogate gradient vanished.
          0.02-0.20   healthy. Sparse enough for the energy argument to hold.
          >0.5        barely sparse. Accuracy may be fine but you have given up
                      the entire reason for using an SNN.
        """
        if self.step_count == 0 or self.v is None:
            return 0.0
        neurons_per_step = self.v.numel()
        return self.spike_count / (neurons_per_step * self.step_count)

    def extra_repr(self) -> str:
        return (
            f"beta={self.beta.item():.3f}, threshold={self.threshold}, "
            f"reset={self.reset_mode}, surrogate={self.surrogate}"
        )


def reset_all(module: nn.Module) -> None:
    """Reset every stateful layer in a network. Call at the start of each sample.

    Walks StatefulModule rather than LIF specifically, so BNTT (and anything
    stateful added later) is reset too without touching this function.
    """
    for m in module.modules():
        if isinstance(m, StatefulModule):
            m.reset()


def collect_firing_rates(module: nn.Module) -> dict[str, float]:
    """Per-layer firing rates, for diagnostics and logging."""
    return {
        name: m.firing_rate()
        for name, m in module.named_modules()
        if isinstance(m, LIF)
    }
