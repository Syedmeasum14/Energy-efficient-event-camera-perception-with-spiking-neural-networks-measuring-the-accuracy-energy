"""Energy accounting: MACs for dense networks, SynOps for spiking ones.

THE ARGUMENT THIS FILE MAKES
----------------------------
A dense CNN computes, for every layer, a full matrix multiply. Every output
element costs a multiply-accumulate (MAC), regardless of the input. The cost is
fixed by the architecture: a zero input costs exactly as much as a busy one.

A spiking network is different in two ways:

1. **No multiplication.** A spike is binary. `w * 1` is just `w`, so a synaptic
   operation is an *accumulate*, not a multiply-accumulate. On the standard 45nm
   32-bit accounting (Horowitz, ISSCC 2014) that is 0.9 pJ versus 4.6 pJ --
   about 5x cheaper per operation.

2. **No work where there is no spike.** `w * 0` contributes nothing, and
   neuromorphic hardware simply does not perform it. So cost scales with the
   *firing rate*, not with the layer size. At a 10% firing rate, a layer does
   roughly a tenth of the operations.

Multiply those together and a sparse SNN can be 20-50x cheaper than the
equivalent CNN. That is the entire claim, and the job of this file is to measure
it honestly rather than assert it.

    E_cnn = MACs x 4.6 pJ
    E_snn = SynOps x 0.9 pJ    where  SynOps = sum_layers MACs_l x rate_l x T

WHAT THIS IS NOT
----------------
An *estimate*, not a measurement. Real hardware energy includes memory traffic,
which frequently dominates arithmetic, plus spike routing overhead that this
model ignores entirely. Every number produced here must be reported as
"estimated energy under the Horowitz 45nm model". Claiming measured energy
without silicon is the fastest way to lose a reviewer's trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

# Horowitz (2014), 45nm, 32-bit operations.
PJ_PER_MAC = 4.6
PJ_PER_SOP = 0.9

# Layer types that carry weights and therefore dominate the operation count.
# Pooling, normalisation and activations are ignored: standard practice in the
# SNN literature, and they are 1-2 orders of magnitude cheaper.
COUNTED_LAYERS = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)


@dataclass
class LayerStats:
    """Per-layer accounting."""

    name: str
    kind: str
    dense_macs: int = 0          # MACs if every input were nonzero
    synops: float = 0.0          # actual accumulates, weighted by input sparsity
    input_nonzero_frac: float = 0.0
    calls: int = 0               # forward passes seen (== timesteps for an SNN)

    @property
    def mean_input_density(self) -> float:
        return self.input_nonzero_frac / self.calls if self.calls else 0.0


@dataclass
class EnergyReport:
    """Aggregate result of a measurement run."""

    mode: str                    # "dense" or "spiking"
    total_macs: int = 0
    total_synops: float = 0.0
    timesteps: int = 1
    layers: dict[str, LayerStats] = field(default_factory=dict)

    @property
    def energy_pj(self) -> float:
        if self.mode == "dense":
            return self.total_macs * PJ_PER_MAC
        return self.total_synops * PJ_PER_SOP

    @property
    def energy_uj(self) -> float:
        return self.energy_pj / 1e6

    @property
    def mean_density(self) -> float:
        """Mean input density across counted layers -- the SNN's firing rate as
        seen by the weights, which is what actually drives cost."""
        if not self.layers:
            return 0.0
        return sum(s.mean_input_density for s in self.layers.values()) / len(self.layers)

    def summary(self) -> str:
        if self.mode == "dense":
            head = f"dense: {self.total_macs / 1e6:.2f} M MAC"
        else:
            head = (
                f"spiking: {self.total_synops / 1e6:.2f} M SynOp "
                f"over {self.timesteps} steps, mean input density "
                f"{self.mean_density * 100:.1f}%"
            )
        return f"{head}  ->  {self.energy_uj:.2f} uJ/sample (estimated)"


def _dense_macs(module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> int:
    """MACs for one forward pass, assuming a fully dense input.

    Conv2d: every output element is a dot product over (in_channels/groups) x
    kernel area, so MACs = numel(output) x (C_in/groups) x kH x kW. Batch is
    divided out so the figure is per-sample.
    """
    if isinstance(module, nn.Linear):
        batch = inp.shape[0] if inp.dim() > 1 else 1
        return int(out.numel() / batch * module.in_features)

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        batch = inp.shape[0]
        kernel_ops = module.in_channels // module.groups
        for k in module.kernel_size:
            kernel_ops *= k
        return int(out.numel() / batch * kernel_ops)

    return 0


class EnergyCounter:
    """Measures MACs and SynOps by hooking weighted layers during a forward pass.

    Why hooks rather than a static analysis: SynOps depend on the *actual* input
    sparsity, which is a property of the data, not the architecture. A static
    count cannot know that a night-time driving scene produces a third as many
    events as a daytime one -- and that difference is precisely what the project
    sets out to measure.

    Usage:

        counter = EnergyCounter(model, mode="spiking")
        with counter:
            reset_all(model)
            for t in range(num_steps):
                model(x[t])
        print(counter.report(timesteps=num_steps).summary())
    """

    def __init__(self, model: nn.Module, mode: str = "dense"):
        if mode not in ("dense", "spiking"):
            raise ValueError(f"mode must be 'dense' or 'spiking', got {mode!r}")
        self.model = model
        self.mode = mode
        self.stats: dict[str, LayerStats] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            inp = inputs[0]
            macs = _dense_macs(module, inp, output)

            stats = self.stats.setdefault(
                name, LayerStats(name=name, kind=type(module).__name__)
            )

            # Fraction of input elements that are nonzero. For an SNN layer fed
            # by LIF neurons this is exactly the input firing rate; for a dense
            # network it is ~1.0 (post-ReLU activations are sparse-ish but the
            # hardware computes them anyway, so we do not credit that).
            density = float((inp != 0).float().mean().item())

            stats.dense_macs = macs
            stats.input_nonzero_frac += density
            stats.calls += 1

            if self.mode == "spiking":
                # Only spikes trigger work. Accumulate across timesteps.
                stats.synops += macs * density
            else:
                # Dense hardware pays full price regardless of input content.
                stats.synops += macs

        return hook

    def __enter__(self) -> "EnergyCounter":
        for name, module in self.model.named_modules():
            if isinstance(module, COUNTED_LAYERS):
                self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        self.stats.clear()

    def report(self, timesteps: int = 1) -> EnergyReport:
        report = EnergyReport(mode=self.mode, timesteps=timesteps, layers=dict(self.stats))
        # dense_macs is the per-forward-pass cost; a dense model runs once.
        report.total_macs = sum(s.dense_macs for s in self.stats.values())
        report.total_synops = sum(s.synops for s in self.stats.values())
        return report


def compare(dense: EnergyReport, spiking: EnergyReport) -> str:
    """Human-readable head-to-head. This is the project's headline number."""
    e_dense, e_spike = dense.energy_uj, spiking.energy_uj
    ratio = e_dense / e_spike if e_spike > 0 else float("inf")
    op_ratio = (
        dense.total_macs / spiking.total_synops if spiking.total_synops > 0 else float("inf")
    )
    return (
        f"  dense   {dense.total_macs / 1e6:8.2f} M MAC   {e_dense:8.2f} uJ\n"
        f"  spiking {spiking.total_synops / 1e6:8.2f} M SOP   {e_spike:8.2f} uJ\n"
        f"  -> {op_ratio:.1f}x fewer ops, {ratio:.1f}x less estimated energy\n"
        f"  (mean input density {spiking.mean_density * 100:.1f}%, "
        f"{spiking.timesteps} timesteps)"
    )
