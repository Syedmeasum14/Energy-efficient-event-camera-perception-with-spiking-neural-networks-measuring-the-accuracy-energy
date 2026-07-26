"""Paired CNN / SNN classifiers for the energy-accuracy comparison.

THE DESIGN CONSTRAINT THAT MATTERS
----------------------------------
Both models must have *identical* architecture -- same channels, same kernels,
same layer count, same parameter count. The ONLY difference is that ReLU is
replaced by a LIF neuron and the forward pass loops over timesteps.

This is not tidiness, it is the experimental design. If the CNN and SNN differed
in depth or width, any energy difference could be attributed to the architecture
rather than to spiking, and the whole comparison would be worthless. A reviewer
will check this first.

Used unchanged in Rung 1 (N-MNIST, 10 classes) and Rung 2 (N-CARS, 2 classes).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.lif import LIF, StatefulModule, reset_all


class BNTT(StatefulModule):
    """Batch Normalization Through Time (Kim & Panda, 2021).

    WHY PLAIN BATCHNORM FAILS IN AN SNN
    -----------------------------------
    BatchNorm keeps ONE set of running statistics, used at eval time. An SNN's
    activation distribution changes at every timestep: early steps are sparse
    because membranes are still charging, later steps are dense. One mean and
    variance cannot describe all T distributions, so eval-mode normalisation is
    wrong at every single step.

    Measured on N-MNIST with plain BatchNorm -- same weights, same data, only
    the BN mode differing:

        batch statistics   (train mode)   93.36%
        running statistics (eval  mode)   65.62%

    A 28-point gap that has nothing to do with what the network learned. It
    looks exactly like catastrophic overfitting, which is what makes it such an
    expensive bug to chase.

    THE FIX
    -------
    Keep a separate BatchNorm per timestep. Each one then sees a consistent
    distribution and its running statistics are meaningful. Costs T x the BN
    parameters (negligible -- 2 x C floats each) and nothing at inference.
    """

    def __init__(self, num_features: int, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(num_steps)])
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp rather than wrap: running past num_steps is a caller error, but
        # silently reusing step 0's statistics would be far harder to notice
        # than reusing the last step's.
        idx = min(self.step, self.num_steps - 1)
        out = self.bns[idx](x)
        self.step += 1
        return out


class ConvBlock(nn.Module):
    """conv -> (norm) -> activation -> pool. Shared skeleton for both models.

    `norm` is a plain BatchNorm2d for the CNN and a BNTT for the SNN. Parameter
    counts differ by (T-1) x 2 x C as a result, which `count_parameters` reports
    and the tests account for explicitly.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: nn.Module,
        use_norm: bool = True,
        pool: bool = True,
        num_steps: int | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_norm)
        # Normalisation is what makes the SNN trainable at all: it keeps the
        # input current to each LIF layer in a range where the threshold is
        # actually reachable. Without it, deep SNNs go silent -- exactly the
        # "DEAD" rows in scripts/demo_energy.py.
        if not use_norm:
            self.norm: nn.Module = nn.Identity()
        elif num_steps is not None:
            self.norm = BNTT(out_ch, num_steps)
        else:
            self.norm = nn.BatchNorm2d(out_ch)
        self.act = activation
        # MaxPool is deliberate for the SNN: it is the only common pooling op
        # that preserves binary outputs. AvgPool would emit {0, .25, .5, .75, 1},
        # so the next conv would perform real multiplications and the SynOps
        # energy accounting would silently become a lie.
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.norm(self.conv(x))))


class CNNClassifier(nn.Module):
    """Dense baseline. Consumes one collapsed representation (e.g. a voxel grid).

    Input:  (B, in_channels, H, W)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        width: int = 16,
        num_blocks: int = 3,
    ):
        super().__init__()
        chans = [in_channels] + [width * (2**i) for i in range(num_blocks)]
        self.blocks = nn.Sequential(
            *[
                ConvBlock(chans[i], chans[i + 1], nn.ReLU())
                for i in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(chans[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


class SNNClassifier(nn.Module):
    """Spiking counterpart. Architecturally identical to CNNClassifier.

    Input:  (T, B, in_channels, H, W) -- a binary spike tensor
    Output: (B, num_classes) logits, accumulated over T timesteps

    HOW THE OUTPUT IS READ OUT
    --------------------------
    The final layer does not spike. We accumulate its membrane current across
    timesteps and treat the sum as logits ("rate coding" of the decision).
    Letting the output layer spike too would quantise the logits to integers and
    throw away gradient resolution at the loss, which hurts accuracy for no
    energy benefit -- the last layer is a negligible fraction of total ops.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        width: int = 16,
        num_blocks: int = 3,
        beta: float = 0.95,
        threshold: float = 0.5,
        learn_beta: bool = True,
        surrogate: str = "atan",
        num_steps: int | None = None,
    ):
        super().__init__()
        # num_steps enables BNTT (per-timestep BatchNorm). Strongly recommended:
        # plain BatchNorm costs ~28 accuracy points on N-MNIST, see BNTT above.
        self.num_steps = num_steps
        chans = [in_channels] + [width * (2**i) for i in range(num_blocks)]
        self.blocks = nn.Sequential(
            *[
                ConvBlock(
                    chans[i],
                    chans[i + 1],
                    LIF(
                        beta=beta,
                        threshold=threshold,
                        learn_beta=learn_beta,
                        surrogate=surrogate,
                    ),
                    num_steps=num_steps,
                )
                for i in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(chans[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is (T, B, C, H, W). Membrane state persists across the T loop."""
        if self.num_steps is not None and x.shape[0] != self.num_steps:
            raise ValueError(
                f"model was built for num_steps={self.num_steps} (BNTT has that "
                f"many BatchNorms) but got input with {x.shape[0]} timesteps"
            )
        # Resets membrane potentials AND every BNTT's timestep counter.
        reset_all(self)

        logits = 0.0
        for t in range(x.shape[0]):
            # Each timestep is a full forward pass through the same weights.
            # State (membrane potential) is what carries information forward,
            # not the activations -- this is the fundamental difference from
            # running a CNN on T frames independently.
            logits = logits + self.head(self.blocks(x[t]))

        # Mean over timesteps keeps logit magnitude independent of T, so the
        # learning rate does not need retuning when T changes.
        return logits / x.shape[0]


def build_model(kind: str, **kwargs) -> nn.Module:
    """Factory so configs can select a model by name."""
    if kind == "cnn":
        return CNNClassifier(**kwargs)
    if kind == "snn":
        return SNNClassifier(**kwargs)
    raise ValueError(f"unknown model kind: {kind!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
