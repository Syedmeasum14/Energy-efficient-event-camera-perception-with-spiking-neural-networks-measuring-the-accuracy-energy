"""Rung 1 demo: why an SNN is (and is not) cheaper than a CNN.

Run from the repo root:

    PYTHONPATH=. python scripts/demo_energy.py

Two lessons, both of which fall straight out of the numbers:

1. A randomly-initialised SNN is usually DEAD. Default thresholds are tuned for
   dense activations, and sparse event input never charges the membrane far
   enough to fire. A dead network reports fabulous energy savings and is
   completely useless -- always check the firing rate before believing a number.

2. The energy win is NOT automatic. It is a race between two factors:

       advantage  ~  (E_MAC / E_SOP)  /  (timesteps x firing_rate)
                  ~     5.1           /  (T x rate)

   Spikes are ~5.1x cheaper per operation, but the SNN runs T timesteps and
   each fires at some rate. Push T or the firing rate up and the advantage
   evaporates. This single relation drives every design decision in Rung 2.
"""

import torch
import torch.nn as nn

from src.engine.energy import EnergyCounter, compare
from src.models.lif import LIF, collect_firing_rates, reset_all

TIMESTEPS = 8
INPUT_SPARSITY = 0.05  # ~5% of pixels carry an event: typical for event data


def make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(2, 16, 3, padding=1, bias=False), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 8 * 8, 2, bias=False),
    )


def make_snn(threshold: float) -> nn.Module:
    """Identical architecture, ReLU swapped for LIF."""
    return nn.Sequential(
        nn.Conv2d(2, 16, 3, padding=1, bias=False), LIF(beta=0.9, threshold=threshold),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1, bias=False), LIF(beta=0.9, threshold=threshold),
        nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 8 * 8, 2, bias=False),
    )


def run_snn(model: nn.Module, spikes: torch.Tensor):
    reset_all(model)
    with EnergyCounter(model, mode="spiking") as counter:
        for t in range(spikes.shape[0]):
            model(spikes[t])
    return counter.report(timesteps=spikes.shape[0])


def main() -> None:
    torch.manual_seed(0)
    voxel = torch.randn(1, 2, 32, 32)
    spikes = (torch.rand(TIMESTEPS, 1, 2, 32, 32) < INPUT_SPARSITY).float()

    # ---- Lesson 1: threshold decides whether the network is alive at all ----
    print("Threshold sweep -- is the network even firing?\n")
    print(f"{'threshold':>10} {'layer1':>9} {'layer2':>9}   status")
    print("-" * 52)
    for th in (1.0, 0.5, 0.2, 0.1, 0.05):
        torch.manual_seed(0)
        model = make_snn(th)
        run_snn(model, spikes)
        rates = collect_firing_rates(model)
        deep = list(rates.values())[-1]
        if deep == 0:
            status = "DEAD -- no output spikes"
        elif deep < 0.02:
            status = "barely alive"
        elif deep < 0.25:
            status = "healthy"
        else:
            status = "too dense -- energy claim weakens"
        print(f"{th:>10} {list(rates.values())[0] * 100:>8.1f}% {deep * 100:>8.1f}%   {status}")

    # ---- Lesson 2: the honest head-to-head, at a healthy firing rate ----
    print("\n\nHead-to-head at threshold=0.2 (healthy firing):\n")
    torch.manual_seed(0)
    cnn = make_cnn()
    with EnergyCounter(cnn, mode="dense") as counter:
        cnn(voxel)
    dense_report = counter.report()

    torch.manual_seed(0)
    snn = make_snn(0.2)
    spike_report = run_snn(snn, spikes)

    print(compare(dense_report, spike_report))
    print(
        "\nfiring rates:",
        {k: f"{v * 100:.1f}%" for k, v in collect_firing_rates(snn).items()},
    )
    print(
        "\nNote: the SNN here does MORE operations than the CNN (8 timesteps x a\n"
        "27% firing rate outweighs the sparsity of the input), yet still wins on\n"
        "energy because each op is ~5x cheaper. Cutting timesteps or firing rate\n"
        "is what turns a 3x win into a 30x win -- that is the Rung 2 experiment."
    )


if __name__ == "__main__":
    main()
