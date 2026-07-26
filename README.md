# Event-Based Vision with Spiking Neural Networks

**How much accuracy do you trade for how much energy** when you replace a
conventional CNN with a spiking neural network on event-camera data?

Event cameras report per-pixel brightness changes as a sparse `(x, y, t, p)`
stream instead of frames — microsecond latency, ~120 dB dynamic range, and no
data at all where nothing moves. Spiking networks consume that stream natively
and only compute where spikes occur. This repo measures the trade-off honestly,
with the same architecture on both sides and a real operation count underneath.

---

## Headline result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/accuracy-vs-energy-dark.png">
  <img alt="Accuracy versus estimated inference energy. The SNN with BNTT reaches 95.55% at 6.25 uJ; the CNN reaches 97.85% at 15.38 uJ." src="docs/figures/accuracy-vs-energy-light.png" width="620">
</picture>

| Model | Accuracy | Operations / sample | Est. energy / sample | Params |
|---|---:|---:|---:|---:|
| CNN | **97.85%** | 3.34 M MAC | 15.38 µJ | 24,634 |
| **SNN + BNTT** | **95.55%** | 6.95 M SynOp | **6.25 µJ** | 26,221 |
| SNN, plain BatchNorm | 66.10% | 8.00 M SynOp | 7.20 µJ | 26,221 |

**2.30 points of accuracy for 2.5× less estimated energy.**

> Energy is **estimated**, not measured on hardware. It uses the Horowitz (2014)
> 45 nm model — 4.6 pJ per MAC, 0.9 pJ per synaptic operation — and ignores
> memory traffic, which often dominates real accelerators.

Benchmark: N-MNIST, 8,000-sample training subset, 15 epochs, 10 timesteps.
Both models share the same architecture, depth, and width — only the neuron
type and the input representation differ.

---

## What the model actually sees

An event camera produces no images. This is one N-MNIST sample as the binary
spike tensor the SNN consumes — each timestep almost entirely empty, which is
exactly the sparsity the energy argument rests on.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/event-representations-dark.png">
  <img alt="One event sample shown accumulated and then as five individual timesteps, each sparse." src="docs/figures/event-representations-light.png" width="880">
</picture>

## What it predicts

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/predictions-dark.png">
  <img alt="Twelve held-out event samples with the SNN's prediction, confidence, and ground truth." src="docs/figures/predictions-light.png" width="880">
</picture>

Predictions from the trained spiking network on held-out data, with confidence
and ground truth. Events are accumulated over all 10 timesteps for display; the
model consumes them step by step.

---

## The finding that mattered

The SNN first scored **66.10%** while reaching 96.70% on the training set. That
looks like textbook overfitting. It wasn't.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/training-curves-dark.png">
  <img alt="Validation accuracy over 15 epochs for the CNN, the SNN with BNTT, and the SNN with plain BatchNorm." src="docs/figures/training-curves-light.png" width="700">
</picture>

Evaluating the **same checkpoint** on the **same data**, changing only the
BatchNorm mode:

| BatchNorm statistics | Accuracy |
|---|---:|
| Batch statistics (train mode) | **93.36%** |
| Running statistics (eval mode) | 65.62% |

The weights were fine. An SNN's activation distribution **changes at every
timestep** — early steps are sparse while membranes charge, later steps dense —
so one set of running statistics is wrong at inference for every step. A CNN
never hits this: one forward pass, one distribution.

Fixed with **BNTT** (Batch Normalization Through Time, Kim & Panda 2021) — one
BatchNorm per timestep. Reproduce the failure with `--plain-bn`.

Two earlier hypotheses — model capacity and MaxPool saturation — were both
wrong. They're kept in [`docs/FINDINGS.md`](docs/FINDINGS.md) along with what
ruled them out.

---

## Where the energy goes

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/energy-breakdown-dark.png">
  <img alt="The SNN performs 6.95 M operations versus the CNN's 3.34 M, yet uses 6.25 uJ versus 15.38 uJ." src="docs/figures/energy-breakdown-light.png" width="760">
</picture>

The SNN performs **more** operations than the CNN and still wins, because each
one is an accumulate rather than a multiply-accumulate. The governing relation:

```
advantage  ≈  (E_MAC / E_SOP) / (timesteps × firing rate)  =  5.1 / (T × r)
```

The 5.1× from binary spikes is free. Everything after is a fight to keep
`T × r` small.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/firing-rate-dark.png">
  <img alt="Mean firing rate across training, settling near 21 percent for both SNN runs." src="docs/figures/firing-rate-light.png" width="700">
</picture>

At `T=10` and `r≈21%` the denominator is ~2.1, which is where the 2.5× comes
from. **That's the remaining headroom**: driving density toward 10% via the
firing-rate penalty and a higher threshold should push this well past 5×.
Tracing that curve is the next experiment.

---

## Method

Both models are built from the same `ConvBlock` stack. The **only** differences:

| | CNN | SNN |
|---|---|---|
| Activation | ReLU | LIF neuron (surrogate gradient) |
| Input | Voxel grid, 5 bins | Binary spike tensor, 10 steps |
| Normalisation | BatchNorm | BNTT (per-timestep) |
| Forward pass | Once | Once per timestep, membrane state carried |

Parameter parity is enforced by a test — any capacity difference would make the
energy comparison meaningless.

**The LIF neuron is written from scratch** ([`src/models/lif.py`](src/models/lif.py))
rather than imported, so the dynamics stay debuggable, and it is verified
against `snntorch` to 1e-6 on spikes, membrane potential, and surrogate
gradients.

**Energy is measured, not estimated statically** ([`src/engine/energy.py`](src/engine/energy.py)).
Forward hooks count real input sparsity per layer, because SynOps depend on the
data — which is the whole point of the eventual day/night analysis.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q        # 100 tests
```

Reproduce the benchmark:

```bash
PYTHONPATH=. python scripts/run_nmnist.py --epochs 15 --num-steps 10 --lr 5e-3
```

Regenerate the figures from committed results (no training needed):

```bash
PYTHONPATH=. python scripts/make_figures.py
```

See why a randomly-initialised SNN is usually dead:

```bash
PYTHONPATH=. python scripts/demo_energy.py
```

---

## Roadmap

| | Status |
|---|---|
| **Rung 1** — foundations, energy accounting, N-MNIST | ✅ complete |
| **Rung 2** — N-CARS: 24,029 real automotive event recordings, firing-rate sweep | next |
| **Rung 3** — object detection with bounding boxes on driving data (GEN1 / DSEC), mAP + latency, day vs night | planned |

**Detection results with bounding boxes do not exist yet** — that is Rung 3.
What is shown above is classification on event data, which is the same
accuracy/energy question at a scale that gets answered in weeks rather than
months. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full plan.

## Layout

```
src/data/        event -> tensor representations, dataset wrappers
src/models/      LIF neurons, surrogate gradients, paired CNN/SNN, BNTT
src/engine/      MAC/SynOps energy accounting, shared train/eval loop
scripts/         benchmark, figures, energy demo
results/         measured metrics (CSV) -- figures regenerate from these
docs/            project plan, findings, figures
tests/           100 tests
```

## References

- Gallego et al., *Event-based Vision: A Survey*, TPAMI 2020
- Neftci et al., *Surrogate Gradient Learning in Spiking Neural Networks*, IEEE SPM 2019
- Kim & Panda, *Revisiting Batch Normalization for Training Low-latency Deep SNNs*, 2021 (BNTT)
- Sironi et al., *HATS*, CVPR 2018 (N-CARS)
- Gehrig & Scaramuzza, *Recurrent Vision Transformers for Object Detection with Event Cameras*, CVPR 2023
- Zhang et al., *Automotive Object Detection via Learning Sparse Events by Spiking Neurons*, IEEE TCDS 2024
- Horowitz, *Computing's Energy Problem*, ISSCC 2014 (energy accounting)
