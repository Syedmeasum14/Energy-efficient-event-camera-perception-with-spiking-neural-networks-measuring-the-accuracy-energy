# Event-Based Vision with Spiking Neural Networks

**How much accuracy do you trade for how much energy** when you replace a
conventional CNN with a spiking neural network on event-camera data?

Event cameras report per-pixel brightness changes as a sparse `(x, y, t, p)`
stream instead of frames — microsecond latency, ~120 dB dynamic range, and no
data at all where nothing moves. Spiking networks consume that stream natively
and only compute where spikes occur. This repo measures the trade-off with the
same architecture on both sides and a real operation count underneath.

---

## Result — N-CARS, real automotive event data

24,029 recordings from an ATIS event camera behind a car windshield in urban
driving. Car vs background, 100 ms per sample.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/ncars-pareto-dark.png">
  <img alt="Accuracy versus estimated energy on N-CARS. The CNN reaches 91.37% at 228 uJ; the sparsity-penalised SNN reaches 88.52% at 31 uJ." src="docs/figures/ncars-pareto-light.png" width="680">
</picture>

| Model | Accuracy | Operations / sample | Est. energy / sample | Spike density |
|---|---:|---:|---:|---:|
| CNN | **91.37%** | 49.56 M MAC | 227.97 uJ | — |
| SNN (no penalty) | 88.27% | 120.88 M SynOp | 108.79 uJ | 37.9% |
| **SNN (λ = 1.0)** | **88.52%** | 34.87 M SynOp | **31.38 uJ** | 22.0% |

**7.3× less energy for 2.9 points of accuracy** — competitive with published
spiking work on the same benchmark ([CarSNN](https://arxiv.org/pdf/2107.00401) 86.94%,
[HATS](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf) 90.2%).

> Energy is **estimated**, not measured on hardware: Horowitz (2014) 45 nm
> model, 4.6 pJ per MAC and 0.9 pJ per synaptic operation, ignoring memory
> traffic. Single seed, no hyperparameter search — a soundness check, not a
> state-of-the-art claim.

---

## The rungs

Each is a complete piece of work with its own write-up.

| | | |
|---|---|---|
| **[Rung 1 — N-MNIST](docs/RUNG1_NMNIST.md)** | ✅ complete | Foundations: LIF neurons from scratch, energy accounting, and the BatchNorm finding that cost 29 accuracy points |
| **[Rung 2 — N-CARS](docs/RUNG2_NCARS.md)** | ✅ complete | Real automotive data, the sparsity sweep, and the result above |
| **Rung 3 — detection** | planned | Bounding boxes on GEN1 / DSEC: mAP, latency, day vs night |

**Bounding-box detection does not exist yet** — that is Rung 3. Everything
above is classification, which answers the same accuracy/energy question at a
scale that resolves in weeks rather than months.

## Two findings worth the click

**[Plain BatchNorm costs an SNN 29 accuracy points.](docs/RUNG1_NMNIST.md#the-finding-plain-batchnorm-costs-29-accuracy-points)**
The same checkpoint scored 93.36% with batch statistics and 65.62% with running
statistics. An SNN's activation distribution changes at every timestep, so one
set of running statistics is wrong at inference for all of them. It presents as
catastrophic overfitting while the weights are fine.

**[The sparsity penalty turned out to be free.](docs/RUNG2_NCARS.md#the-penalty-is-free-which-was-not-the-expectation)**
It was built on the assumption that sparsity must be bought with accuracy. On
N-CARS, λ=1.0 is marginally *more* accurate than λ=0 while using 3.2× less
energy — the unpenalised network was simply spiking more than the task needed.

---

## Method

Both models are built from the same `ConvBlock` stack. The **only** differences:

| | CNN | SNN |
|---|---|---|
| Activation | ReLU | LIF neuron (surrogate gradient) |
| Input | Voxel grid | Binary spike tensor over T steps |
| Normalisation | BatchNorm | BNTT (per-timestep) |
| Forward pass | Once | Once per timestep, membrane state carried |

Parameter parity is enforced by a test — any capacity difference would make the
energy comparison meaningless.

The **LIF neuron is written from scratch** ([`src/models/lif.py`](src/models/lif.py))
rather than imported, so the dynamics stay debuggable, and it is verified
against `snntorch` to 1e-6 on spikes, membrane potential, and surrogate
gradients.

**Energy is measured, not estimated statically** ([`src/engine/energy.py`](src/engine/energy.py)):
forward hooks count real input sparsity per layer, because SynOps depend on the
data — which is exactly what the sparsity sweep exploits.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q        # 129 tests
```

Full instructions for a CUDA training box, including the Windows pitfalls, are
in [`docs/SETUP.md`](docs/SETUP.md). Datasets are not committed; see the rung
pages for how to obtain them.

Run the N-CARS benchmark:

```bash
PYTHONPATH=. python scripts/run_ncars.py --epochs 30 --batch-size 64
```

Or on Modal GPUs, with the six-point sweep fanned out in parallel:

```bash
modal run --detach modal_app.py::sweep
```

## Layout

```
src/data/        event -> tensor representations, N-MNIST and N-CARS loaders,
                 Prophesee .dat reader
src/models/      LIF neurons, surrogate gradients, paired CNN/SNN, BNTT
src/engine/      MAC/SynOps energy accounting, shared train/eval loop
scripts/         benchmarks, figure generation, energy demo
modal_app.py     parallel GPU execution on Modal
results/         measured metrics (CSV) -- figures regenerate from these
docs/            rung write-ups, project plan, findings log, setup guide
tests/           129 tests
```

## References

- Gallego et al., *Event-based Vision: A Survey*, TPAMI 2020
- Sironi et al., *HATS*, CVPR 2018 — the N-CARS dataset
- Viale et al., *CarSNN*, IJCNN 2021 — SNN on N-CARS, deployed to Loihi
- Neftci et al., *Surrogate Gradient Learning in Spiking Neural Networks*, IEEE SPM 2019
- Kim & Panda, *Revisiting Batch Normalization for Training Low-latency Deep SNNs*, 2021 — BNTT
- Gehrig & Scaramuzza, *Recurrent Vision Transformers for Object Detection with Event Cameras*, CVPR 2023
- Zhang et al., *Automotive Object Detection via Learning Sparse Events by Spiking Neurons*, IEEE TCDS 2024
- Horowitz, *Computing's Energy Problem*, ISSCC 2014 — energy accounting
