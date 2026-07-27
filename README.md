# Event-Based Vision with Spiking Neural Networks

**How much accuracy do you trade for how much energy** when you replace a
conventional CNN with a spiking neural network on event-camera data?

Event cameras report per-pixel brightness changes as a sparse `(x, y, t, p)`
stream instead of frames — microsecond latency, ~120 dB dynamic range, and no
data at all where nothing moves. Spiking networks consume that stream natively
and only compute where spikes occur. This repo measures the trade-off with the
**same architecture on both sides** and a real operation count underneath.

Two benchmarks, both complete:

| | Dataset | CNN | SNN | Energy saving |
|---|---|---:|---:|---:|
| **Rung 2** | N-CARS — real automotive event data | 91.37% | **89.13%** | **7.5×** |
| **Rung 1** | N-MNIST — foundations | 97.85% | 95.55% | 2.5× |

> Energy is **estimated**, not measured on hardware: Horowitz (2014) 45 nm
> model, 4.6 pJ per MAC and 0.9 pJ per synaptic operation, ignoring memory
> traffic. Key points carry 3 seeds; the rest are single runs, and there was
> no hyperparameter search — a soundness check, not a state-of-the-art claim.

## Contents

- [Background — the three ideas this rests on](#background--the-three-ideas-this-rests-on)
- [Rung 1 — N-MNIST: building the measuring instrument](#rung-1--n-mnist-building-the-measuring-instrument)
- [Rung 2 — N-CARS: real automotive event data](#rung-2--n-cars-real-automotive-event-data)
- [Method](#method) · [Formulas and references](docs/FORMULAS.md)
- [Setup and reproduction](#setup-and-reproduction)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Layout](#layout) · [References](#references)

> **New to the project?** [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) explains
> every module — what it does, why it exists, how it works — plus the likely
> questions and their answers. Written for presenting the work.

---

# Background — the three ideas this rests on

### 1. An event camera has no frames

A normal camera samples every pixel on a fixed clock, whether or not anything
moved. An event camera's pixels fire **independently**, only when the log
intensity they see changes past a threshold:

```
x, y, t, p   ->   pixel (x, y) saw brightness go up (p=+1) or down (p=-1) at time t
```

Consequences that matter here: microsecond temporal resolution, ~120 dB dynamic
range against a conventional camera's ~60 dB, and **no data at all in static
regions**. That last property is what the whole project exploits.

### 2. A spiking neuron only works when it fires

A Leaky Integrate-and-Fire neuron accumulates input on a membrane, leaks it away
over time, and emits a binary spike when it crosses a threshold:

```
V[t] = beta * V[t-1] + I[t]        S[t] = 1 if V[t] >= threshold else 0
```

*Discrete LIF — Gerstner & Kistler (2002). Full form, including the reset term,
in [formula 5](docs/FORMULAS.md#5-lif-neuron-dynamics).*

`beta` is the only source of memory. A spike is exactly 0 or 1 — never a real
number — which is what makes the energy argument valid.

The catch: a step function has zero derivative everywhere, so backpropagation
dies. The fix is a **surrogate gradient** — use the true step forward, and a
smooth approximation of its derivative backward. Forward and backward
deliberately disagree.

*Technique: Neftci et al. (2019). The arctan form used here: Fang et al.
(ICCV 2021) — [formula 6](docs/FORMULAS.md#6-surrogate-gradient).*

### 3. Energy is counted, not assumed

A dense CNN performs a multiply-accumulate (MAC) for every output element,
regardless of input. A spiking network performs an **accumulate** (SynOp) only
where a spike occurred, because `w × 1 = w` and `w × 0` is skipped entirely.

On the standard 45 nm accounting: **4.6 pJ per MAC, 0.9 pJ per SynOp**. So

```
advantage  ≈  (E_MAC / E_SOP) / (timesteps × firing rate)  =  5.1 / (T × r)
```

The 5.1× from binary spikes is free. Everything after is a fight to keep
`T × r` small. **That single relation drives every experiment below.**

*Energy constants: Horowitz, ISSCC 2014. SynOps methodology: Rueckauer et al.
(2017), after Merolla et al. (Science 2014). Derivation in
[formulas 9–12](docs/FORMULAS.md#9-mac-count) — note the ratio is a first-order
model; all reported numbers come from the measured counter, not from it.*

---

# Rung 1 — N-MNIST: building the measuring instrument

A toy dataset, deliberately. The point was to build and validate the
apparatus before pointing it at data that matters.

### Step 1 — Goal

Prove that (a) an SNN can be trained at all with surrogate gradients, (b) the
energy counter is correct, and (c) the CNN/SNN comparison is fair. Accuracy on
handwritten digits is not the deliverable.

### Step 2 — Setup

| | |
|---|---|
| Dataset | N-MNIST — MNIST digits recorded by moving an event camera over a screen |
| Resolution | 34 × 34, 10 classes |
| Training subset | 8,000 samples, 15 epochs |
| Timesteps | T = 10 |

### Step 3 — Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/accuracy-vs-energy-dark.png">
  <img alt="Accuracy versus estimated energy on N-MNIST for the CNN, the SNN with BNTT, and the SNN with plain BatchNorm." src="docs/figures/accuracy-vs-energy-light.png" width="620">
</picture>

| Model | Accuracy | Operations / sample | Est. energy / sample | Params |
|---|---:|---:|---:|---:|
| CNN | 97.85% | 3.34 M MAC | 15.38 µJ | 24,634 |
| **SNN + BNTT** | **95.55%** | 6.95 M SynOp | **6.25 µJ** | 26,221 |
| SNN, plain BatchNorm | 66.10% | 8.00 M SynOp | 7.20 µJ | 26,221 |

**2.30 points of accuracy for 2.5× less energy.**

### Step 4 — What the model actually sees

An event camera produces no images. This is one sample as the binary spike
tensor the SNN consumes — each timestep almost entirely empty, which is exactly
the sparsity the energy argument rests on.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/event-representations-dark.png">
  <img alt="One event sample shown accumulated and then as five individual timesteps, each sparse." src="docs/figures/event-representations-light.png" width="880">
</picture>

Predictions from the trained spiking network on held-out data, with confidence
and ground truth:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/predictions-dark.png">
  <img alt="Twelve held-out event samples with the SNN's prediction, confidence, and ground truth." src="docs/figures/predictions-light.png" width="880">
</picture>

### Step 5 — The finding: plain BatchNorm costs an SNN 29 accuracy points

The SNN first scored **66.10%** while reaching 96.70% on the training set.
Textbook overfitting, apparently. It wasn't.

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
timestep** — early steps sparse while membranes charge, later steps dense — so
one set of running statistics is wrong at inference for every step. A CNN never
hits this: one forward pass, one distribution.

Fixed with **BNTT** (Batch Normalization Through Time, Kim & Panda 2021): one
BatchNorm per timestep. Reproduce the failure with `--plain-bn`.

Two earlier hypotheses — model capacity and MaxPool saturation — were both
wrong. Both are kept in [`docs/FINDINGS.md`](docs/FINDINGS.md) with what ruled
them out.

### Step 6 — Where the energy actually goes

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/energy-breakdown-dark.png">
  <img alt="The SNN performs more operations than the CNN yet uses less energy." src="docs/figures/energy-breakdown-light.png" width="760">
</picture>

The SNN performs **more** operations than the CNN and still wins, because each
one is an accumulate rather than a multiply-accumulate. This is the
`5.1 / (T × r)` relation in practice: at `T=10` and `r ≈ 21%` the denominator is
~2.1, which is exactly the 2.5× observed.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/firing-rate-dark.png">
  <img alt="Mean firing rate across training, settling near 21 percent." src="docs/figures/firing-rate-light.png" width="700">
</picture>

**A dead network reports spectacular energy savings.** An early demo showed
59.7× — from a network whose firing rate was 0.0% and which computed nothing at
all. Firing rate is therefore a first-class diagnostic, built into the neuron
rather than bolted on. Never report an energy number without it.

### Step 7 — Conclusion

The apparatus works: surrogate gradients train, the energy counter is
sparsity-aware and verified, and parameter parity is enforced by test. The
2.5× saving is real but modest — and the firing-rate plot shows why. **Density
is the lever, and it was left untouched.** That is what Rung 2 pulls.

→ Full write-up: [`docs/RUNG1_NMNIST.md`](docs/RUNG1_NMNIST.md)

---

# Rung 2 — N-CARS: real automotive event data

### Step 1 — Goal

Same question, real data: an ATIS event camera **mounted behind a car
windshield in urban driving**. And attack the density lever Rung 1 exposed.

### Step 2 — Dataset

24,029 recordings, car vs background, 100 ms per sample, 15,422 train / 8,607
test. Variable-size crops padded to a 120 × 100 canvas that covers 100% of
samples with no clipping.

### Step 3 — Baseline results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/ncars-pareto-dark.png">
  <img alt="Accuracy versus estimated energy on N-CARS across nine sparsity settings. The CNN reaches 91.37% at 228 uJ; the SNN reaches 89.13% at 30 uJ and still scores 86.21% at 2.9 uJ." src="docs/figures/ncars-pareto-light.png" width="680">
</picture>

| Model | Accuracy | Operations / sample | Est. energy / sample | Spike density | Params |
|---|---:|---:|---:|---:|---:|
| CNN | **91.37%** | 49.56 M MAC | 227.97 µJ | — | 98,226 |
| SNN (no penalty) | 88.05% ± 0.62 | 114.30 M SynOp | 102.86 µJ | 36.9% | 102,118 |
| **SNN (λ = 1.0)** | **89.13% ± 0.88** | 33.67 M SynOp | **30.30 µJ** | 21.2% | 102,118 |
| SNN (λ = 10) | 86.21% | 3.18 M SynOp | 2.87 µJ | 5.8% | 102,118 |

**7.5× less energy for 2.24 points of accuracy** at λ=1.0, rising to
**79.4×** at λ=10 for 5.16 points. Accuracies with ± are means over 3 seeds
(observed half-range); the rest are single runs.

### Step 4 — The sparsity sweep

Nine values of the firing-rate penalty `λ`, 15 runs total across three parallel
launches on Modal T4s, 30 epochs each. Three values carry seed repeats (n=3);
the spread is the observed half-range, not a standard deviation — with three
runs a std implies more distributional information than the data supports
([formula 15](docs/FORMULAS.md#15-seed-spread)). Overlapping intervals are
reported as "not separated" — a descriptive convention, not a significance test.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/ncars-sweep-dark.png">
  <img alt="Sweeping the sparsity penalty across nine values: spike density falls from 37 to 6 percent, energy from 103 to 3 microjoules, while accuracy stays within about 3 points." src="docs/figures/ncars-sweep-light.png" width="960">
</picture>

| λ | accuracy | seeds | spike density | est. energy | vs CNN |
|---|---:|---:|---:|---:|---:|
| 0 | 88.05% ± 0.62 | 3 | 36.9% | 102.86 µJ | 2.2× |
| 0.01 | 87.29% | 1 | 37.4% | 105.63 µJ | 2.2× |
| 0.05 | 87.19% ± 1.05 | 3 | 34.9% | 89.89 µJ | 2.5× |
| 0.1 | 88.24% | 1 | 33.5% | 82.22 µJ | 2.8× |
| 0.5 | 88.44% | 1 | 25.6% | 41.90 µJ | 5.4× |
| **1** | **89.13% ± 0.88** | **3** | **21.2%** | **30.30 µJ** | **7.5×** |
| 2 | 88.67% | 1 | 16.4% | 21.00 µJ | 10.9× |
| 5 | 87.27% | 1 | 9.1% | 6.72 µJ | 33.9× |
| 10 | 86.21% | 1 | 5.8% | 2.87 µJ | 79.4× |

CNN reference: **91.37%**, 49.56 M MAC, **227.97 µJ**.

**The penalty does not cost accuracy.** At λ=1.0 the network is
**89.13% ± 0.88** against the unpenalised **88.05% ± 0.62** — a mean 1.08 points
*higher* while using 3.4× less energy. The observed ranges overlap slightly
(88.25–90.01 vs 87.43–88.67), so the honest reading is that sparsity is **free
here, and possibly beneficial** — not that the improvement is established.
Separating it would need more seeds.

The mechanism is straightforward: nothing in a plain cross-entropy loss
discourages spiking, so the unpenalised network fires more than the task
requires. That activity carries no information, and removing it costs nothing.

**The λ=0.05 dip was noise.** The single-seed sweep showed 86.36% and it looked
like a real anomaly. With n=3 it is **87.19% ± 1.05**, overlapping λ=0's range.
It was seed variance. This is exactly why the repeats were worth the compute —
the original figure invited a reader to explain a feature that does not exist.

**How far sparsity goes.** Pushing λ past 1.0 keeps buying energy at a mild and
predictable accuracy cost:

- **λ=2**: 88.67% at 21.00 µJ — **10.9× less energy than the CNN**
- **λ=5**: 87.27% at 6.72 µJ — **33.9×**
- **λ=10**: 86.21% at 2.87 µJ — **79.4×**

From λ=0 to λ=10 is a **36× energy cut for 1.84 accuracy points**. There is no
cliff in this range, only a gentle slope. At the far end the network still
scores 86.21% — essentially CarSNN's published 86.94% — at roughly 1/80th the
CNN's energy.

λ=2, 5 and 10 are **single-seed**; only λ=0, 0.05 and 1.0 carry repeats.

### Step 5 — Predictions on real driving data

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/ncars-predictions-dark.png">
  <img alt="Twelve held-out N-CARS crops with the spiking network's prediction, confidence, and ground truth. Cars are clearly visible in the accumulated event data." src="docs/figures/ncars-predictions-light.png" width="880">
</picture>

The λ=1.0 spiking network on held-out crops — the sparse model that runs at
30 µJ. Car bodies, windshields and headlights are visible in the accumulated
events despite there being no image sensor involved.

The two errors differ in a way worth noting. One is a hedge — a textured
background called a car at 58% confidence, right at the decision boundary. The
other is **confidently wrong**: a car called background at 97%, on a crop with
5,593 spikes, well above the median of 3,504 for these samples. So it is not a
lack of data. Confident errors on data-rich inputs are the failure mode that
matters for a safety-critical application, and calibration is not something
this project has measured.

> **These are classification results, not detection.** The task is "is there a
> car in this crop", not "where are the cars". Bounding-box detection is Rung 3
> and does not exist anywhere in this repository yet.

### Step 6 — Against published results on the same benchmark

| Method | Accuracy | |
|---|---:|---|
| **Our CNN** | **91.37%** | |
| [HATS](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf) (Sironi et al., CVPR 2018) | 90.2% | the dataset's own paper |
| **Our SNN** | **89.13%** | mean of 3 seeds |
| [CarSNN](https://arxiv.org/pdf/2107.00401) (Viale et al., IJCNN 2021) | 86.94% | SNN deployed to Loihi |
| Gabor-SNN | 78.9% | |
| HOTS | 62.4% | |

The SNN sits above the published spiking reference. Caveats stated plainly: no
hyperparameter search, and the protocol is not matched to CarSNN's (they use an
attention window; we use the full crop). This establishes the implementation is
sound — it is not a SOTA claim.

### Step 7 — Conclusion

Three things came out of Rung 2 that were not true going in:

1. **The energy advantage is much larger than Rung 1 suggested** — 7.5× at
   matched accuracy, 79× if 5 points can be spent. Rung 1's 2.5× was not a
   ceiling; it was an unpenalised network wasting spikes.
2. **Sparsity is free on this task.** The Pareto framing assumed a trade. There
   isn't one in the useful range, because the surplus activity carried no
   information.
3. **A single-seed anomaly evaporated.** The λ=0.05 dip was seed variance, and
   would have been reported as a real feature without the repeats.

→ Full write-up: [`docs/RUNG2_NCARS.md`](docs/RUNG2_NCARS.md)

---

# Method

Both models are built from the same `ConvBlock` stack. The **only** differences:

| | CNN | SNN |
|---|---|---|
| Activation | ReLU | LIF neuron (surrogate gradient) |
| Input | Voxel grid | Binary spike tensor over T steps |
| Normalisation | BatchNorm | BNTT (per-timestep) |
| Forward pass | Once | Once per timestep, membrane state carried |

Parameter parity is **enforced by a test** — any capacity difference would make
the energy comparison meaningless, and it is the first thing a reviewer checks.

**The LIF neuron is written from scratch** ([`src/models/lif.py`](src/models/lif.py))
rather than imported, so the dynamics stay debuggable, and it is verified
against `snntorch` to 1e-6 on spikes, membrane potential, and surrogate
gradients.

**Binary spikes are load-bearing.** `events_to_snn_input` assigns `1.0` rather
than accumulating counts. Real-valued inputs would make synapses perform
multiply-accumulates and invalidate the entire energy argument. Guarded by
`test_snn_input_is_strictly_binary`.

**Energy is measured, not estimated statically** ([`src/engine/energy.py`](src/engine/energy.py)):
forward hooks count real input sparsity per layer, because SynOps depend on the
data — which is exactly what the sparsity sweep exploits.

**MaxPool is deliberate.** It is the only common pooling op that preserves
binary outputs; AvgPool emits `{0, .25, .5, .75, 1}`, which would silently make
the following convolution perform real multiplications and invalidate the
SynOps count.

## Every formula, with its source

Full derivations, symbol tables and code line references are in
[`docs/FORMULAS.md`](docs/FORMULAS.md).

| Quantity | Formula | Source |
|---|---|---|
| [Voxel grid](docs/FORMULAS.md#2-voxel-grid) | `V[b,y,x] = Σᵢ pᵢ · max(0, 1 − \|b − tᵢ*\|)` | Zhu et al., CVPR 2019 |
| [Time surface](docs/FORMULAS.md#3-time-surface) | `S = exp(−(t_end − t_last) / τ)` | Lagorce et al., TPAMI 2017 |
| [Spike tensor](docs/FORMULAS.md#4-binary-spike-tensor) | `S[b,c,y,x] = 1` if any event in bin | *standard* |
| [LIF dynamics](docs/FORMULAS.md#5-lif-neuron-dynamics) | `V[t] = βV[t−1] + I[t] − S[t−1]θ` | Gerstner & Kistler, 2002 |
| [Surrogate gradient](docs/FORMULAS.md#6-surrogate-gradient) | `∂S/∂x = α / (1 + (παx)²)` | Neftci 2019; Fang, ICCV 2021 |
| [Learnable decay](docs/FORMULAS.md#7-learnable-decay) | `β = σ(clamp(ℓ, −8, 8))` | Fang et al., ICCV 2021 |
| [BNTT](docs/FORMULAS.md#8-bntt-batch-normalization-through-time) | per-timestep `(μₜ, σₜ, γₜ, βₜ)` | Kim & Panda, 2021 |
| [MAC count](docs/FORMULAS.md#9-mac-count) | `MACs = \|O\| · (C_in/g) · k_H · k_W` | *standard* |
| [SynOps](docs/FORMULAS.md#10-synaptic-operations-synops) | `SynOps = Σ_l Σ_t MACs_l · r_{l,t}` | Merolla 2014; Rueckauer 2017 |
| [Energy](docs/FORMULAS.md#11-energy-model) | `E = MACs × 4.6 pJ` · `SynOps × 0.9 pJ` | Horowitz, ISSCC 2014 |
| [Energy ratio](docs/FORMULAS.md#12-energy-ratio) | `E_CNN / E_SNN = 5.1 / (T · r̄)` | *derived here* |
| [Firing rate](docs/FORMULAS.md#13-firing-rate) | `r = spikes / (N_neurons · T)` | *standard* |
| [Sparsity penalty](docs/FORMULAS.md#14-sparsity-penalty) | `L = L_CE + λ · (1/L)Σ_l r̄_l` | *standard* regularisation |
| [Seed spread](docs/FORMULAS.md#15-seed-spread) | `(max − min) / 2`, **not** std | *standard* |

---

# Setup and reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q        # 129 tests
```

Full instructions for a CUDA training box, including the Windows pitfalls, are
in [`docs/SETUP.md`](docs/SETUP.md). Datasets are not committed — the rung pages
explain how to obtain them.

Run the N-CARS benchmark:

```bash
PYTHONPATH=. python scripts/run_ncars.py --epochs 30 --batch-size 64
```

Run the Rung 1 benchmark:

```bash
PYTHONPATH=. python scripts/run_nmnist.py --epochs 15 --num-steps 10 --lr 5e-3
```

On Modal GPUs, with the sweep fanned out in parallel:

```bash
modal run --detach modal_app.py::sweep
```

Seed repeats and the λ extension:

```bash
modal run --detach modal_app.py::sweep_seeds
```

Re-pool every launch into the committed result CSVs:

```bash
PYTHONPATH=. python scripts/aggregate_results.py
```

Regenerate every figure from those CSVs, no training required:

```bash
PYTHONPATH=. python scripts/make_figures.py
```

See why a randomly-initialised SNN is usually dead:

```bash
PYTHONPATH=. python scripts/demo_energy.py
```

---

# Limitations

Stated here rather than left for a reader to discover.

| | |
|---|---|
| **Energy is estimated** | Horowitz (2014) 45 nm model. Memory traffic is ignored and often dominates real accelerators. No silicon was involved. |
| **λ=2, 5, 10 are single-seed** | The far end of the front, including the 79× figure, has no error bars. |
| **The CNN baseline is single-seed** | And it is the denominator of every energy ratio quoted anywhere here. |
| **Timesteps were never varied** | The relation is `5.1/(T×r)`; only `r` was ever attacked, with `T=10` throughout. Halving `T` is an untested second lever worth roughly another 2×. |
| **Calibration is unmeasured** | The qualitative figure contains a 97%-confident wrong prediction on a data-rich crop. |
| **N-MNIST used a subset** | 8,000 of 60,000 training samples. |
| **No detection** | Everything here is classification. Bounding boxes are Rung 3. |

---

# Roadmap

| | Status | |
|---|---|---|
| **[Rung 1 — N-MNIST](docs/RUNG1_NMNIST.md)** | ✅ complete | Foundations, energy accounting, the BatchNorm finding |
| **[Rung 2 — N-CARS](docs/RUNG2_NCARS.md)** | ✅ complete | Real automotive data, the sparsity sweep, seed repeats |
| **Rung 3 — detection** | planned | Bounding boxes on GEN1 / DSEC: mAP, latency, day vs night |

**Bounding-box detection does not exist yet** — that is Rung 3. Everything above
is classification, which answers the same accuracy/energy question at a scale
that resolves in weeks rather than months.

Next candidates, in order of value: vary `T` (the untouched half of the energy
relation), seed the far end of the front, then Rung 3.

# Layout

```
src/data/        event -> tensor representations, N-MNIST and N-CARS loaders,
                 Prophesee .dat reader
src/models/      LIF neurons, surrogate gradients, paired CNN/SNN, BNTT
src/engine/      MAC/SynOps energy accounting, shared train/eval loop
scripts/         benchmarks, figure generation, result aggregation, energy demo
modal_app.py     parallel GPU execution on Modal
results/         measured metrics (CSV) -- figures regenerate from these
docs/            rung write-ups, project plan, findings log, setup guide
tests/           129 tests
```

# References

- Gallego et al., *Event-based Vision: A Survey*, TPAMI 2020
- Sironi et al., *HATS*, CVPR 2018 — the N-CARS dataset
- Viale et al., *CarSNN*, IJCNN 2021 — SNN on N-CARS, deployed to Loihi
- Neftci et al., *Surrogate Gradient Learning in Spiking Neural Networks*, IEEE SPM 2019
- Kim & Panda, *Revisiting Batch Normalization for Training Low-latency Deep SNNs*, 2021 — BNTT
- Zhu et al., *Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion*, CVPR 2019 — voxel grid
- Gehrig & Scaramuzza, *Recurrent Vision Transformers for Object Detection with Event Cameras*, CVPR 2023
- Zhang et al., *Automotive Object Detection via Learning Sparse Events by Spiking Neurons*, IEEE TCDS 2024
- Horowitz, *Computing's Energy Problem*, ISSCC 2014 — energy accounting
