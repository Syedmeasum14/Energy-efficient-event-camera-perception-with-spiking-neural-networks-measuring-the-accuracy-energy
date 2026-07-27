# Walkthrough — what, why, how

Presentation notes for the whole repository. Every module: **what** it does,
**why** it exists, **how** it works. Read top to bottom and you can present the
project without opening the code.

---

## 0. The pitch, in three sentences

> Event cameras report brightness *changes* as a sparse stream instead of
> frames. Spiking neural networks consume that stream natively and only compute
> where spikes occur, so they should use less energy. **This project measures
> how much accuracy that saves cost — on real automotive data, with the same
> architecture on both sides.**

**Headline:** on N-CARS, **7.5× less energy for 2.24 accuracy points** — and up
to **79×** if you can spend 5 points.

---

## 1. The three ideas everything rests on

### Event cameras

**What:** a camera with no frames. Each pixel fires independently when its log
intensity changes past a threshold, emitting `(x, y, t, p)`.

**Why it matters:** microsecond latency, ~120 dB dynamic range (vs ~60 dB), and
**no data at all where nothing moves**. That sparsity is the whole opportunity.

**Say this:** "A normal camera samples every pixel on a clock whether or not
anything happened. An event camera only reports change."

### Spiking neurons (LIF)

**What:** a neuron that accumulates charge, leaks it, and emits a **binary**
spike when it crosses a threshold.

```
V[t] = beta * V[t-1] + I[t]        S[t] = 1 if V[t] >= threshold else 0
```

**Why binary matters:** `w × 1 = w`, so a synapse does an **add**, not a
multiply. That is the entire energy argument. If spikes were real numbers the
claim would collapse.

**The problem:** a step function has zero derivative everywhere, so
backpropagation dies.

**The fix — surrogate gradients:** use the true step *forward*, and a smooth
approximation of its derivative *backward*. Forward and backward deliberately
disagree. This is standard practice (Neftci et al., 2019), not a hack.

### The energy relation

```
advantage  ≈  (E_MAC / E_SOP) / (timesteps × firing rate)  =  5.1 / (T × r)
```

- **4.6 pJ** per multiply-accumulate, **0.9 pJ** per synaptic operation
  (Horowitz 2014, 45 nm) → 5.1× free from binary spikes
- Divided by `T` because the SNN runs T timesteps per sample
- Divided by `r` because only firing neurons cost anything

**Say this:** "You get 5× for free. Everything after that is a fight to keep
timesteps times firing rate small." **This one equation predicts every result
in the project.**

---

## 2. The data pipeline

### `src/data/representations.py`

**What:** four ways to turn a raw event stream into a tensor.

| Function | Keeps | Destroys |
|---|---|---|
| `events_to_histogram` | event density, shape | all timing |
| `events_to_voxel_grid` | coarse timing (interpolated) | fine timing |
| `events_to_time_surface` | recency per pixel | density — only the *last* event survives |
| `events_to_snn_input` | timing as a real axis | event counts (binarised) |

**Why four:** a CNN needs a dense grid, so time must be collapsed — and *how*
you collapse it is the biggest design decision in event vision. Each option
loses something different. Histogram and time surface are opposites: one keeps
*how many*, the other keeps *how recent*.

**How the important one works:** `events_to_snn_input` assigns `1.0` — it does
**not** accumulate counts.

```python
spikes[step_idx, pol_idx, y, x] = 1.0   # assignment, not +=
```

**Why that line matters:** real-valued counts would make every downstream
synapse perform a multiply-accumulate, and the "0.9 pJ vs 4.6 pJ" claim would
silently become false. A test (`test_snn_input_is_strictly_binary`) guards it.

### `src/data/prophesee.py`

**What:** reader for Prophesee's `.dat` binary format.

**Why:** `tonic` (the standard event-data library) doesn't ship N-CARS. The same
format is used by GEN1 and 1 Mpx, so this also unblocks Rung 3.

**How:** ASCII header, then 8 bytes per event — 4 for the timestamp, 4 packing
14 bits x, 14 bits y, 1 bit polarity. Unpacked with vectorised bit operations
(`unpack_events`); a Python loop over millions of events is unusably slow.
`write_dat` exists so tests can round-trip without the dataset.

### `src/data/datasets.py`

**What:** the bridge from raw events to model-ready tensors.

- `EventTensorDataset` — wraps any event source, applies a representation
- `NCarsDataset` / `load_ncars` — walks the Prophesee folder layout
- `load_nmnist` — Rung 1 data via `tonic`
- `SubsetWrapper` — a **shuffled** subset for fast tests

**Why `SubsetWrapper` shuffles:** N-CARS and N-MNIST are stored **sorted by
class**. Taking the first N samples gives a single-class dataset on which any
model scores 100% for free. This actually happened — a 1,730-parameter network
"achieved" 100%. A regression test now guards it.

**Say this:** "Most event datasets are stored sorted by class. If you subset
them naively your model looks perfect and has learned nothing."

---

## 3. The model

### `src/models/lif.py` — the heart

| Piece | What | Why |
|---|---|---|
| `SpikeFunction` | Heaviside forward, smooth derivative backward | The surrogate gradient. Without it nothing learns. |
| `LIF` | The neuron layer, stateful across timesteps | Membrane potential is the only memory |
| `reset_all` | Clears state before each sample | Otherwise state leaks between unrelated samples |
| `collect_firing_rates` | Per-layer diagnostic (detached) | For reporting |
| `firing_rate_loss` | Same quantity, **differentiable** | For *optimising* sparsity |

**Why write LIF by hand instead of importing `snntorch`?** When a deep SNN stops
learning it is almost always one of three things — the membrane decayed too
fast, the neurons went silent, or the surrogate gradient vanished. None are
debuggable through a library call. It is then **verified against `snntorch` to
1e-6** so the hand-written version is provably correct.

**How `beta` works:** the decay factor, and the *only* source of memory.
`beta=0.1` → the neuron forgets almost everything each step and the SNN
degenerates into a CNN run T times (all cost, no benefit). `beta=0.999` → it
integrates forever and can't tell recent input from old.

**A real bug worth mentioning:** `beta` is learnable, parameterised as
`sigmoid(logit)` to stay in (0,1). But `sigmoid(50)` saturates to *exactly* 1.0
in float32 — a neuron that never leaks, so the membrane explodes. Fixed by
clamping the logit to ±8. Caught by a test, not by training.

### `src/models/classifier.py`

| Piece | What | Why |
|---|---|---|
| `ConvBlock` | conv → norm → activation → pool | Shared skeleton, so both models are identical |
| `CNNClassifier` | ReLU version | The baseline |
| `SNNClassifier` | LIF version, loops over timesteps | The contribution |
| `BNTT` | One BatchNorm **per timestep** | See below — worth 29 accuracy points |
| `count_parameters` | Parameter count | Parity is enforced by test |

**Why identical architecture:** if the two models differed in depth or width,
any energy difference could be blamed on the architecture and the comparison
would be worthless. **This is the first thing a reviewer checks**, so it's a
test, not a convention.

**Why MaxPool and not AvgPool:** MaxPool is the only common pooling op that
keeps outputs binary. AvgPool emits `{0, .25, .5, .75, 1}` — the next conv would
then do real multiplications and the SynOps count would silently become a lie.

**Why the output layer doesn't spike:** spiking would quantise the logits to
integers and destroy gradient resolution at the loss. It's acceptable because
the head is one small `Linear` — a rounding error against millions of conv
operations.

---

## 4. The measurement — the actual contribution

### `src/engine/energy.py`

**What:** counts operations and converts to picojoules.

- `EnergyCounter` — attaches forward hooks to every conv/linear layer
- `_dense_macs` — MACs per layer, hand-checked against the standard formula
- `EnergyReport` / `compare` — aggregate and print

**Why hooks instead of a static formula:** SynOps depend on the **actual input
sparsity**, which is a property of the *data*, not the architecture. A static
count cannot know that a sparser scene produces fewer spikes — and that
difference is exactly what the project measures.

**How:** for each layer, `macs × fraction_of_input_that_is_nonzero`, summed over
timesteps. For a dense model, full price regardless of input.

**Say this:** "This file is the difference between a tutorial and a research
artifact. Without it you have 'I trained an SNN.' With it you have a
measurement."

### `src/engine/train.py`

- `fit` — train/validate loop, keeps the best checkpoint
- `run_epoch` — one pass; trains if given an optimizer, else evaluates
- `measure_energy` — energy on **real batches**, not synthetic input
- `pick_device` — CUDA on the training box, MPS on the laptop, else CPU

**Why gradient clipping is not optional:** SNNs backpropagate *through time*, so
gradients compound across T timesteps and blow up readily.

**The sparsity penalty:**

```
loss = cross_entropy + lambda * firing_rate_loss(model)
```

**Why:** nothing in a plain loss discourages firing. Dense firing often gives
slightly better accuracy, so gradient descent will happily trade away the entire
reason you built an SNN — because nothing told it not to.

---

## 5. The experiments

### Rung 1 — N-MNIST

**What:** CNN vs SNN on handwritten digits recorded with an event camera.
**Why:** validate the apparatus on something cheap before spending money on real
data. Accuracy on digits is not the deliverable.
**Result:** 97.85% vs 95.55%, **2.5× less energy**.

### Rung 2 — N-CARS

**What:** same comparison on 24,029 recordings from a camera **behind a car
windshield**, car vs background.
**Why:** this is the number that actually matters — real automotive data, and a
published benchmark to compare against.
**Result:** 91.37% vs **89.13% ± 0.88**, **7.5× less energy**.

### The sparsity sweep

**What:** train nine times with different `λ`, from 0 to 10.
**Why:** Rung 1 showed density was the untouched lever in `5.1/(T×r)`. The sweep
pulls it.
**How:** nine independent runs fanned across nine GPUs on Modal in parallel —
80 minutes wall-clock instead of 12 hours, for the same cost.

| λ | accuracy | density | energy | vs CNN |
|---|---:|---:|---:|---:|
| 0 | 88.05% ± 0.62 | 36.9% | 102.86 µJ | 2.2× |
| **1** | **89.13% ± 0.88** | 21.2% | 30.30 µJ | **7.5×** |
| 10 | 86.21% | 5.8% | 2.87 µJ | **79.4×** |

---

## 6. Findings — the things worth presenting

### Finding 1: plain BatchNorm costs an SNN 29 accuracy points

**What happened:** the SNN scored 66% while hitting 96.7% on training data.
Looked like catastrophic overfitting.

**How it was diagnosed:** evaluate the **same checkpoint** on the **same data**,
changing only the BatchNorm mode:

| | Accuracy |
|---|---:|
| Batch statistics (train mode) | **93.36%** |
| Running statistics (eval mode) | 65.62% |

**Why it happens:** BatchNorm keeps *one* set of running statistics. An SNN's
activation distribution **changes at every timestep** — sparse early while
membranes charge, dense later. One mean/variance cannot describe all ten.

**Fix:** BNTT — one BatchNorm per timestep.

**Say this:** "When training accuracy is high and validation isn't, suspect
normalisation before architecture."

### Finding 2: the sparsity penalty is free

**What:** λ=1.0 scored **higher** than λ=0 (89.13% vs 88.05%) while using 3.4×
less energy.

**Why:** the unpenalised network was firing more than the task required. That
surplus carried no information — nothing in the loss discouraged it.

**Honest caveat:** the seed ranges overlap slightly, so the claim is "sparsity
is free, possibly beneficial," not "sparsity improves accuracy."

### Finding 3: a single-seed anomaly evaporated

**What:** λ=0.05 showed 86.36% and looked like a real dip. With three seeds it
is 87.19% ± 1.05, overlapping everything else. It was noise.

**Why it matters:** the published figure was inviting readers to explain a
feature that did not exist. **This is the argument for paying for seed repeats.**

### Finding 4: a dead network reports spectacular savings

An early demo reported **59.7× less energy** — from a network with a **0.0%
firing rate** that computed nothing at all.

**Why:** default thresholds are tuned for dense activations; sparse event input
never charges the membrane enough to fire.

**Lesson:** never report an energy number without the firing rate beside it.
That's why `firing_rate()` is built into the neuron.

---

## 7. Likely questions, and the answers

**"Is the energy real?"**
No — *estimated*, under the Horowitz 45 nm model, ignoring memory traffic which
often dominates real accelerators. Say this before being asked.

**"Why is the SNN doing more operations and still winning?"**
Because each is an add (0.9 pJ), not a multiply-accumulate (4.6 pJ). At T=10 and
21% firing the operation count goes *up*, and energy still goes down.

**"How do you know the comparison is fair?"**
Identical architecture, and parameter parity is enforced by a test.

**"How do you know your LIF is correct?"**
Verified against `snntorch` to 1e-6 on spikes, membrane potential, and gradients.

**"Is this state of the art?"**
No. Above the published SNN reference (CarSNN 86.94%), but single-seed at the
far end, no hyperparameter search, and the protocol isn't matched to theirs.

**"Where's the detection?"**
Doesn't exist. Everything here is classification. Bounding boxes are Rung 3.

**"What's the biggest weakness?"**
Timesteps were never varied. The relation is `5.1/(T×r)` and only `r` was ever
attacked — `T=10` throughout. That's an untested second lever worth ~2×.

---

## 8. Repository map

```
src/data/representations.py   4 event -> tensor conversions
src/data/prophesee.py         .dat reader (N-CARS, GEN1, 1 Mpx)
src/data/datasets.py          dataset wrappers, shuffled subsets
src/models/lif.py             LIF neuron, surrogate gradients, sparsity loss
src/models/classifier.py      paired CNN/SNN, BNTT
src/engine/energy.py          MAC/SynOps counting  <- the contribution
src/engine/train.py           shared train/eval loop
scripts/run_nmnist.py         Rung 1 benchmark
scripts/run_ncars.py          Rung 2 benchmark
scripts/make_figures.py       figures from committed CSVs
scripts/aggregate_results.py  pools every run into result CSVs
scripts/demo_energy.py        shows why an untrained SNN is dead
modal_app.py                  parallel GPU execution
tests/                        129 tests
```

**Why 129 tests on a research project:** with SNNs, a broken gradient and a
merely hard-to-train network look identical from the loss curve. Tests turn a
week of confusion into a red line of output. Several findings above were caught
by tests, not by training.

---

## 9. Three things to say if you only have a minute

1. **"Same architecture on both sides."** Only the neuron type changes. That's
   what makes the number mean anything.
2. **"7.5× less energy for 2.2 accuracy points on real automotive data"** —
   above the published spiking benchmark.
3. **"The sparsity penalty turned out to be free."** It was built assuming a
   trade-off; there wasn't one, because the network had been wasting spikes.
