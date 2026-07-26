# Findings

A running log of things that were measured, not assumed. Each entry records
what was believed, what the experiment showed, and what changed as a result.
Wrong hypotheses are kept — the discarded ones are often the most useful part.

---

## F1. A randomly-initialised SNN is usually dead

**Context:** first end-to-end energy demo, before any training.

**Observation:** every LIF layer reported a 0.0% firing rate with the default
`threshold=1.0` on 5%-sparse input. The energy counter dutifully reported a
**59.7x energy saving** — for a network computing nothing at all.

Threshold sweep, 32x32 input at 5% density, 8 timesteps:

| threshold | layer 1 | layer 2 | status |
|---|---|---|---|
| 1.0 | 0.0% | 0.0% | dead |
| 0.5 | 0.6% | 0.3% | barely alive |
| 0.2 | 6.2% | 24.6% | healthy |
| 0.1 | 14.7% | 38.8% | too dense |

**Consequence:** firing rate is a first-class diagnostic, built into `LIF`
rather than bolted on. **Never report an energy number without the firing rate
beside it.** Reproduce with `scripts/demo_energy.py`.

---

## F2. The energy advantage is not automatic

**Belief going in:** SNNs are dramatically cheaper than CNNs.

**Measured**, identical architecture, 8 timesteps, healthy firing:

```
dense       1.48 M MAC       6.80 uJ
spiking     2.14 M SOP       1.93 uJ
-> 0.7x fewer ops, 3.5x less estimated energy
```

The SNN performed **more** operations than the CNN. It won only because each
operation is ~5x cheaper. The governing relation:

```
advantage  ~  (E_MAC / E_SOP) / (T x firing_rate)  =  5.1 / (T x r)
```

The 5.1x from binary spikes is free. Everything beyond it is a fight to keep
`T x r` small. At `T=8, r=27%` the denominator is 2.2, giving ~2x. At
`T=4, r=8%` it is 0.32, giving ~16x.

**Consequence:** this equation drives the Rung 2 experiment. The question is
not *whether* an SNN saves energy but *where on this curve accuracy survives*.

---

## F3. snntorch applies the spike reset one timestep late

**Observation:** the from-scratch `LIF` produced identical spike trains to
`snntorch.Leaky` but membrane traces differed by **exactly the threshold**.

**Cause:** `snntorch` defaults to `reset_delay=True`, applying the reset at the
start of the *next* timestep to mimic one-cycle hardware latency. Ours resets
immediately.

**Consequence:** both conventions are valid and appear in the literature. Ours
is documented in `lif.py` and pinned by `tests/test_snntorch_equivalence.py`
against `reset_delay=False`. Worth knowing before comparing membrane traces to
any published figure.

---

## F4. MaxPool saturates binary spike maps — but AvgPool is not the fix

**Hypothesis:** MaxPool destroys information in a spiking network, since a
binary map pooled by max transforms density as `d -> 1 - (1-d)^4`.

**Measured** density entering each conv (N-MNIST, 17% input density):

| layer | density |
|---|---|
| `blocks.0.conv` | 17.0% |
| `blocks.1.conv` | 71.7% |
| `blocks.2.conv` | 75.9% |

Saturation confirmed. By block 2, three-quarters of units are active — not a
sparse network.

**But the proposed fix was wrong.** AvgPool gave essentially identical density
(70.9%, 78.4%), because averaging binary values is nonzero wherever any input
is nonzero — same support. Worse, AvgPool emits `{0, .25, .5, .75, 1}`, which
is **not binary**, so the following conv performs real multiplications and the
SynOps accounting silently becomes false.

**Consequence:** MaxPool is retained deliberately — it is the only common
pooling op preserving binarity. The real lever on saturation is the firing rate
itself (threshold, and the sparsity penalty in `train.py`).

---

## F5. Plain BatchNorm costs ~28 accuracy points in an SNN

**Context:** the first full N-MNIST run. CNN reached 97.85%; the SNN plateaued
at 66.10% while reaching **96.70% on the training set**. That looks like
textbook overfitting, and validation accuracy was wildly unstable (37.40% at
epoch 7, 66.10% at epoch 9).

**Decisive test:** evaluate the *same weights* on the *same data*, changing only
the BatchNorm mode.

```
batch statistics   (train mode)   93.36%
running statistics (eval  mode)   65.62%
```

**Cause:** BatchNorm keeps one set of running statistics. An SNN's activation
distribution changes at every timestep — early steps are sparse because
membranes are still charging, later steps are dense. One mean/variance pair
cannot describe all `T` distributions, so eval-mode normalisation is wrong at
every step. A CNN never hits this: one forward pass, one distribution.

**This was not overfitting at all.** The network had learned; the normalisation
statistics were simply wrong at inference.

**Consequence:** implemented **BNTT** — Batch Normalization Through Time
(Kim & Panda, 2021) — one BatchNorm per timestep, in `classifier.py`. Costs
`T x 2 x C` extra parameters (negligible) and nothing at inference. Guarded by
`test_snn_with_bntt_is_stable_between_train_and_eval`. Reproduce the failure
with `scripts/run_nmnist.py --plain-bn`.

**Result after the fix** (N-MNIST, 8,000-sample subset, 15 epochs, T=10):

| | plain BatchNorm | BNTT |
|---|---|---|
| SNN accuracy | 66.10% | **95.55%** |

One change, **+29.45 points**. The CNN baseline is 97.85%, so the true cost of
going spiking on this task is **2.30 points**, not 32.

**Method note:** the two earlier hypotheses for this gap — network capacity and
MaxPool saturation — were both wrong. The train/eval split was the signal that
mattered, and it was visible in the epoch logs from the start. *When train
accuracy is high and validation is not, suspect normalisation before
architecture.*

---

## F6. Rung 1 baseline result

N-MNIST, 8,000-sample subset, identical architecture, 15 epochs.

| | accuracy | operations | est. energy | params |
|---|---|---|---|---|
| CNN | 97.85% | 3.34 M MAC | 15.38 uJ | 24,634 |
| SNN | 95.55% | 6.95 M SynOp | 6.25 uJ | 26,221 |

**2.30 points of accuracy for 2.5x less estimated energy.**

Note the SNN performs *more* operations (6.95 M vs 3.34 M) and still wins,
exactly as F2 predicts: 10 timesteps at ~40% mean density outweighs the input
sparsity, but each operation costs 0.9 pJ instead of 4.6 pJ.

**40% density is the headroom.** Per F2, the advantage is `5.1 / (T x r)`.
Driving density from 40% to 10% — via the sparsity penalty and a higher
threshold — should move this from 2.5x toward 10x. That sweep is the Rung 2
experiment, and it is the reason the sparsity penalty already exists in
`train.py`.

Caveat: a subset, not the full 60,000 samples, and N-MNIST is a toy dataset.
The number to trust is the one from N-CARS in Rung 2.

---

## Open

- Firing-rate penalty in `train.py` uses the *measured* rate, a detached scalar.
  It reports correctly but produces no gradient. A differentiable version must
  penalise the spike tensors directly — required before the Rung 2 lambda sweep.
- N-MNIST results use an 8,000-sample subset, not the full 60,000.
