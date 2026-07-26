# Rung 1 — N-MNIST: foundations

**Status: complete.** A toy dataset, deliberately. The point was to build and
validate the measurement apparatus before pointing it at data that matters.

[Rung 2: N-CARS](RUNG2_NCARS.md) → · [Project plan](PROJECT_PLAN.md) · [Findings log](FINDINGS.md)

---

## Result

N-MNIST, 8,000-sample training subset, 15 epochs, 10 timesteps.

| Model | Accuracy | Operations / sample | Est. energy / sample | Params |
|---|---:|---:|---:|---:|
| CNN | 97.85% | 3.34 M MAC | 15.38 uJ | 24,634 |
| SNN + BNTT | 95.55% | 6.95 M SynOp | 6.25 uJ | 26,221 |
| SNN, plain BatchNorm | 66.10% | 8.00 M SynOp | 7.20 uJ | 26,221 |

![Accuracy versus estimated energy on N-MNIST](figures/accuracy-vs-energy-light.png)

## What the model sees

An event camera produces no images. This is one sample as the binary spike
tensor the SNN consumes — each timestep almost entirely empty, which is exactly
the sparsity the energy argument rests on.

![One event sample accumulated and as individual timesteps](figures/event-representations-light.png)

![Held-out predictions with confidence and ground truth](figures/predictions-light.png)

## The finding: plain BatchNorm costs 29 accuracy points

The SNN first scored **66.10%** while reaching 96.70% on the training set —
textbook overfitting, apparently. It wasn't.

![Validation accuracy over training for all three models](figures/training-curves-light.png)

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
wrong. Both are kept in [FINDINGS.md](FINDINGS.md) with what ruled them out.

## Where the energy goes

![Operations and energy compared](figures/energy-breakdown-light.png)

The SNN performs **more** operations than the CNN and still wins, because each
is an accumulate rather than a multiply-accumulate:

```
advantage  ~  (E_MAC / E_SOP) / (timesteps x firing rate)  =  5.1 / (T x r)
```

The 5.1x from binary spikes is free. Everything after is a fight to keep
`T x r` small.

![Firing rate across training](figures/firing-rate-light.png)

At `T=10` and `r~21%` the denominator is ~2.1 — which is where the 2.5x comes
from, and why density became the lever pulled in Rung 2.

## What Rung 1 built

| | |
|---|---|
| [`src/models/lif.py`](../src/models/lif.py) | LIF neurons and surrogate gradients, written from scratch and verified against `snntorch` to 1e-6 |
| [`src/models/classifier.py`](../src/models/classifier.py) | Paired CNN/SNN with enforced parameter parity, plus BNTT |
| [`src/engine/energy.py`](../src/engine/energy.py) | MAC/SynOps counting via forward hooks, so sparsity is measured rather than assumed |
| [`src/engine/train.py`](../src/engine/train.py) | Shared train/eval loop |
| [`src/data/representations.py`](../src/data/representations.py) | Four event→tensor conversions |

### Why the LIF neuron is hand-written

`snntorch` provides all of it. Implementing it once anyway means that when a
deep SNN stops learning, the three usual causes — membrane decaying too fast,
neurons falling silent, surrogate gradient vanishing — are inspectable rather
than hidden behind a library call. The equivalence tests keep it honest.

## Reproducing

```bash
PYTHONPATH=. python scripts/run_nmnist.py --epochs 15 --num-steps 10 --lr 5e-3
```

Regenerate the figures from committed CSVs, no training required:

```bash
PYTHONPATH=. python scripts/make_figures.py
```

See why a randomly-initialised SNN is usually dead:

```bash
PYTHONPATH=. python scripts/demo_energy.py
```

## Known gaps

- 8,000-sample subset, not the full 60,000.
- Single seed.
- N-MNIST is handwritten digits recorded off a screen — genuine event data with
  real temporal structure, but not a driving task. That is what
  [Rung 2](RUNG2_NCARS.md) is for.
