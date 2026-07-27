# Rung 2 — N-CARS: real automotive event data

**Status: complete.** This is the result the project's claim rests on.

← [Rung 1: N-MNIST foundations](RUNG1_NMNIST.md) · [Project plan](PROJECT_PLAN.md) · [Findings log](FINDINGS.md)

---

## The benchmark

**N-CARS** (Sironi et al., [HATS](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf), CVPR 2018) — recorded with an ATIS
event camera mounted behind a car windshield in real urban driving.

| | |
|---|---|
| Samples | 24,029 (12,336 car / 11,693 background) |
| Split | 15,422 train / 8,607 test |
| Duration | 100 ms each |
| Task | Binary — is there a car in this crop? |
| Crops | Variable size, padded to a 120x100 canvas |

The canvas covers **100% of crops with no clipping**, verified by measuring
max width and height across 600 random files.

## Result

| Model | Accuracy | Operations / sample | Est. energy / sample | Spike density |
|---|---:|---:|---:|---:|
| CNN | **91.37%** | 49.56 M MAC | 227.97 uJ | — |
| SNN (no penalty) | 88.05% +- 0.62 | 114.30 M SynOp | 102.86 uJ | 36.9% |
| **SNN (lambda = 1.0)** | **89.13% +- 0.88** | 33.67 M SynOp | **30.30 uJ** | 21.2% |
| SNN (lambda = 10) | 86.21% | 3.18 M SynOp | 2.87 uJ | 5.8% |

**7.5x less energy for 2.24 points of accuracy** at lambda=1.0, rising to
**79.4x** at lambda=10 for 5.16 points.

![Accuracy versus estimated energy on N-CARS](figures/ncars-pareto-light.png)

Identical architecture on both sides — 4 conv blocks, width 16, ~100k
parameters each. The only differences are the neuron type, the input
representation, and BNTT in place of BatchNorm.

## The sparsity sweep

Nine lambda values across three parallel launches on Modal T4s, 30 epochs each,
15 runs in total. Spreads are the observed half-range over 3 seeds; with n=3 a
standard deviation would imply more than the data supports.

| lambda | accuracy | seeds | spike density | est. energy | vs CNN |
|---|---:|---:|---:|---:|---:|
| 0 | 88.05% +- 0.62 | 3 | 36.9% | 102.86 uJ | 2.2x |
| 0.01 | 87.29% | 1 | 37.4% | 105.63 uJ | 2.2x |
| 0.05 | 87.19% +- 1.05 | 3 | 34.9% | 89.89 uJ | 2.5x |
| 0.1 | 88.24% | 1 | 33.5% | 82.22 uJ | 2.8x |
| 0.5 | 88.44% | 1 | 25.6% | 41.90 uJ | 5.4x |
| 1 | 89.13% +- 0.88 | 3 | 21.2% | 30.30 uJ | 7.5x |
| 2 | 88.67% | 1 | 16.4% | 21.00 uJ | 10.9x |
| 5 | 87.27% | 1 | 9.1% | 6.72 uJ | 33.9x |
| 10 | 86.21% | 1 | 5.8% | 2.87 uJ | 79.4x |

![Sparsity penalty against density, energy and accuracy](figures/ncars-sweep-light.png)

### The penalty does not cost accuracy

lambda=1.0 averages **1.08 points above** lambda=0 while using 3.4x less
energy. The ranges overlap (88.25-90.01 vs 87.43-88.67), so the supportable
claim is that sparsity is **free, and possibly beneficial** — not that the
improvement is established.

Nothing in a plain cross-entropy loss discourages spiking, so the unpenalised
network fires more than the task needs. That surplus carries no information.

### The lambda=0.05 dip was seed noise

The single-seed sweep showed 86.36% and it looked like a real anomaly. At n=3
it is 87.19% +- 1.05, overlapping lambda=0. There was nothing to explain — and
the original figure invited a reader to explain it anyway. That is the argument
for paying for repeats.

### No cliff up to lambda=10

lambda=0 to lambda=10 is a **36x energy cut for 1.84 accuracy points**, degrading
gently the whole way. At the far end the network scores 86.21% at 2.87 uJ —
essentially CarSNN's published 86.94% — at roughly 1/80th of the CNN's energy.

## Predictions on held-out data

![Held-out N-CARS predictions from the sparse SNN](figures/ncars-predictions-light.png)

The lambda=1.0 network -- the sparse one running at 31 uJ -- on held-out crops.
Car bodies, windshields and headlights are legible in the accumulated events
with no image sensor involved.

The two errors are not the same kind of failure. One is a hedge: a textured
background called a car at 58% confidence, sitting on the decision boundary.
The other is **confidently wrong** -- a car called background at 97%, on a crop
carrying 5,593 spikes against a median of 3,504 for these samples. It is not
short of data. Confident errors on data-rich inputs are the failure mode that
matters for safety-critical use, and this project has not measured calibration
at all.

## Against published results

| Method | Accuracy | |
|---|---:|---|
| Our CNN | 91.37% | |
| HATS (Sironi et al., CVPR 2018) | 90.2% | the dataset's own paper |
| **Our SNN** | **89.13%** | mean of 3 seeds |
| [CarSNN](https://arxiv.org/pdf/2107.00401) (Viale et al., IJCNN 2021) | 86.94% | SNN deployed to Loihi |
| Gabor-SNN | 78.9% | |
| HOTS | 62.4% | |

**Not a state-of-the-art claim.** Single runs at one seed, no hyperparameter
search, and the protocol is not matched to CarSNN's (they use an attention
window; we use the full crop). What it establishes is that the implementation
is sound — the SNN is competitive with published spiking work on the same data.

## Reproducing

Locally (needs the dataset — see [SETUP.md](SETUP.md)):

```bash
PYTHONPATH=. python scripts/run_ncars.py --epochs 30 --batch-size 64
```

On Modal, including the sweep fanned across six GPUs:

```bash
modal run --detach modal_app.py::sweep
```

Total compute for every number on this page: ~8 GPU-hours on T4s, about $5.

## What Rung 2 added to the codebase

| | |
|---|---|
| [`src/data/prophesee.py`](../src/data/prophesee.py) | Reader for the Prophesee `.dat` format (14-bit x, 14-bit y, 1-bit polarity). Shared by N-CARS, GEN1 and 1 Mpx, so Rung 3 needs no new parser. |
| [`src/data/datasets.py`](../src/data/datasets.py) | `NCarsDataset`, `load_ncars` |
| [`src/models/lif.py`](../src/models/lif.py) | `firing_rate_loss` — a **differentiable** sparsity penalty |
| [`scripts/run_ncars.py`](../scripts/run_ncars.py) | The benchmark |
| [`modal_app.py`](../modal_app.py) | Parallel GPU execution |

### A bug worth recording

The sparsity penalty originally used the *detached* diagnostic counter, so
`lambda` was added to the reported loss but **produced no gradient at all**.
It did nothing. Every sweep point would have been identical, and the flatness
would have looked like a finding.

Now accumulated from the spike tensors themselves, kept as a running scalar so
memory stays O(1) rather than O(T x batch x neurons). Guarded by a test that
trains with and without the penalty and asserts firing actually drops.

## Known gaps

- **lambda=2, 5 and 10 are single-seed**, so the far end of the front —
  including the 79x figure — carries no error bars.
- **The CNN baseline is single-seed**, and it is the denominator of every
  energy ratio quoted here.
- **Timesteps were never varied.** The relation is `5.1/(T x r)`; only `r` was
  ever attacked, with `T=10` throughout. Halving `T` is an untested second
  lever worth roughly another 2x.
- **Calibration is unmeasured** — see the 97%-confident error above.
- Energy is **estimated** under the Horowitz (2014) 45 nm model, not measured
  on hardware. Memory traffic is ignored and often dominates in practice.

## Next

[Rung 3](PROJECT_PLAN.md) — object detection with bounding boxes on GEN1 or
DSEC: mAP, latency, and the day/night split.
