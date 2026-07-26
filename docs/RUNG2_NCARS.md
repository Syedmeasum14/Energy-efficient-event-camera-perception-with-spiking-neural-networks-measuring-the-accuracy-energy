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
| SNN (no penalty) | 88.27% | 120.88 M SynOp | 108.79 uJ | 37.9% |
| **SNN (lambda = 1.0)** | **88.52%** | 34.87 M SynOp | **31.38 uJ** | 22.0% |

**7.3x less energy for 2.9 points of accuracy.**

![Accuracy versus estimated energy on N-CARS](figures/ncars-pareto-light.png)

Identical architecture on both sides — 4 conv blocks, width 16, ~100k
parameters each. The only differences are the neuron type, the input
representation, and BNTT in place of BatchNorm.

## The sparsity sweep

Six values of the firing-rate penalty `lambda`, run in parallel on Modal T4s,
30 epochs each.

| lambda | accuracy | spike density | est. energy | SynOps |
|---|---:|---:|---:|---:|
| 0.0 | 88.50% | 36.7% | 100.65 uJ | 111.83 M |
| 0.01 | 87.29% | 37.4% | 105.63 uJ | 117.36 M |
| 0.05 | 86.36% | 34.9% | 90.06 uJ | 100.07 M |
| 0.1 | 88.24% | 33.5% | 82.22 uJ | 91.35 M |
| 0.5 | 88.44% | 25.6% | 41.90 uJ | 46.55 M |
| **1.0** | **88.52%** | **22.0%** | **31.38 uJ** | **34.87 M** |

![Sparsity penalty against density and energy](figures/ncars-sweep-light.png)

### The penalty is free, which was not the expectation

The whole Pareto framing assumes sparsity must be **bought** with accuracy.
On N-CARS it costs nothing: `lambda=1.0` is marginally *more* accurate than
`lambda=0` (88.52% vs 88.50%) while using **3.2x less energy**.

The unpenalised network was spiking more than the task required — activity
carrying no information. Nothing in the plain cross-entropy loss discouraged
it, so the optimiser had no reason to stop.

This is what moved the headline against the CNN from 2.1x to **7.3x**.

## Against published results

| Method | Accuracy | |
|---|---:|---|
| Our CNN | 91.37% | |
| HATS (Sironi et al., CVPR 2018) | 90.2% | the dataset's own paper |
| **Our SNN** | **88.52%** | |
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

- **Single seed.** The dip at `lambda=0.05` (86.36%) is probably noise —
  `lambda=0.1` recovers — but that needs repeats to say.
- **The front is not fully traced.** Density is still 22% at `lambda=1.0` and
  accuracy never fell, so the point where sparsity finally costs something was
  never found. Higher `lambda` would show it.
- Energy is **estimated** under the Horowitz (2014) 45 nm model, not measured
  on hardware. Memory traffic is ignored and often dominates in practice.

## Next

[Rung 3](PROJECT_PLAN.md) — object detection with bounding boxes on GEN1 or
DSEC: mAP, latency, and the day/night split.
