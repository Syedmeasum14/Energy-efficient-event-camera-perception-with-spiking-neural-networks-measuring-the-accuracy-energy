# Event-Based Object Detection at Night with Spiking Neural Networks

Do spiking neural networks give up accuracy for energy on automotive event-camera
data — and does that trade-off hold up at night, where RGB cameras struggle?

This repo trains three detectors on the same [DSEC](https://dsec.ifi.uzh.ch/)
driving sequences with the same detection head, changing only the backbone:

| | Backbone | Input representation |
|---|---|---|
| **A** | Dense CNN | Voxel grid |
| **B** | Recurrent (ConvLSTM) | Event chunks |
| **C** | **Spiking (LIF + surrogate gradients)** | Binary spike tensor |

and reports mAP against estimated inference energy, split by **day vs night**.

> **Status:** Rung 1 complete (foundations validated on N-MNIST). Rung 2
> (N-CARS, real automotive event data) is next. See
> [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the plan and
> [`docs/FINDINGS.md`](docs/FINDINGS.md) for measured results.

## Current result (Rung 1, N-MNIST, 8,000-sample subset)

| | accuracy | operations | est. energy | params |
|---|---|---|---|---|
| CNN | 97.85% | 3.34 M MAC | 15.38 uJ | 24,634 |
| SNN | 95.55% | 6.95 M SynOp | 6.25 uJ | 26,221 |

**2.30 points of accuracy for 2.5x less estimated energy.** Energy is estimated
under the Horowitz (2014) 45nm model — 4.6 pJ per MAC, 0.9 pJ per synaptic
operation — not measured on hardware.

Getting there required implementing **BNTT** (per-timestep BatchNorm): with
plain BatchNorm the same SNN scored 66.10%, because one set of running
statistics cannot describe an activation distribution that changes at every
timestep. See [`docs/FINDINGS.md`](docs/FINDINGS.md) F5.

## Why event cameras

A conventional camera samples every pixel on a fixed clock. An event camera's
pixels fire independently, only when the log-intensity they see changes:

```
x, y, t, p   ->   pixel (x,y) saw brightness go up (p=+1) or down (p=-1) at time t
```

The result is microsecond temporal resolution, ~120 dB dynamic range (vs ~60 dB),
and no data at all in static regions. That last property is what an SNN exploits:
no event means no spike means no computation.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then copy the config and point it at your data:

```bash
cp configs/base.yaml configs/local.yaml
```

Edit `configs/local.yaml` and set `dataset.root`. It is gitignored, so the same
repo works unchanged on both a laptop and a training box.

## Tests

```bash
python -m pytest tests/ -q
```

## Layout

```
configs/     base.yaml + gitignored local.yaml override
src/data/    event loading and representations
src/models/  backbones A/B/C + shared detection head
src/engine/  train / eval loops, energy accounting
scripts/     download, visualise, train, benchmark entrypoints
notebooks/   learning notebooks (SNN primer, data exploration)
docs/        project plan and final report
```

## References

- Gallego et al., *Event-based Vision: A Survey*, TPAMI 2020
- Gehrig et al., *DSEC: A Stereo Event Camera Dataset for Driving Scenarios*, RA-L 2021
- Zhu et al., *Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion*, CVPR 2019 (voxel grid)
- Neftci et al., *Surrogate Gradient Learning in Spiking Neural Networks*, IEEE SPM 2019
- Horowitz, *Computing's Energy Problem*, ISSCC 2014 (energy accounting)
