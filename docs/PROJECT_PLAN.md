# Energy-Efficient Event-Based Perception for ADAS

## The question

> On automotive event-camera data, how much accuracy is traded for how much
> inference energy when a dense CNN backbone is replaced with a spiking one —
> and how does that trade-off change with scene conditions?

Every rung below answers a version of this question. The rungs differ in scale,
not in intent.

---

## Rung 1 — SNN foundations (~1 week)

Learn the mechanism on a toy problem. Runs on a laptop CPU.

- LIF neuron implemented **from scratch** (so the dynamics are not a black box)
- Surrogate gradients, and why they are needed at all
- The same thing again in `snntorch`, verified to agree with the scratch version
- The energy counter (MACs vs SynOps) — built here, used in every later rung
- Dataset: N-MNIST

**Artifact:** `notebooks/00_snn_primer.ipynb` + tested modules.
**Not** a portfolio piece. Foundation only.

---

## Rung 2 — N-CARS: the first real project (~3 weeks)

Same question, real automotive data, classification instead of detection.

- **Dataset:** N-CARS (Sironi et al., CVPR 2018). 24,029 samples, 100 ms each,
  12,336 car / 11,693 background. Recorded with an ATIS camera behind a car
  windshield in real urban driving. Pre-split 15,422 train / 8,607 test.
- **Model A:** small CNN on voxel grids.
- **Model C:** identical architecture with LIF neurons, on binary spike tensors.
- **Measured:** accuracy, MACs vs SynOps, estimated energy, firing rate,
  and a `num_steps` sweep showing the accuracy/compute trade-off.

**Reference point:** CarSNN (arXiv 2107.00401) does SNN classification on
N-CARS and deploys to Loihi. Matching their accuracy is the success criterion.
Beating it is not required.

**Artifact:** a complete, honest energy-accuracy study on automotive event data.
This is the portfolio piece. If the project stops here, it still stands up.

---

## Rung 3 — Detection (~6-8 weeks, only after Rung 2 ships)

Add the detection head. ~70% of Rung 2's code carries over.

- **Dataset:** Prophesee GEN1 first (304x240 — roughly 4x faster to iterate than
  DSEC on an 8 GB card, and the dataset every comparable paper uses).
  DSEC only if the day/night analysis below is reached.
- **Models:** dense CNN, recurrent (RVT-lite), spiking.
- **ADAS metrics** beyond mAP:
  - **Latency** — time-to-first-confident-detection, not just forward-pass wall
    clock. This is where SNNs have a structural advantage, and it is the number
    an automotive lab asks about first.
  - **Pedestrian AP specifically** — aggregate mAP hides the safety-critical class.
- **Stretch:** day/night split on DSEC. Hypothesis: night means fewer events
  means sparser input, so SNN energy *drops* at night while the CNN's stays
  flat — the advantage widens exactly where RGB cameras fail. This measurement
  does not appear in the current literature.

---

## Metrics (consistent across all rungs)

**Accuracy** — top-1 for classification; mAP@0.5 and mAP@0.5:0.95 for detection.

**Efficiency**
- Dense: **MACs**, counted analytically per layer.
- Spiking: **SynOps** = sum over layers of (input spikes) x (fan-out), measured
  empirically with forward hooks, since it depends on real input sparsity.
- Energy, Horowitz (2014) 45nm 32-bit accounting:
  - `E_MAC = 4.6 pJ`, `E_SOP = 0.9 pJ`
  - These are **estimates**, not hardware measurements. Every report must say so.

**Firing rate** — mean spikes per neuron per timestep. Diagnostic: near 0 means
the network died, near 1 means it is not being sparse and the energy argument
collapses.

---

## Hardware and workflow

- **MacBook Air M1, 8 GB** — authoring, tests, visualisation, Rung 1.
- **RTX 3070, 8 GB VRAM** — all training. Separate machine.
- Git is the bridge. **No hardcoded paths**: `configs/base.yaml` is committed,
  `configs/local.yaml` is gitignored and holds per-machine paths.
- Every module ships with tests. With SNNs a wrong gradient is indistinguishable
  from "training is hard" — tests are what make that debuggable.

---

## Non-goals

- Beating state of the art. The contribution is a controlled, honest comparison.
- Stereo or depth. Monocular only.
- Real neuromorphic silicon. Simulator at most, and only as a stretch.

## References

- Sironi et al., *HATS*, CVPR 2018 — N-CARS dataset
- Viale et al., *CarSNN*, IJCNN 2021 — SNN on N-CARS + Loihi
- Gehrig & Scaramuzza, *RVT*, CVPR 2023 — event detection reference
- Zhang et al., *SpikeFPN*, IEEE TCDS 2024 — spiking detection + energy accounting
- Neftci et al., *Surrogate Gradient Learning in SNNs*, IEEE SPM 2019
- Horowitz, *Computing's Energy Problem*, ISSCC 2014
