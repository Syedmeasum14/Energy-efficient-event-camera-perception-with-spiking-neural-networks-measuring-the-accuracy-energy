# Formulas and references

Every quantity this project computes, with its equation, its source, and where
it is implemented. Nothing here is invented — each formula is either standard
textbook arithmetic (marked *standard*) or taken from a cited paper.

| # | Quantity | Source | Implemented in |
|---|---|---|---|
| [1](#1-event-histogram) | Event histogram | *standard* | `representations.py` |
| [2](#2-voxel-grid) | Voxel grid | Zhu et al., CVPR 2019 | `representations.py` |
| [3](#3-time-surface) | Time surface | Lagorce et al., TPAMI 2017 | `representations.py` |
| [4](#4-binary-spike-tensor) | Binary spike tensor | *standard* | `representations.py` |
| [5](#5-lif-neuron-dynamics) | LIF dynamics | Gerstner & Kistler, 2002 | `lif.py` |
| [6](#6-surrogate-gradient) | Surrogate gradient (arctan) | Neftci et al. 2019; Fang et al. ICCV 2021 | `lif.py` |
| [7](#7-learnable-decay) | Learnable decay `beta` | Fang et al., ICCV 2021 | `lif.py` |
| [8](#8-bntt-batch-normalization-through-time) | BNTT | Kim & Panda, 2021 | `classifier.py` |
| [9](#9-mac-count) | MAC count | *standard* | `energy.py` |
| [10](#10-synaptic-operations-synops) | SynOps | Merolla et al. 2014; Rueckauer et al. 2017 | `energy.py` |
| [11](#11-energy-model) | Energy model | Horowitz, ISSCC 2014 | `energy.py` |
| [12](#12-energy-ratio) | Energy ratio `5.1/(T·r)` | *derived* (this repo) | — |
| [13](#13-firing-rate) | Firing rate | *standard* | `lif.py` |
| [14](#14-sparsity-penalty) | Sparsity penalty | Regularisation, *standard* form | `lif.py`, `train.py` |
| [15](#15-seed-spread) | Seed spread (half-range) | *standard* | `aggregate_results.py` |

---

## Event representations

Given a set of events $\{(x_i, y_i, t_i, p_i)\}$ over a window, with polarity
$p_i \in \{-1, +1\}$ and sensor size $H \times W$.

### 1. Event histogram

$$H_{c,y,x} = \sum_i \mathbb{1}[x_i = x] \cdot \mathbb{1}[y_i = y] \cdot \mathbb{1}[c_i = c]$$

Count events per pixel, split into two polarity channels. Time is discarded
entirely. *Standard* — no citation needed.

**Implemented:** `events_to_histogram` ([representations.py:37](../src/data/representations.py#L37))

### 2. Voxel grid

Normalise timestamps to $[0, B-1]$ and distribute each event's polarity into
its two neighbouring temporal bins by **linear interpolation**:

$$t_i^* = \frac{t_i - t_0}{t_{N} - t_0}(B - 1)$$

$$V_{b,y,x} = \sum_i p_i \cdot \max\left(0,\ 1 - |b - t_i^*|\right)$$

The interpolation is the point: it preserves sub-bin timing that a hard
assignment would discard.

**Source:** Zhu, Yuan, Chaney & Daniilidis, *Unsupervised Event-Based Learning
of Optical Flow, Depth, and Egomotion*, CVPR 2019.

**Implemented:** `events_to_voxel_grid` ([representations.py:69](../src/data/representations.py#L69))

### 3. Time surface

Exponentially decayed map of the most recent event time per pixel:

$$S_{c,y,x} = \exp\left(-\frac{t_{\text{end}} - t_{\text{last}}(c,y,x)}{\tau}\right)$$

Keeps recency, discards density — the opposite trade to the histogram.
$\tau$ is in the same units as $t$ (microseconds here).

**Source:** Lagorce, Orchard, Galluppi, Shi & Benosman, *HOTS: A Hierarchy of
Event-Based Time-Surfaces for Pattern Recognition*, IEEE TPAMI 2017. Also used
by Sironi et al., *HATS*, CVPR 2018.

**Implemented:** `events_to_time_surface` ([representations.py:135](../src/data/representations.py#L135))

### 4. Binary spike tensor

$$S_{b,c,y,x} = \mathbb{1}\left[\exists\, i : \text{bin}(t_i) = b,\ c_i = c,\ (y_i, x_i) = (y, x)\right]$$

Assignment, **not** accumulation. Multiple events on the same pixel in the same
timestep still yield exactly one spike.

**Why this matters:** the binary constraint is what makes formula 10 valid. Real
valued counts would make every downstream synapse perform a multiply-accumulate
and the entire energy argument would silently become false. Pinned by
`test_snn_input_is_strictly_binary`.

**Implemented:** `events_to_snn_input` ([representations.py:173](../src/data/representations.py#L173))

---

## The spiking neuron

### 5. LIF neuron dynamics

Discrete-time Leaky Integrate-and-Fire, with a soft ("subtract") reset:

$$V[t] = \beta V[t-1] + I[t] - S[t-1]\,\theta$$

$$S[t] = \Theta\left(V[t] - \theta\right) = \begin{cases} 1 & V[t] \geq \theta \\ 0 & \text{otherwise}\end{cases}$$

| Symbol | Meaning | Default |
|---|---|---|
| $V$ | membrane potential | — |
| $\beta$ | decay per timestep, $\beta \in [0,1)$ | 0.95 |
| $\theta$ | firing threshold | 0.5 |
| $I[t]$ | input current (conv output) | — |
| $\Theta$ | Heaviside step | — |

$\beta$ is the **only** source of memory. A hard ("zero") reset
$V[t] \leftarrow V[t](1 - S[t])$ is also implemented.

**Source:** standard discrete LIF — Gerstner & Kistler, *Spiking Neuron Models*,
Cambridge University Press 2002. Verified against `snntorch`'s `Leaky` to 1e-6
(`test_snntorch_equivalence.py`).

**Implemented:** `LIF.forward` ([lif.py:125](../src/models/lif.py#L125))

### 6. Surrogate gradient

$\Theta$ has zero derivative everywhere and is undefined at $0$, so
backpropagation fails. Use the true step **forward** and a smooth surrogate
**backward**. This repo uses the arctan surrogate:

$$\tilde{S}(x) = \frac{1}{\pi}\arctan(\pi \alpha x) + \frac{1}{2}
\qquad\Longrightarrow\qquad
\frac{\partial \tilde{S}}{\partial x} = \frac{\alpha}{1 + (\pi \alpha x)^2}$$

with $x = V - \theta$ and $\alpha = 2.0$ by default. A fast-sigmoid alternative
is also implemented:

$$\frac{\partial \tilde{S}}{\partial x} = \frac{\alpha}{(1 + \alpha|x|)^2}$$

Both peak at the threshold — a neuron sitting near $\theta$ is the one whose
weights most affect whether it fires.

**Sources:** Neftci, Mostafa & Zenke, *Surrogate Gradient Learning in Spiking
Neural Networks*, IEEE Signal Processing Magazine 2019 (the technique); Fang,
Yu, Chen, Masquelier, Huang & Tian, *Incorporating Learnable Membrane Time
Constant to Enhance Learning of Spiking Neural Networks*, ICCV 2021 (the arctan
form, as used in SpikingJelly). Fast sigmoid: Zenke & Ganguli, *SuperSpike*,
Neural Computation 2018.

**Implemented:** `SpikeFunction` ([lif.py:79](../src/models/lif.py#L79)), checked
against the closed form in `test_atan_gradient_matches_closed_form`.

### 7. Learnable decay

$\beta$ is optimised through a sigmoid so it cannot leave $(0,1)$:

$$\beta = \sigma(\text{clamp}(\ell, -8, 8)), \qquad \ell_{\text{init}} = \log\frac{\beta_0}{1-\beta_0}$$

**Why the clamp:** $\sigma(50)$ saturates to *exactly* 1.0 in float32 — a neuron
that never leaks, so $V$ integrates without bound. Caught by
`test_beta_stays_in_range_when_learned`, not by training.

**Source:** learnable time constants — Fang et al., ICCV 2021.

**Implemented:** `LIF.beta` ([lif.py:198](../src/models/lif.py#L198))

### 8. BNTT (Batch Normalization Through Time)

Standard BatchNorm keeps one set of running statistics:

$$\hat{x} = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_{\text{BN}}$$

BNTT keeps a **separate** $(\mu_t, \sigma_t, \gamma_t, \beta_t)$ per timestep:

$$\hat{x}[t] = \gamma_t \frac{x[t] - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}} + \beta_{\text{BN},t}$$

**Why:** an SNN's activation distribution changes at every timestep — sparse
early while membranes charge, dense later. One mean/variance pair cannot
describe all $T$ of them, so eval-mode normalisation is wrong at every step.

**Measured cost of getting this wrong (N-MNIST):** 66.10% with plain BatchNorm
vs 95.55% with BNTT. Diagnosed by evaluating the *same checkpoint* on the *same
data* in train mode (batch statistics, 93.36%) against eval mode (running
statistics, 65.62%) — a 29-point gap with identical weights.

**Sources:** BatchNorm — Ioffe & Szegedy, ICML 2015. BNTT — Kim & Panda,
*Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural
Networks From Scratch*, Frontiers in Neuroscience 2021.

**Implemented:** `BNTT` ([classifier.py:25](../src/models/classifier.py#L25))

---

## Energy accounting

### 9. MAC count

For a `Conv2d` layer with output $O$, input channels $C_{in}$, groups $g$ and
kernel $k_H \times k_W$, **per sample**:

$$\text{MACs} = |O| \cdot \frac{C_{in}}{g} \cdot k_H \cdot k_W$$

For `Linear` with $F_{in}$ input features:

$$\text{MACs} = |O| \cdot F_{in}$$

Determined by architecture alone — a zero input costs the same as a busy one,
because dense hardware computes it regardless.

**Source:** *standard*. Hand-verified in `test_conv2d_macs_match_hand_count`
and `test_linear_macs_match_hand_count`.

**Implemented:** `_dense_macs` ([energy.py:110](../src/engine/energy.py#L110))

### 10. Synaptic operations (SynOps)

A spike triggers work only where it occurs. For layer $l$ at timestep $t$, with
input spike density $r_{l,t}$:

$$\text{SynOps} = \sum_{l}\sum_{t=1}^{T} \text{MACs}_l \cdot r_{l,t},
\qquad r_{l,t} = \frac{1}{|x_l[t]|}\sum \mathbb{1}[x_l[t] \neq 0]$$

$r_{l,t}$ is measured empirically with forward hooks, **not** assumed —
because it depends on the data, which is exactly what the sparsity sweep
manipulates.

**Sources:** the SOP metric — Merolla et al., *A million spiking-neuron
integrated circuit with a scalable communication network and interface*,
Science 2014 (TrueNorth). The ANN/SNN operation comparison methodology —
Rueckauer, Lungu, Hu, Pfeiffer & Liu, *Conversion of Continuous-Valued Deep
Networks to Efficient Event-Driven Networks for Image Classification*,
Frontiers in Neuroscience 2017.

**Implemented:** `EnergyCounter._make_hook` ([energy.py:160](../src/engine/energy.py#L158))

### 11. Energy model

$$E_{\text{CNN}} = \text{MACs} \times E_{\text{MAC}}, \qquad
E_{\text{SNN}} = \text{SynOps} \times E_{\text{SOP}}$$

| Constant | Value | Operation |
|---|---:|---|
| $E_{\text{MAC}}$ | **4.6 pJ** | 32-bit float multiply (3.7) + add (0.9) |
| $E_{\text{SOP}}$ | **0.9 pJ** | 32-bit float add only |

45 nm process. The SNN needs no multiplier because its input is binary:
$w \times 1 = w$.

**Source:** Horowitz, *Computing's Energy Problem (and what we can do about
it)*, IEEE ISSCC 2014. These are the values used throughout the SNN
energy-efficiency literature.

**Worked example** — N-CARS, $\lambda = 1.0$:

```
CNN :  49.56 M MAC   x 4.6 pJ = 227.98 uJ
SNN :  33.67 M SynOp x 0.9 pJ =  30.30 uJ     ratio 7.52x
```

**Implemented:** `EnergyReport.energy_pj` ([energy.py:80](../src/engine/energy.py#L81))

### 12. Energy ratio

Because both models share an architecture, the MAC term cancels:

$$\frac{E_{\text{CNN}}}{E_{\text{SNN}}}
= \frac{\text{MACs} \cdot E_{\text{MAC}}}{\text{MACs} \cdot T \cdot \bar{r} \cdot E_{\text{SOP}}}
= \frac{E_{\text{MAC}} / E_{\text{SOP}}}{T \cdot \bar{r}}
= \frac{5.1}{T \cdot \bar{r}}$$

**A first-order model, not the measurement.** It assumes a single $\bar{r}$
across layers. In practice the first layer sees raw event input at ~5% density
while deeper layers sit near 21%, and early layers are the most expensive — so
the flat average *understates* the saving. Measured 7.5× against a predicted
2.4× at $T{=}10$, $\bar{r}{=}21.2\%$.

**Source:** *derived in this repo* from formulas 9–11. Presented as intuition;
all reported numbers come from the measured counter.

### 13. Firing rate

$$r = \frac{\text{total spikes}}{N_{\text{neurons}} \times T}$$

The primary diagnostic. Interpretation:

| $r$ | Meaning |
|---|---|
| $\approx 0$ | **Dead network.** Reports spectacular energy savings while computing nothing. |
| 0.02 – 0.20 | Healthy and sparse |
| $> 0.5$ | Not sparse; the energy argument is largely gone |

**Never report an energy number without it.** An early demo showed 59.7× from a
network with $r = 0$.

**Implemented:** `LIF.firing_rate` ([lif.py:264](../src/models/lif.py#L259))

### 14. Sparsity penalty

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \frac{1}{L}\sum_{l=1}^{L} \bar{r}_l,
\qquad \bar{r}_l = \frac{1}{T}\sum_{t=1}^{T} \text{mean}(S_l[t])$$

Accumulated from the **spike tensors themselves**, so gradients flow. Kept as a
running scalar, making memory $O(1)$ rather than $O(T \cdot B \cdot N)$.

**Why it exists:** nothing in plain cross-entropy discourages firing. Dense
firing often scores marginally better, so the optimiser will trade away the
entire reason for using an SNN unless told not to.

**A bug worth noting:** the first version used the *detached* diagnostic
counter, so $\lambda$ appeared in the reported loss but produced **no gradient**.
Every sweep point would have been identical and the flatness would have looked
like a finding. Guarded by `test_penalty_actually_reduces_firing`.

**Source:** standard L1-style activity regularisation; the SNN application
follows the firing-rate-regularisation approach common in the literature (e.g.
Zhang et al., *SpikeFPN*, IEEE TCDS 2024).

**Implemented:** `firing_rate_loss` ([lif.py:301](../src/models/lif.py#L301))

### 15. Seed spread

$$\bar{a} = \frac{1}{n}\sum_{i=1}^{n} a_i, \qquad
\text{spread} = \frac{\max_i a_i - \min_i a_i}{2}$$

**Half-range, not standard deviation.** With $n = 3$, a std implies more
distributional information than three runs support; the observed range is
exactly what was seen and nothing more.

Two results are reported as **not separated** when their $\bar{a} \pm
\text{spread}$ intervals overlap. This is a descriptive convention, **not** a
significance test — no hypothesis testing was performed, and $n=3$ would not
support it.

**Implemented:** `aggregate` ([aggregate_results.py:63](../scripts/aggregate_results.py#L61))

---

## Full reference list

1. **Gerstner, W. & Kistler, W.** (2002). *Spiking Neuron Models*. Cambridge University Press. — LIF dynamics
2. **Horowitz, M.** (2014). Computing's Energy Problem (and what we can do about it). *IEEE ISSCC*. — energy constants
3. **Ioffe, S. & Szegedy, C.** (2015). Batch Normalization. *ICML*. — BatchNorm
4. **Merolla, P. et al.** (2014). A million spiking-neuron integrated circuit. *Science*, 345(6197). — SOP metric
5. **Lagorce, X. et al.** (2017). HOTS: A Hierarchy of Event-Based Time-Surfaces. *IEEE TPAMI*. — time surfaces
6. **Rueckauer, B. et al.** (2017). Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks. *Frontiers in Neuroscience*. — SynOps methodology
7. **Sironi, A. et al.** (2018). HATS: Histograms of Averaged Time Surfaces. *CVPR*. — N-CARS dataset
8. **Zenke, F. & Ganguli, S.** (2018). SuperSpike. *Neural Computation*, 30(6). — fast-sigmoid surrogate
9. **Neftci, E., Mostafa, H. & Zenke, F.** (2019). Surrogate Gradient Learning in Spiking Neural Networks. *IEEE Signal Processing Magazine*. — surrogate gradients
10. **Zhu, A. et al.** (2019). Unsupervised Event-Based Learning of Optical Flow, Depth, and Egomotion. *CVPR*. — voxel grid
11. **Gallego, G. et al.** (2020). Event-based Vision: A Survey. *IEEE TPAMI*. — background
12. **Kim, Y. & Panda, P.** (2021). Revisiting Batch Normalization for Training Low-Latency Deep SNNs From Scratch. *Frontiers in Neuroscience*. — BNTT
13. **Fang, W. et al.** (2021). Incorporating Learnable Membrane Time Constant. *ICCV*. — arctan surrogate, learnable decay
14. **Viale, A. et al.** (2021). CarSNN. *IJCNN*. — N-CARS SNN baseline
15. **Zhang, H. et al.** (2024). Automotive Object Detection via Learning Sparse Events by Spiking Neurons. *IEEE TCDS*. — SpikeFPN, energy accounting precedent
