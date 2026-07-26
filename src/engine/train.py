"""Training and evaluation loops, shared by the CNN and SNN.

One loop for both models. The only branch is how a batch is shaped: the CNN
takes (B, C, H, W), the SNN takes (T, B, C, H, W). Everything else -- optimiser,
schedule, metrics, checkpointing -- is identical, which is what keeps the
comparison fair.

FIRING-RATE REGULARISATION
--------------------------
The SNN loss has an optional second term:

    loss = cross_entropy  +  lambda * mean(firing_rate)

Without it, nothing stops the network from firing constantly. Dense firing
often gives slightly better accuracy, and it destroys the energy advantage --
the optimiser will happily trade away the only reason you built an SNN, because
nothing in the loss told it not to. This term makes sparsity an explicit
objective. Sweeping `lambda` traces out the accuracy-energy Pareto front, which
is the headline figure of Rung 2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.engine.energy import EnergyCounter, EnergyReport
from src.models.lif import LIF, collect_firing_rates


@dataclass
class EpochResult:
    loss: float = 0.0
    accuracy: float = 0.0
    firing_rate: float = 0.0
    seconds: float = 0.0
    per_layer_rates: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        base = f"loss {self.loss:.4f}  acc {self.accuracy * 100:5.2f}%"
        if self.firing_rate > 0:
            base += f"  fire {self.firing_rate * 100:5.2f}%"
        return f"{base}  ({self.seconds:.1f}s)"


def pick_device(preferred: str = "auto") -> torch.device:
    """auto -> CUDA on the training box, MPS on the Mac, else CPU."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_firing_rate(model: nn.Module) -> float:
    rates = collect_firing_rates(model)
    return sum(rates.values()) / len(rates) if rates else 0.0


def _has_lif(model: nn.Module) -> bool:
    return any(isinstance(m, LIF) for m in model.modules())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    sparsity_lambda: float = 0.0,
    grad_clip: float = 1.0,
    log_every: int = 0,
) -> EpochResult:
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    is_snn = _has_lif(model)

    total_loss = total_correct = total_seen = 0
    rate_sum = 0.0
    rate_batches = 0
    start = time.time()

    for step, (x, y) in enumerate(loader):
        # SNN batches arrive as (B, T, C, H, W) from the dataloader's collate;
        # the model wants time first.
        if is_snn and x.dim() == 5:
            x = x.permute(1, 0, 2, 3, 4).contiguous()
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            if is_snn:
                rate = _mean_firing_rate(model)
                rate_sum += rate
                rate_batches += 1
                if sparsity_lambda > 0:
                    # Penalise firing. Note this uses the *measured* rate, which
                    # is a detached scalar -- it biases the reported loss but
                    # cannot itself produce gradients. For a differentiable
                    # version, penalise the spike tensors directly; that is a
                    # Rung 2 refinement and is called out in the plan.
                    loss = loss + sparsity_lambda * rate

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                # SNNs backpropagate through time, so gradients compound across
                # T steps and blow up readily. Clipping is not optional here.
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch = y.size(0)
        total_loss += loss.item() * batch
        total_correct += (logits.argmax(1) == y).sum().item()
        total_seen += batch

        if log_every and step % log_every == 0:
            print(
                f"    step {step:4d}  loss {loss.item():.4f}  "
                f"acc {total_correct / total_seen * 100:.1f}%"
            )

    return EpochResult(
        loss=total_loss / max(total_seen, 1),
        accuracy=total_correct / max(total_seen, 1),
        firing_rate=rate_sum / rate_batches if rate_batches else 0.0,
        seconds=time.time() - start,
        per_layer_rates=collect_firing_rates(model) if is_snn else {},
    )


def measure_energy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 8,
) -> EnergyReport:
    """Average per-sample energy over a few real batches.

    Real batches, not synthetic input: SynOps depend on actual data sparsity,
    and that is the entire point of measuring rather than estimating.
    """
    model.eval()
    is_snn = _has_lif(model)
    mode = "spiking" if is_snn else "dense"

    timesteps = 1
    counter = EnergyCounter(model, mode=mode)
    with counter, torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            if is_snn and x.dim() == 5:
                x = x.permute(1, 0, 2, 3, 4).contiguous()
                timesteps = x.shape[0]
            model(x.to(device))

    report = counter.report(timesteps=timesteps)
    # Hooks accumulate across batches; normalise to a per-sample figure.
    seen = min(num_batches, len(loader))
    if seen > 0:
        report.total_synops /= seen
    return report


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    sparsity_lambda: float = 0.0,
    device: torch.device | None = None,
    checkpoint_dir: str | Path | None = None,
    tag: str = "model",
) -> dict:
    """Train, validate each epoch, keep the best checkpoint by val accuracy."""
    device = device or pick_device()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list] = {"train": [], "val": []}
    best_acc = 0.0

    print(f"[{tag}] training on {device} for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        train_res = run_epoch(
            model, train_loader, device, optimizer, sparsity_lambda=sparsity_lambda
        )
        val_res = run_epoch(model, val_loader, device)
        scheduler.step()

        history["train"].append(train_res)
        history["val"].append(val_res)
        print(f"  epoch {epoch:3d}  train: {train_res}   val: {val_res}")

        if val_res.accuracy > best_acc:
            best_acc = val_res.accuracy
            if checkpoint_dir:
                path = Path(checkpoint_dir)
                path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": best_acc,
                    },
                    path / f"{tag}_best.pt",
                )

    energy = measure_energy(model, val_loader, device)
    print(f"[{tag}] best val acc {best_acc * 100:.2f}%  |  {energy.summary()}")

    return {"history": history, "best_accuracy": best_acc, "energy": energy}
