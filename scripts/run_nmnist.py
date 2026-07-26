"""Rung 1 capstone: CNN vs SNN on N-MNIST, with energy accounting.

This is the template every later experiment follows. Rung 2 changes the dataset
and the class count; nothing else here needs to move.

Smoke test on a laptop (a few minutes, low accuracy -- only checks the pipeline):

    PYTHONPATH=. python scripts/run_nmnist.py --subset 500 --epochs 2

Full run on the training box:

    PYTHONPATH=. python scripts/run_nmnist.py --epochs 15 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.datasets import load_nmnist
from src.engine.energy import compare
from src.engine.train import fit, measure_energy, pick_device
from src.models.classifier import CNNClassifier, SNNClassifier, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/nmnist", help="download/cache location")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--num-bins", type=int, default=5, help="CNN voxel-grid bins")
    p.add_argument("--num-steps", type=int, default=10, help="SNN timesteps")
    p.add_argument("--width", type=int, default=16, help="base channel width")
    p.add_argument("--blocks", type=int, default=3, help="conv blocks")
    p.add_argument(
        "--sparsity-lambda",
        type=float,
        default=0.0,
        help="firing-rate penalty; sweep this to trace the energy/accuracy front",
    )
    p.add_argument(
        "--plain-bn",
        action="store_true",
        help="use plain BatchNorm instead of BNTT (reproduces the 28pp failure)",
    )
    p.add_argument("--subset", type=int, default=None, help="use only N training samples")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/nmnist")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"loading N-MNIST from {args.data_root} (downloads ~1 GB on first run)\n")

    # ---------------------------------------------------------------- CNN ---
    # The CNN sees a voxel grid: time collapsed into num_bins channels.
    train_cnn, test_cnn, (w, h) = load_nmnist(
        args.data_root, representation="voxel_grid",
        num_bins=args.num_bins, subset=args.subset,
    )
    cnn_train = DataLoader(
        train_cnn, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    cnn_test = DataLoader(test_cnn, batch_size=args.batch_size, num_workers=args.workers)

    cnn = CNNClassifier(
        in_channels=train_cnn.in_channels, num_classes=10,
        width=args.width, num_blocks=args.blocks,
    )
    print(f"sensor {w}x{h}  |  CNN params {count_parameters(cnn):,}")
    cnn_result = fit(
        cnn, cnn_train, cnn_test, epochs=args.epochs, lr=args.lr,
        device=device, checkpoint_dir=out_dir, tag="cnn",
    )

    # ---------------------------------------------------------------- SNN ---
    # The SNN sees the same events as a binary spike tensor over num_steps.
    train_snn, test_snn, _ = load_nmnist(
        args.data_root, representation="snn_spikes",
        num_steps=args.num_steps, subset=args.subset,
    )
    snn_train = DataLoader(
        train_snn, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    snn_test = DataLoader(test_snn, batch_size=args.batch_size, num_workers=args.workers)

    snn = SNNClassifier(
        in_channels=train_snn.in_channels, num_classes=10,
        width=args.width, num_blocks=args.blocks,
        # BNTT: one BatchNorm per timestep. Plain BatchNorm cost 28 accuracy
        # points here, because eval-mode running statistics matched no single
        # timestep's distribution. Pass --plain-bn to reproduce that failure.
        num_steps=None if args.plain_bn else args.num_steps,
    )
    print(f"\nSNN params {count_parameters(snn):,}")
    snn_result = fit(
        snn, snn_train, snn_test, epochs=args.epochs, lr=args.lr,
        sparsity_lambda=args.sparsity_lambda,
        device=device, checkpoint_dir=out_dir, tag="snn",
    )

    # ------------------------------------------------------------- report ---
    cnn_energy = measure_energy(cnn, cnn_test, device)
    snn_energy = measure_energy(snn, snn_test, device)

    print("\n" + "=" * 62)
    print("RESULT")
    print("=" * 62)
    print(f"  CNN accuracy   {cnn_result['best_accuracy'] * 100:6.2f}%")
    print(f"  SNN accuracy   {snn_result['best_accuracy'] * 100:6.2f}%")
    drop = (cnn_result["best_accuracy"] - snn_result["best_accuracy"]) * 100
    print(f"  accuracy cost  {drop:+6.2f} pp\n")
    print(compare(cnn_energy, snn_energy))
    print("\n  (estimated energy, Horowitz 45nm model -- not a hardware measurement)")

    summary = {
        "args": vars(args),
        "cnn": {
            "accuracy": cnn_result["best_accuracy"],
            "macs": cnn_energy.total_macs,
            "energy_uj": cnn_energy.energy_uj,
            "params": count_parameters(cnn),
        },
        "snn": {
            "accuracy": snn_result["best_accuracy"],
            "synops": snn_energy.total_synops,
            "energy_uj": snn_energy.energy_uj,
            "mean_density": snn_energy.mean_density,
            "timesteps": args.num_steps,
            "params": count_parameters(snn),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
