"""Rung 2: CNN vs SNN on N-CARS, with energy accounting.

N-CARS is real automotive event data -- an ATIS camera behind a car windshield
in urban driving, 24,029 samples of 100 ms, car vs background. This is the
benchmark the project's claim actually rests on; N-MNIST was the rehearsal.

Smoke test on a laptop:

    PYTHONPATH=. python scripts/run_ncars.py --subset 400 --epochs 2 --workers 0

Full run on the training box:

    PYTHONPATH=. python scripts/run_ncars.py --epochs 30 --batch-size 64

Sparsity sweep (the headline experiment -- trace the accuracy/energy front):

    for L in 0 0.01 0.05 0.1 0.5 1.0; do
      PYTHONPATH=. python scripts/run_ncars.py --epochs 30 --sparsity-lambda $L \
          --skip-cnn --out runs/ncars_lambda_$L
    done
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.datasets import load_ncars
from src.engine.energy import compare
from src.engine.train import fit, measure_energy, pick_device
from src.models.classifier import CNNClassifier, SNNClassifier, count_parameters

NUM_CLASSES = 2  # background, car


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/ncars")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--num-bins", type=int, default=5, help="CNN voxel-grid bins")
    p.add_argument("--num-steps", type=int, default=10, help="SNN timesteps")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--blocks", type=int, default=4, help="N-CARS crops are larger than N-MNIST")
    p.add_argument(
        "--sparsity-lambda", type=float, default=0.0,
        help="firing-rate penalty weight; sweep this to trace the Pareto front",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--skip-cnn", action="store_true", help="SNN only (for sweeps)")
    p.add_argument("--plain-bn", action="store_true", help="disable BNTT")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/ncars")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    summary: dict = {"args": vars(args)}

    # ---------------------------------------------------------------- CNN ---
    cnn_energy = None
    if not args.skip_cnn:
        train_cnn, test_cnn, canvas = load_ncars(
            args.data_root, representation="voxel_grid",
            num_bins=args.num_bins, subset=args.subset,
        )
        cnn_train = DataLoader(train_cnn, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers)
        cnn_test = DataLoader(test_cnn, batch_size=args.batch_size, num_workers=args.workers)

        cnn = CNNClassifier(in_channels=train_cnn.in_channels, num_classes=NUM_CLASSES,
                            width=args.width, num_blocks=args.blocks)
        print(f"canvas {canvas[0]}x{canvas[1]}  |  train {len(train_cnn)}  test {len(test_cnn)}")
        print(f"CNN params {count_parameters(cnn):,}")
        cnn_result = fit(cnn, cnn_train, cnn_test, epochs=args.epochs, lr=args.lr,
                         device=device, checkpoint_dir=out_dir, tag="cnn")
        cnn_energy = measure_energy(cnn, cnn_test, device)
        summary["cnn"] = {
            "accuracy": cnn_result["best_accuracy"],
            "macs": cnn_energy.total_macs,
            "energy_uj": cnn_energy.energy_uj,
            "params": count_parameters(cnn),
        }

    # ---------------------------------------------------------------- SNN ---
    train_snn, test_snn, canvas = load_ncars(
        args.data_root, representation="snn_spikes",
        num_steps=args.num_steps, subset=args.subset,
    )
    snn_train = DataLoader(train_snn, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers)
    snn_test = DataLoader(test_snn, batch_size=args.batch_size, num_workers=args.workers)

    snn = SNNClassifier(
        in_channels=train_snn.in_channels, num_classes=NUM_CLASSES,
        width=args.width, num_blocks=args.blocks, threshold=args.threshold,
        num_steps=None if args.plain_bn else args.num_steps,
    )
    print(f"\nSNN params {count_parameters(snn):,}  |  lambda {args.sparsity_lambda}")
    snn_result = fit(snn, snn_train, snn_test, epochs=args.epochs, lr=args.lr,
                     sparsity_lambda=args.sparsity_lambda,
                     device=device, checkpoint_dir=out_dir, tag="snn")
    snn_energy = measure_energy(snn, snn_test, device)
    summary["snn"] = {
        "accuracy": snn_result["best_accuracy"],
        "synops": snn_energy.total_synops,
        "energy_uj": snn_energy.energy_uj,
        "mean_density": snn_energy.mean_density,
        "timesteps": args.num_steps,
        "sparsity_lambda": args.sparsity_lambda,
        "params": count_parameters(snn),
    }

    # ------------------------------------------------------------- report ---
    print("\n" + "=" * 62)
    print("N-CARS RESULT")
    print("=" * 62)
    if cnn_energy is not None:
        print(f"  CNN accuracy   {summary['cnn']['accuracy'] * 100:6.2f}%")
    print(f"  SNN accuracy   {summary['snn']['accuracy'] * 100:6.2f}%")
    if cnn_energy is not None:
        drop = (summary["cnn"]["accuracy"] - summary["snn"]["accuracy"]) * 100
        print(f"  accuracy cost  {drop:+6.2f} pp\n")
        print(compare(cnn_energy, snn_energy))
    else:
        print(f"  {snn_energy.summary()}")
    print("\n  (estimated energy, Horowitz 45nm model -- not a hardware measurement)")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
