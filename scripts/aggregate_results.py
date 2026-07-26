"""Combine every N-CARS sweep run into the CSVs the figures are built from.

Three launches contributed, all at identical config (30 epochs, batch 64,
10 timesteps) so they pool legitimately:

    results/ncars_sweep.json       seed 42, lambda 0 .. 1.0
    results/ncars_sweep_v2.json    seeds 43/44 at lambda 0.05; seed 42 at 2/5/10
    results/ncars_sweep_fill.json  seeds 43/44 at lambda 0 and 1.0

Writes:
    results/ncars_sweep.csv        per-lambda aggregate (mean, spread, n)
    results/ncars_runs.csv         every individual run, for transparency
    results/ncars_summary.csv      CNN vs the two headline SNN operating points

    PYTHONPATH=. python scripts/aggregate_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SOURCES = [
    ("ncars_sweep.json", 42),        # seed not recorded per-run; it was the default
    ("ncars_sweep_v2.json", None),   # seed carried on each record
    ("ncars_sweep_fill.json", None),
]


def load_runs() -> list[dict]:
    """Flatten every run into {seed, lambda, accuracy, density, energy, synops}."""
    runs: list[dict] = []
    for name, default_seed in SOURCES:
        path = RESULTS / name
        if not path.exists():
            print(f"  (skipping {name} -- not present)")
            continue
        for rec in json.loads(path.read_text()):
            snn = rec.get("snn")
            if not snn:
                continue
            runs.append({
                "seed": rec.get("seed", default_seed),
                "sparsity_lambda": rec["sparsity_lambda"],
                "accuracy": round(snn["accuracy"] * 100, 2),
                "mean_density": round(snn["mean_density"] * 100, 2),
                "energy_uj": round(snn["energy_uj"], 2),
                "synops_m": round(snn["synops"] / 1e6, 2),
            })
    # Deduplicate: a lambda/seed pair should appear once.
    seen: dict[tuple, dict] = {}
    for r in runs:
        seen[(r["sparsity_lambda"], r["seed"])] = r
    return sorted(seen.values(), key=lambda r: (r["sparsity_lambda"], r["seed"]))


def aggregate(runs: list[dict]) -> list[dict]:
    """Per-lambda mean and half-range. Half-range rather than std: with n=3 a
    standard deviation implies more distributional information than three runs
    support, whereas (max-min)/2 is exactly what was observed."""
    by_lambda: dict[float, list[dict]] = {}
    for r in runs:
        by_lambda.setdefault(r["sparsity_lambda"], []).append(r)

    rows = []
    for lam, group in sorted(by_lambda.items()):
        accs = [g["accuracy"] for g in group]
        rows.append({
            "sparsity_lambda": lam,
            "n_seeds": len(group),
            "accuracy": round(sum(accs) / len(accs), 2),
            "accuracy_spread": round((max(accs) - min(accs)) / 2, 2) if len(accs) > 1 else 0.0,
            "accuracy_min": min(accs),
            "accuracy_max": max(accs),
            "mean_density": round(sum(g["mean_density"] for g in group) / len(group), 2),
            "energy_uj": round(sum(g["energy_uj"] for g in group) / len(group), 2),
            "synops_m": round(sum(g["synops_m"] for g in group) / len(group), 2),
        })
    return rows


def write_summary(agg: list[dict]) -> None:
    """CNN plus the two SNN operating points worth naming."""
    cnn = json.loads((RESULTS / "modal" / "ncars_summary.json").read_text())["cnn"]

    baseline = next(r for r in agg if r["sparsity_lambda"] == 0.0)
    # Best accuracy among the sparsity-penalised runs.
    best = max((r for r in agg if r["sparsity_lambda"] > 0), key=lambda r: r["accuracy"])
    # Cheapest run of all -- the far end of the front.
    cheapest = min(agg, key=lambda r: r["energy_uj"])

    with open(RESULTS / "ncars_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "label", "accuracy", "operations_m", "op_kind",
                    "energy_uj", "params", "mean_density", "n_seeds"])
        w.writerow(["cnn", "CNN", round(cnn["accuracy"] * 100, 2),
                    round(cnn["macs"] / 1e6, 2), "MAC", round(cnn["energy_uj"], 2),
                    cnn["params"], "", 1])
        for key, label, r in (
            ("snn", f"SNN (lambda={baseline['sparsity_lambda']:g})", baseline),
            ("snn_best", f"SNN (lambda={best['sparsity_lambda']:g})", best),
            ("snn_cheapest", f"SNN (lambda={cheapest['sparsity_lambda']:g})", cheapest),
        ):
            w.writerow([key, label, r["accuracy"], r["synops_m"], "SynOp",
                        r["energy_uj"], 102118, round(r["mean_density"] / 100, 3),
                        r["n_seeds"]])

    ratio = cnn["energy_uj"] / best["energy_uj"]
    cost = cnn["accuracy"] * 100 - best["accuracy"]
    print(f"\n  headline: lambda={best['sparsity_lambda']:g} -> "
          f"{best['accuracy']:.2f}% at {best['energy_uj']:.2f} uJ "
          f"= {ratio:.1f}x less energy for {cost:.2f} pp")
    print(f"  cheapest: lambda={cheapest['sparsity_lambda']:g} -> "
          f"{cheapest['accuracy']:.2f}% at {cheapest['energy_uj']:.2f} uJ "
          f"= {cnn['energy_uj'] / cheapest['energy_uj']:.1f}x")


def main() -> None:
    runs = load_runs()
    agg = aggregate(runs)

    with open(RESULTS / "ncars_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)

    with open(RESULTS / "ncars_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)

    write_summary(agg)

    print(f"\n  {len(runs)} individual runs -> results/ncars_runs.csv")
    print(f"  {len(agg)} lambda values    -> results/ncars_sweep.csv")
    print(f"                             -> results/ncars_summary.csv")
    seeded = [r for r in agg if r["n_seeds"] > 1]
    print(f"  {len(seeded)} lambda values have seed repeats "
          f"(n={max(r['n_seeds'] for r in agg)} max)")


if __name__ == "__main__":
    main()
