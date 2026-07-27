"""Run Rung 2 training on Modal GPUs.

The training code is NOT duplicated here -- this file only describes the
environment, mounts the dataset, and calls `scripts/run_ncars.py`. Anything
that runs locally runs here unchanged.

Why Modal for this project: the sparsity sweep is six independent runs, and
Modal executes them in parallel rather than back to back. Six 20-minute runs
finish in 20 minutes instead of two hours, for roughly the cost of one.

    pip install modal
    modal setup                       # one-time browser auth

    # one-time: push the dataset to a persistent volume (~1 GB)
    modal run modal_app.py::upload_dataset

    # single run
    modal run modal_app.py::train --epochs 30

    # the headline experiment: all six lambdas at once
    modal run modal_app.py::sweep

    # pull results back
    modal run modal_app.py::fetch_results
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "event-snn-detection"
LOCAL_ROOT = Path(__file__).parent

# T4 is plenty for N-CARS (120x100 crops) and the cheapest option that is
# still ~20x a laptop. Switch to "A10G" or "L4" for Rung 3 resolutions.
GPU = "T4"
TIMEOUT_S = 60 * 60 * 3  # a full 30-epoch N-CARS run is ~1h on a T4

# CUDA wheels must come from PyTorch's index -- the default PyPI torch is
# CPU-only on some platforms, and a silently-CPU run wastes the whole budget.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2", "torchvision>=0.17",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy>=1.24", "pyyaml>=6.0", "h5py>=3.10", "hdf5plugin>=4.1",
        "snntorch>=0.9", "tonic>=1.6", "pycocotools>=2.0.7", "thop>=0.1.1",
        "opencv-python-headless>=4.9", "matplotlib>=3.8", "tqdm>=4.66",
        "pytest>=8.0", "py7zr>=1.0",
    )
    # Ship the repo source. Excludes keep the image small -- the dataset lives
    # in a Volume, not the image, so it is uploaded once rather than per build.
    .add_local_dir(
        LOCAL_ROOT, remote_path="/root/project",
        ignore=[".venv", "data", "runs", ".git", "__pycache__",
                "*.pyc", "docs/figures", ".pytest_cache"],
    )
)

app = modal.App(APP_NAME, image=image)

# Persistent storage. `data` holds N-CARS (uploaded once); `runs` collects
# checkpoints and summaries so results survive container shutdown.
data_volume = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)
runs_volume = modal.Volume.from_name(f"{APP_NAME}-runs", create_if_missing=True)

VOLUMES = {"/data": data_volume, "/runs": runs_volume}


def _run_training(args: list[str], volume=None, commit_every_s: int = 300) -> dict:
    """Invoke scripts/run_ncars.py as a subprocess and return its summary.

    Commits the runs volume periodically while training, not just at the end.

    This matters because Modal runs on preemptible capacity. A volume write is
    only durable once `commit()` is called; until then it lives in the
    container's view. A first attempt at this sweep lost ~7 GPU-hours of work
    when containers were reclaimed at epoch 24 of 30 -- checkpoints had been
    written every time validation improved, but none had been committed, so
    every one was discarded with the container.
    """
    import subprocess
    import sys
    import threading
    import time

    cmd = [sys.executable, "scripts/run_ncars.py", "--data-root", "/data/ncars", *args]
    print("+ " + " ".join(cmd), flush=True)
    env = {
        "PYTHONPATH": "/root/project", "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/root", "MPLCONFIGDIR": "/tmp/mpl",
    }
    proc = subprocess.Popen(cmd, cwd="/root/project", env=env)

    stop = threading.Event()

    def _periodic_commit() -> None:
        while not stop.wait(commit_every_s):
            try:
                (volume or runs_volume).commit()
            except Exception as exc:  # never let a commit failure kill training
                print(f"[commit] skipped: {exc}", flush=True)

    committer = threading.Thread(target=_periodic_commit, daemon=True)
    committer.start()
    try:
        returncode = proc.wait()
    finally:
        stop.set()
        committer.join(timeout=5)

    if returncode != 0:
        raise RuntimeError(f"training failed with exit code {returncode}")

    out_dir = next((a for i, a in enumerate(args) if args[i - 1] == "--out"), "runs/ncars")
    summary_path = Path("/root/project") / out_dir / "summary.json"
    return json.loads(summary_path.read_text()) if summary_path.exists() else {}


@app.function(volumes=VOLUMES, timeout=TIMEOUT_S)
def extract_dataset(archive: str = "/data/download.zip") -> str:
    """Unpack the Prophesee archive inside the volume.

    Uploading the 285 MB archive and extracting here is far faster than pushing
    24,029 individual .dat files over the wire -- per-file overhead dominates
    at that count.

    Handles the shipped nesting: download.zip -> "NCARS ... Dataset/" ->
    Prophesee_Dataset_n_cars.7z -> the actual split directories.
    """
    import zipfile

    import py7zr

    src = Path(archive)
    if not src.exists():
        return f"{src} not found. Upload it first with `modal volume put`."

    work = Path("/data/_unpack")
    work.mkdir(parents=True, exist_ok=True)

    if src.suffix == ".zip":
        print(f"unzipping {src}...", flush=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(work)
        sevenz = next(work.rglob("*.7z"), None)
    else:
        sevenz = src

    if sevenz is None:
        return f"no .7z found inside {src}"

    print(f"extracting {sevenz.name} (24k files, takes a minute)...", flush=True)
    with py7zr.SevenZipFile(sevenz, "r") as z:
        z.extractall("/data/ncars/")

    # Drop the intermediate copy so the volume holds only the dataset.
    import shutil

    shutil.rmtree(work, ignore_errors=True)
    data_volume.commit()

    n = len(list(Path("/data/ncars").rglob("*.dat")))
    return f"extracted {n} .dat files (expected 24029)"


@app.function(volumes=VOLUMES, timeout=TIMEOUT_S)
def upload_dataset() -> str:
    """One-time: verify the dataset landed in the volume.

    The upload itself is done from the CLI, which streams far faster than
    pushing 24k small files through a function call:

        modal volume put event-snn-detection-data \\
            data/ncars/Prophesee_Dataset_n_cars /ncars/Prophesee_Dataset_n_cars
    """
    root = Path("/data/ncars/Prophesee_Dataset_n_cars")
    if not root.is_dir():
        return (
            "Dataset not found. Run this from the repo root:\n"
            f"  modal volume put {APP_NAME}-data "
            "data/ncars/Prophesee_Dataset_n_cars /ncars/Prophesee_Dataset_n_cars"
        )
    counts = {
        f"{split}/{cls}": len(list((root / f"n-cars_{split}" / cls).glob("*.dat")))
        for split in ("train", "test")
        for cls in ("cars", "background")
    }
    total = sum(counts.values())
    return f"{counts}  total={total} (expected 24029)"


@app.function(gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT_S)
def train(
    epochs: int = 30,
    batch_size: int = 64,
    sparsity_lambda: float = 0.0,
    num_steps: int = 10,
    skip_cnn: bool = False,
    subset: int = 0,          # 0 = use the full dataset
    tag: str = "ncars",
) -> dict:
    """One CNN+SNN comparison run."""
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    out = f"/runs/{tag}"
    args = [
        "--epochs", str(epochs), "--batch-size", str(batch_size),
        "--num-steps", str(num_steps),
        "--sparsity-lambda", str(sparsity_lambda),
        "--out", out, "--workers", "2",
    ]
    if skip_cnn:
        args.append("--skip-cnn")
    if subset > 0:
        # Note: a plain int rather than Optional[int] -- Modal builds its CLI
        # from type hints and cannot parse PEP 604 unions on Python 3.9.
        args += ["--subset", str(subset)]

    summary = _run_training(args)
    runs_volume.commit()
    return summary


@app.function(gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT_S)
def train_one_lambda(lam: float, epochs: int = 30) -> dict:
    """A single point on the Pareto front. Fanned out by `sweep`."""
    import torch

    print(f"lambda={lam} on {torch.cuda.get_device_name(0)}", flush=True)
    summary = _run_training([
        "--epochs", str(epochs), "--batch-size", "64", "--num-steps", "10",
        "--sparsity-lambda", str(lam), "--skip-cnn",
        "--out", f"/runs/ncars_lambda_{lam}", "--workers", "2",
    ])
    runs_volume.commit()
    return {"sparsity_lambda": lam, **summary}


@app.local_entrypoint()
def sweep(epochs: int = 30) -> None:
    """The Rung 2 headline experiment: six lambdas, run in PARALLEL.

    `.map()` fans out across containers, so wall-clock time is one run, not six.
    """
    lambdas = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    print(f"launching {len(lambdas)} runs in parallel on {GPU}...")

    results = list(train_one_lambda.map(lambdas, kwargs={"epochs": epochs}))

    print("\n" + "=" * 72)
    print(f"{'lambda':>8} {'accuracy':>10} {'density':>9} {'energy uJ':>11} {'SynOps M':>10}")
    print("=" * 72)
    for r in sorted(results, key=lambda r: r["sparsity_lambda"]):
        snn = r.get("snn", {})
        print(f"{r['sparsity_lambda']:>8} {snn.get('accuracy', 0) * 100:>9.2f}% "
              f"{snn.get('mean_density', 0) * 100:>8.1f}% "
              f"{snn.get('energy_uj', 0):>11.2f} {snn.get('synops', 0) / 1e6:>10.2f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/ncars_sweep.json").write_text(json.dumps(results, indent=2))
    print("\nwrote results/ncars_sweep.json")


@app.function(gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT_S,
               retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train_point(seed: int, lam: float, epochs: int = 30) -> dict:
    """One (seed, lambda) point. Fanned out by `sweep_seeds`."""
    import torch

    print(f"seed={seed} lambda={lam} on {torch.cuda.get_device_name(0)}", flush=True)
    summary = _run_training([
        "--epochs", str(epochs), "--batch-size", "64", "--num-steps", "10",
        "--sparsity-lambda", str(lam), "--seed", str(seed), "--skip-cnn",
        "--out", f"/runs/v2_s{seed}_l{lam}", "--workers", "2",
    ])
    runs_volume.commit()
    return {"seed": seed, "sparsity_lambda": lam, **summary}


@app.local_entrypoint()
def sweep_seeds(epochs: int = 30) -> None:
    """Seed repeats plus a lambda extension, in one parallel launch.

    Two questions with different sampling needs:

    1. *Is the published result stable?* Needs several seeds, but only on the
       lambda values the claims actually rest on -- lambda=0 (the baseline),
       lambda=0.05 (the unexplained dip), lambda=1.0 (the headline).
    2. *Where does sparsity finally cost accuracy?* Needs lambda pushed well
       past 1.0, but one seed is enough to locate a cliff; error bars can come
       afterwards if the cliff turns out to be interesting.

    A full 3 x 7 grid would be 21 runs for no extra information.

    Budget note: the first sweep (results/ncars_sweep.json) already ran seed 42
    at exactly this config -- 30 epochs, batch 64, 10 timesteps -- so that arm
    is reused rather than recomputed. Only seeds 43 and 44 are launched here.
    That is 9 runs instead of 21, ~$7 instead of ~$17, for the same conclusions.
    """
    # Budget-driven subset. A first attempt at 9 points was preempted after
    # ~7 GPU-hours, so what remains buys ~6 runs. Priorities:
    #   * the lambda extension -- nothing is known past lambda=1.0
    #   * seed repeats at lambda=0.05 only -- that is the disputed dip; the
    #     lambda=0 and lambda=1.0 results are not in question, so paying for
    #     their repeats buys less than finding the cliff.
    extension = [2.0, 5.0, 10.0]
    disputed = 0.05
    points = [(42, l) for l in extension]
    points += [(s, disputed) for s in (43, 44)]

    print(f"launching {len(points)} runs in parallel on {GPU}...")
    results = list(train_point.starmap([(s, l, epochs) for s, l in points]))

    Path("results").mkdir(exist_ok=True)
    Path("results/ncars_sweep_v2.json").write_text(json.dumps(results, indent=2))

    # Seed repeats: mean and spread per lambda.
    print("\n" + "=" * 72)
    print("SEED REPEATS")
    print(f"{'lambda':>8} {'accuracy (mean +- range)':>28} {'density':>9} {'energy uJ':>11}")
    print("=" * 72)
    prior = {}
    prior_path = Path("results/ncars_sweep.json")
    if prior_path.exists():
        for r in json.loads(prior_path.read_text()):
            prior[r["sparsity_lambda"]] = {"seed": 42, **r}

    for lam in [disputed]:
        runs = [r for r in results if r["sparsity_lambda"] == lam]
        if lam in prior:
            runs = runs + [prior[lam]]
        accs = [r["snn"]["accuracy"] * 100 for r in runs if "snn" in r]
        if not accs:
            continue
        mean = sum(accs) / len(accs)
        spread = (max(accs) - min(accs)) / 2
        dens = sum(r["snn"]["mean_density"] for r in runs) / len(runs) * 100
        energy = sum(r["snn"]["energy_uj"] for r in runs) / len(runs)
        print(f"{lam:>8} {mean:>19.2f}% +- {spread:<5.2f} {dens:>8.1f}% {energy:>11.2f}")

    # Extension: where does it break?
    print("\n" + "=" * 72)
    print("LAMBDA EXTENSION (seed 42)")
    print(f"{'lambda':>8} {'accuracy':>10} {'density':>9} {'energy uJ':>11}")
    print("=" * 72)
    for lam in [1.0] + extension:
        runs = [r for r in results if r["sparsity_lambda"] == lam and r["seed"] == 42]
        if not runs or "snn" not in runs[0]:
            continue
        s = runs[0]["snn"]
        print(f"{lam:>8} {s['accuracy'] * 100:>9.2f}% {s['mean_density'] * 100:>8.1f}% "
              f"{s['energy_uj']:>11.2f}")

    print("\nwrote results/ncars_sweep_v2.json")


@app.local_entrypoint()
def sweep_fill(epochs: int = 30) -> None:
    """The seed repeats deferred from `sweep_seeds` when budget was tight.

    Completes the 3-seed picture at lambda=0 (the baseline) and lambda=1.0 (the
    headline), so every claim in the README carries a spread rather than a
    single run. Seed 42 for both already exists in results/ncars_sweep.json.
    """
    points = [(s, l) for s in (43, 44) for l in (0.0, 1.0)]
    print(f"launching {len(points)} fill-in runs on {GPU}...")
    results = list(train_point.starmap([(s, l, epochs) for s, l in points]))

    Path("results").mkdir(exist_ok=True)
    Path("results/ncars_sweep_fill.json").write_text(json.dumps(results, indent=2))
    for r in sorted(results, key=lambda r: (r["sparsity_lambda"], r["seed"])):
        snn = r.get("snn", {})
        print(f"  seed {r['seed']} lambda {r['sparsity_lambda']}: "
              f"{snn.get('accuracy', 0) * 100:.2f}%  "
              f"density {snn.get('mean_density', 0) * 100:.1f}%  "
              f"{snn.get('energy_uj', 0):.2f} uJ")
    print("\nwrote results/ncars_sweep_fill.json")


@app.function(volumes=VOLUMES)
def list_results() -> list[str]:
    """What is currently in the runs volume."""
    return sorted(str(p.relative_to("/runs")) for p in Path("/runs").rglob("summary.json"))


@app.local_entrypoint()
def fetch_results() -> None:
    """Copy every summary.json out of the volume into local results/."""
    import subprocess

    names = list_results.remote()
    if not names:
        print("no results in the volume yet")
        return
    Path("results/modal").mkdir(parents=True, exist_ok=True)
    for name in names:
        local = Path("results/modal") / name.replace("/", "_")
        subprocess.run(
            ["modal", "volume", "get", "--force", f"{APP_NAME}-runs", f"/{name}", str(local)],
            check=False,
        )
        print(f"  {name} -> {local}")
