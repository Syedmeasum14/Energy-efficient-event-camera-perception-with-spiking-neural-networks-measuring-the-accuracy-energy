# Setting up on a CUDA training machine

Written for an RTX 3070 (8 GB) but nothing here is card-specific. Windows
commands are given first, Linux/macOS equivalents alongside.

The workflow is: **author on the laptop, push, pull here, train.** All paths
come from CLI flags or `configs/local.yaml`, so nothing needs editing per
machine.

---

## 1. Prerequisites

| | Check with | Notes |
|---|---|---|
| Python 3.9+ | `python --version` | 3.10 or 3.11 recommended |
| Git | `git --version` | |
| NVIDIA driver | `nvidia-smi` | Must print your GPU and a CUDA version |

If `nvidia-smi` is not found, install the NVIDIA driver first — nothing below
will use the GPU without it. You do **not** need the CUDA Toolkit separately;
the PyTorch wheel bundles what it needs.

## 2. Clone

```bash
git clone https://github.com/Syedmeasum14/Energy-efficient-event-camera-perception-with-spiking-neural-networks-measuring-the-accuracy-energy.git event-snn-detection
cd event-snn-detection
```

The trailing `event-snn-detection` clones into a short directory name rather
than one matching the very long repo name.

## 3. Virtual environment

```bash
python -m venv .venv
```

Activate it — **this differs per platform**:

| Platform | Command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows CMD | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

If PowerShell blocks the script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

Your prompt should now start with `(.venv)`.

## 4. Install PyTorch **with CUDA**

This is the step that matters. The default `pip install torch` gives a
**CPU-only** build on Windows — it will run, slowly, and silently never touch
the GPU.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

If that CUDA version does not match your driver, use the selector at
<https://pytorch.org/get-started/locally/> and take the command it gives you.

Then the rest:

```bash
pip install -r requirements.txt
```

## 5. Verify the GPU is actually visible

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True NVIDIA GeForce RTX 3070`.

**If it prints `False`, stop and fix it** — everything will otherwise train on
CPU roughly 20x slower while appearing to work. The usual cause is a CPU-only
wheel from step 4; `pip uninstall torch torchvision` and reinstall with the
`--index-url`.

## 6. Run the tests

```bash
python -m pytest tests/ -q
```

129 tests. The N-CARS ones skip until the dataset is present, which is next.

## 7. Get the dataset

Two options.

**Copy from the laptop** (faster — the archive is 285 MB versus a 1 GB
extraction). Copy `Prophesee_Dataset_n_cars.7z` across, then place it so the
final layout is:

```
data/ncars/Prophesee_Dataset_n_cars/
    n-cars_train/cars/*.dat
    n-cars_train/background/*.dat
    n-cars_test/cars/*.dat
    n-cars_test/background/*.dat
```

**Or re-download** from <https://www.prophesee.ai/2018/03/13/dataset-n-cars/>
(form-gated). The download is a `.zip` containing a `.7z`.

Extract the `.7z` with 7-Zip, or with Python:

```bash
pip install py7zr && python -c "import py7zr; py7zr.SevenZipFile('Prophesee_Dataset_n_cars.7z').extractall('data/ncars/')"
```

Verify — this should report 24029:

```bash
python -c "from src.data.datasets import NCarsDataset; print(len(NCarsDataset('data/ncars', True)) + len(NCarsDataset('data/ncars', False)))"
```

## 8. Set PYTHONPATH and train

`PYTHONPATH` lets the scripts import `src`. It is set differently per shell:

| Shell | Command |
|---|---|
| PowerShell | `$env:PYTHONPATH="."` |
| Windows CMD | `set PYTHONPATH=.` |
| Linux / macOS | `export PYTHONPATH=.` |

Smoke test first — under a minute, confirms the whole path works:

```bash
python scripts/run_ncars.py --subset 400 --epochs 2 --batch-size 16 --out runs/smoke
```

Then the real run:

```bash
python scripts/run_ncars.py --epochs 30 --batch-size 64 --out runs/ncars
```

On a 3070 expect a few seconds per epoch. If you hit CUDA out-of-memory, halve
`--batch-size`; 8 GB is comfortable at 64 for N-CARS.

## 9. The sparsity sweep

This is Rung 2's headline experiment — it traces the accuracy/energy front by
varying how hard the loss penalises spiking.

PowerShell:

```powershell
foreach ($L in 0, 0.01, 0.05, 0.1, 0.5, 1.0) { python scripts/run_ncars.py --epochs 30 --sparsity-lambda $L --skip-cnn --out "runs/ncars_lambda_$L" }
```

Linux / macOS:

```bash
for L in 0 0.01 0.05 0.1 0.5 1.0; do python scripts/run_ncars.py --epochs 30 --sparsity-lambda $L --skip-cnn --out runs/ncars_lambda_$L; done
```

Each run writes `summary.json` with accuracy, SynOps, energy, and mean spike
density — the four numbers the Pareto figure is built from.

## 10. Send results back

`runs/` and `data/` are gitignored, so commit the summaries explicitly:

```bash
git add -f runs/ncars*/summary.json && git commit -m "N-CARS results from RTX 3070" && git push
```

Then the figures regenerate on any machine:

```bash
python scripts/make_figures.py
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False` | CPU-only wheel | Reinstall with `--index-url .../cu124` |
| `ModuleNotFoundError: src` | `PYTHONPATH` not set | See step 8 |
| `CUDA out of memory` | Batch too large | Halve `--batch-size` |
| N-CARS tests skip | Dataset missing or misplaced | Check the layout in step 7 |
| Training is slow but works | Running on CPU | Re-check step 5 |
| PowerShell blocks activation | Execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
