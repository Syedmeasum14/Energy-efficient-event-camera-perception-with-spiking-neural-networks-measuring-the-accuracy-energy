"""Event dataset wrappers.

`tonic` handles downloading and parsing standard event datasets (N-MNIST here,
N-CARS in Rung 2) and hands back raw event arrays. This module is the bridge
between those raw events and the tensors our models expect, using the
representations in `src/data/representations.py`.

Keeping the bridge separate from the loaders matters: Rung 2 swaps the dataset
but reuses this conversion untouched, so the CNN and SNN keep seeing data
prepared exactly the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.representations import (
    events_to_histogram,
    events_to_snn_input,
    events_to_time_surface,
    events_to_voxel_grid,
)


class EventTensorDataset(Dataset):
    """Wraps a tonic dataset, converting raw events to model-ready tensors.

    Args:
        base: a tonic dataset yielding (events, target). Events are structured
            numpy arrays with named fields 'x', 'y', 't', 'p'.
        sensor_size: (width, height) of the sensor.
        representation: one of 'voxel_grid', 'histogram', 'time_surface',
            'snn_spikes'. The first three feed the CNN, the last feeds the SNN.
        num_bins / num_steps / tau: representation-specific parameters.
        transform: optional callable applied to the final tensor.
    """

    def __init__(
        self,
        base: Dataset,
        sensor_size: tuple[int, int],
        representation: str = "voxel_grid",
        num_bins: int = 5,
        num_steps: int = 10,
        tau: float = 50_000.0,
        transform: Callable | None = None,
    ):
        self.base = base
        self.width, self.height = sensor_size
        self.representation = representation
        self.num_bins = num_bins
        self.num_steps = num_steps
        self.tau = tau
        self.transform = transform

        valid = {"voxel_grid", "histogram", "time_surface", "snn_spikes"}
        if representation not in valid:
            raise ValueError(f"representation must be one of {valid}, got {representation!r}")

    def __len__(self) -> int:
        return len(self.base)

    @property
    def in_channels(self) -> int:
        """Channel count the model should be built with."""
        if self.representation == "voxel_grid":
            return self.num_bins
        return 2  # histogram, time_surface and snn_spikes are polarity-split

    def _convert(self, events: np.ndarray) -> torch.Tensor:
        # tonic returns a structured array; pull the named fields out.
        x = events["x"].astype(np.int64)
        y = events["y"].astype(np.int64)
        t = events["t"].astype(np.int64)
        p = events["p"].astype(np.int8)

        # Guard against out-of-range coordinates. Some datasets contain a
        # handful of malformed events, and an index error mid-epoch on the
        # training box is an annoying way to lose an hour.
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]

        if self.representation == "voxel_grid":
            return events_to_voxel_grid(x, y, t, p, self.height, self.width, self.num_bins)
        if self.representation == "histogram":
            return events_to_histogram(x, y, p, self.height, self.width)
        if self.representation == "time_surface":
            return events_to_time_surface(x, y, t, p, self.height, self.width, self.tau)
        return events_to_snn_input(x, y, t, p, self.height, self.width, self.num_steps)

    def __getitem__(self, idx: int):
        events, target = self.base[idx]
        tensor = self._convert(events)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, int(target)


class SubsetWrapper(Dataset):
    """A random `n`-sample subset, for fast smoke tests on a laptop.

    THE SUBSET MUST BE SHUFFLED. N-MNIST (and most event datasets: N-CARS,
    N-Caltech101, GEN1) is stored SORTED BY CLASS. Taking the first n samples
    therefore yields a single-class dataset, on which a model trivially scores
    100% by always predicting that class.

    This is not hypothetical -- the first version of this class took `base[:n]`
    and produced exactly that: 100% train and validation accuracy on N-MNIST
    from a 1,730-parameter network, which looks like success and means nothing.

    A fixed seed keeps the subset reproducible, so shuffling costs nothing.
    """

    def __init__(self, base: Dataset, n: int, seed: int = 42):
        self.base = base
        self.n = min(n, len(base))
        generator = np.random.default_rng(seed)
        self.indices = generator.permutation(len(base))[: self.n]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.base[int(self.indices[idx])]


class NCarsDataset(Dataset):
    """N-CARS (Sironi et al., CVPR 2018) -- real automotive event recordings.

    24,029 samples of 100 ms each, recorded with an ATIS camera mounted behind
    a car windshield in urban driving. Binary: is there a car in this crop?

        12,336 car / 11,693 background
        train: 7,940 car + 7,482 background = 15,422
        test:  4,396 car + 4,211 background =  8,607

    On-disk layout, as shipped by Prophesee:

        <root>/Prophesee_Dataset_n_cars/
            n-cars_train/{cars,background}/obj_*_td.dat
            n-cars_test/{cars,background}/obj_*_td.dat

    Unlike N-MNIST the crops are VARIABLE SIZE -- each sample is a bounding-box
    crop around the object, so widths and heights differ per file. We pad every
    sample into a fixed canvas (`sensor_size`) so a batch can be stacked;
    `max_crop` is the canvas, not the sensor. Events outside it are dropped by
    EventTensorDataset's bounds guard.

    Label convention: 1 = car, 0 = background.
    """

    CLASSES = {"background": 0, "cars": 1}

    def __init__(self, root: str | Path, train: bool = True):
        root = Path(root)
        # Tolerate the archive being extracted with or without its top folder.
        for candidate in (root / "Prophesee_Dataset_n_cars", root):
            split_dir = candidate / ("n-cars_train" if train else "n-cars_test")
            if split_dir.is_dir():
                break
        else:
            raise FileNotFoundError(
                f"could not find n-cars_{'train' if train else 'test'} under {root}. "
                "Expected <root>/Prophesee_Dataset_n_cars/n-cars_train/{cars,background}/"
            )

        self.samples: list[tuple[Path, int]] = []
        for class_name, label in sorted(self.CLASSES.items()):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"missing class directory {class_dir}")
            for path in sorted(class_dir.glob("*.dat")):
                self.samples.append((path, label))

        if not self.samples:
            raise FileNotFoundError(f"no .dat files found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from src.data.prophesee import read_dat

        path, label = self.samples[idx]
        return read_dat(path), label


def load_ncars(
    root: str,
    representation: str = "voxel_grid",
    num_bins: int = 5,
    num_steps: int = 10,
    max_crop: tuple[int, int] = (120, 100),
    subset: int | None = None,
) -> tuple[EventTensorDataset, EventTensorDataset, tuple[int, int]]:
    """N-CARS train/test as model-ready tensors.

    `max_crop` is the fixed canvas every variable-size crop is placed into.
    (120, 100) covers the overwhelming majority of N-CARS boxes; anything
    larger is clipped by the bounds guard in EventTensorDataset.

    Returns (train, test, canvas_size).
    """
    train_base: Dataset = NCarsDataset(root, train=True)
    test_base: Dataset = NCarsDataset(root, train=False)

    if subset:
        train_base = SubsetWrapper(train_base, subset)
        test_base = SubsetWrapper(test_base, max(subset // 4, 1))

    kwargs = dict(
        sensor_size=max_crop,
        representation=representation,
        num_bins=num_bins,
        num_steps=num_steps,
    )
    return (
        EventTensorDataset(train_base, **kwargs),
        EventTensorDataset(test_base, **kwargs),
        max_crop,
    )


def load_nmnist(
    root: str,
    representation: str = "voxel_grid",
    num_bins: int = 5,
    num_steps: int = 10,
    subset: int | None = None,
) -> tuple[EventTensorDataset, EventTensorDataset, tuple[int, int]]:
    """N-MNIST: MNIST digits recorded by moving an event camera over a screen.

    A toy dataset -- 34x34, 10 classes -- but genuine event data with real
    temporal structure, which is exactly what Rung 1 needs. Downloads on first
    call (~1 GB).

    Returns (train, test, sensor_size).
    """
    import tonic  # imported lazily so the rest of the package works without it

    from src.data.download import ensure_nmnist

    # tonic's own downloader is 403'd by Mendeley (see src/data/download.py).
    # Fetch the archives ourselves first; tonic then finds them and extracts.
    ensure_nmnist(root)

    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
    wh = (sensor_size[0], sensor_size[1])

    train_base = tonic.datasets.NMNIST(save_to=root, train=True)
    test_base = tonic.datasets.NMNIST(save_to=root, train=False)

    if subset:
        train_base = SubsetWrapper(train_base, subset)
        test_base = SubsetWrapper(test_base, max(subset // 4, 1))

    kwargs = dict(
        sensor_size=wh,
        representation=representation,
        num_bins=num_bins,
        num_steps=num_steps,
    )
    return (
        EventTensorDataset(train_base, **kwargs),
        EventTensorDataset(test_base, **kwargs),
        wh,
    )
