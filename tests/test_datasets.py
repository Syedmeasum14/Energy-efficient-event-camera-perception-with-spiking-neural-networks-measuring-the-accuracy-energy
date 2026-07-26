"""Tests for the event dataset wrapper.

Uses a fake tonic-shaped dataset so these run offline in milliseconds. The real
download is exercised by scripts/run_nmnist.py --subset.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.datasets import EventTensorDataset, SubsetWrapper

W, H = 34, 34


class FakeEventDataset:
    """Mimics tonic: yields (structured event array, label)."""

    def __init__(self, n: int = 20, n_events: int = 300, seed: int = 0):
        self.n = n
        self.n_events = n_events
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        dtype = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)])
        events = np.empty(self.n_events, dtype=dtype)
        events["x"] = self.rng.integers(0, W, self.n_events)
        events["y"] = self.rng.integers(0, H, self.n_events)
        events["t"] = np.sort(self.rng.integers(0, 100_000, self.n_events))
        events["p"] = self.rng.integers(0, 2, self.n_events)
        return events, idx % 10


@pytest.mark.parametrize(
    "representation,expected_shape",
    [
        ("voxel_grid", (5, H, W)),
        ("histogram", (2, H, W)),
        ("time_surface", (2, H, W)),
        ("snn_spikes", (10, 2, H, W)),
    ],
)
def test_representation_shapes(representation, expected_shape):
    ds = EventTensorDataset(
        FakeEventDataset(), sensor_size=(W, H), representation=representation
    )
    tensor, label = ds[0]
    assert tensor.shape == expected_shape
    assert isinstance(label, int)


def test_in_channels_matches_tensor():
    """The model is built from in_channels; a mismatch is a runtime crash."""
    for rep, bins in [("voxel_grid", 7), ("histogram", 5), ("time_surface", 5)]:
        ds = EventTensorDataset(
            FakeEventDataset(), sensor_size=(W, H), representation=rep, num_bins=bins
        )
        assert ds[0][0].shape[0] == ds.in_channels


def test_snn_representation_is_binary():
    ds = EventTensorDataset(
        FakeEventDataset(), sensor_size=(W, H), representation="snn_spikes"
    )
    x, _ = ds[0]
    assert torch.all((x == 0) | (x == 1))


def test_out_of_range_events_are_dropped():
    """A few malformed events should not crash an epoch mid-training."""

    class Corrupt(FakeEventDataset):
        def __getitem__(self, idx):
            events, label = super().__getitem__(idx)
            events["x"][0] = 9999   # way out of bounds
            events["y"][1] = -5
            return events, label

    ds = EventTensorDataset(Corrupt(), sensor_size=(W, H), representation="voxel_grid")
    assert ds[0][0].shape == (5, H, W)  # no IndexError


def test_invalid_representation_rejected():
    with pytest.raises(ValueError, match="representation must be"):
        EventTensorDataset(FakeEventDataset(), sensor_size=(W, H), representation="fourier")


def test_subset_wrapper_truncates():
    base = FakeEventDataset(n=50)
    assert len(SubsetWrapper(base, 10)) == 10
    assert len(SubsetWrapper(base, 999)) == 50  # clamps, does not error


class SortedByClassDataset(FakeEventDataset):
    """Mimics N-MNIST's real on-disk layout: all class 0, then all class 1, ..."""

    def __getitem__(self, idx):
        events, _ = super().__getitem__(idx)
        return events, idx // (self.n // 10)


def test_subset_is_shuffled_not_sequential():
    """REGRESSION: N-MNIST is stored sorted by class, so an unshuffled subset
    contains a single class and every model scores 100% for free."""
    base = SortedByClassDataset(n=1000)
    subset = SubsetWrapper(base, 100)
    labels = {subset[i][1] for i in range(len(subset))}
    assert len(labels) > 5, f"subset covers only {len(labels)} classes -- not shuffled"


def test_subset_is_reproducible():
    """Shuffled, but deterministic: same seed must give the same subset."""
    base = SortedByClassDataset(n=1000)
    a = SubsetWrapper(base, 50, seed=7)
    b = SubsetWrapper(base, 50, seed=7)
    c = SubsetWrapper(base, 50, seed=8)
    assert list(a.indices) == list(b.indices)
    assert list(a.indices) != list(c.indices)


def test_subset_indices_are_unique():
    """A permutation must not repeat samples."""
    base = FakeEventDataset(n=200)
    subset = SubsetWrapper(base, 100)
    assert len(set(subset.indices.tolist())) == 100


def test_dataloader_collates_correctly():
    """Batch shapes must match what the models expect: (B,C,H,W) for the CNN,
    (B,T,C,H,W) for the SNN -- the train loop permutes the latter."""
    cnn_ds = EventTensorDataset(
        FakeEventDataset(n=8), sensor_size=(W, H), representation="voxel_grid"
    )
    x, y = next(iter(DataLoader(cnn_ds, batch_size=4)))
    assert x.shape == (4, 5, H, W) and y.shape == (4,)

    snn_ds = EventTensorDataset(
        FakeEventDataset(n=8), sensor_size=(W, H), representation="snn_spikes", num_steps=6
    )
    x, y = next(iter(DataLoader(snn_ds, batch_size=4)))
    assert x.shape == (4, 6, 2, H, W)


def test_end_to_end_batch_feeds_both_models():
    """Integration: real dataloader output must run through both models."""
    from src.models.classifier import CNNClassifier, SNNClassifier

    cnn_ds = EventTensorDataset(
        FakeEventDataset(n=8), sensor_size=(W, H), representation="voxel_grid"
    )
    x, _ = next(iter(DataLoader(cnn_ds, batch_size=4)))
    cnn = CNNClassifier(in_channels=cnn_ds.in_channels, num_classes=10)
    assert cnn(x).shape == (4, 10)

    snn_ds = EventTensorDataset(
        FakeEventDataset(n=8), sensor_size=(W, H), representation="snn_spikes", num_steps=6
    )
    x, _ = next(iter(DataLoader(snn_ds, batch_size=4)))
    snn = SNNClassifier(in_channels=snn_ds.in_channels, num_classes=10)
    # Dataloader gives (B,T,C,H,W); the model wants time first.
    assert snn(x.permute(1, 0, 2, 3, 4)).shape == (4, 10)
