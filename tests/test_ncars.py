"""Tests for the N-CARS dataset.

Skipped when the dataset is absent, so the suite still runs on a clean clone.
Where the dataset IS present these check the counts against the published
paper -- a loader that silently drops or duplicates samples would poison every
number downstream.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.datasets import EventTensorDataset, NCarsDataset, load_ncars

ROOT = Path(__file__).resolve().parent.parent
NCARS = ROOT / "data" / "ncars"

pytestmark = pytest.mark.skipif(
    not (NCARS / "Prophesee_Dataset_n_cars" / "n-cars_test").is_dir(),
    reason="N-CARS not downloaded (see README)",
)

# Published in Sironi et al., CVPR 2018.
EXPECTED = {
    ("train", 1): 7940,   # cars
    ("train", 0): 7482,   # background
    ("test", 1): 4396,
    ("test", 0): 4211,
}
CANVAS = (120, 100)


def test_split_counts_match_the_paper():
    for split, train in (("train", True), ("test", False)):
        ds = NCarsDataset(NCARS, train=train)
        labels = [lbl for _, lbl in ds.samples]
        assert labels.count(1) == EXPECTED[(split, 1)], f"{split} car count wrong"
        assert labels.count(0) == EXPECTED[(split, 0)], f"{split} background count wrong"


def test_total_is_24029():
    total = len(NCarsDataset(NCARS, train=True)) + len(NCarsDataset(NCARS, train=False))
    assert total == 24029


def test_labels_are_binary_and_both_present():
    ds = NCarsDataset(NCARS, train=False)
    labels = {lbl for _, lbl in ds.samples}
    assert labels == {0, 1}


def test_reads_real_events():
    ds = NCarsDataset(NCARS, train=False)
    events, label = ds[0]
    assert len(events) > 0
    assert label in (0, 1)
    for field in ("x", "y", "t", "p"):
        assert field in events.dtype.names
    assert set(events["p"].tolist()).issubset({0, 1})


def test_samples_are_about_100ms():
    """The paper specifies 100 ms per sample. A gross mismatch would mean the
    timestamps are being misread."""
    ds = NCarsDataset(NCARS, train=False)
    for i in (0, 500, 2000):
        events, _ = ds[i]
        duration_ms = (int(events["t"].max()) - int(events["t"].min())) / 1000
        assert 50 < duration_ms <= 105, f"sample {i} spans {duration_ms:.1f} ms"


def test_canvas_covers_every_crop():
    """N-CARS crops are variable size. The canvas must not clip real events --
    verified against the measured maxima (width 120, height 100)."""
    ds = NCarsDataset(NCARS, train=True)
    step = max(len(ds) // 200, 1)
    for i in range(0, len(ds), step):
        events, _ = ds[i]
        if len(events) == 0:
            continue
        assert events["x"].max() < CANVAS[0], f"sample {i} exceeds canvas width"
        assert events["y"].max() < CANVAS[1], f"sample {i} exceeds canvas height"


@pytest.mark.parametrize("representation,expected", [
    ("voxel_grid", (5, 100, 120)),
    ("snn_spikes", (10, 2, 100, 120)),
])
def test_tensor_shapes(representation, expected):
    train, test, canvas = load_ncars(str(NCARS), representation=representation, subset=8)
    assert canvas == CANVAS
    x, y = train[0]
    assert x.shape == expected
    assert y in (0, 1)


def test_snn_tensor_is_binary():
    train, _, _ = load_ncars(str(NCARS), representation="snn_spikes", subset=8)
    x, _ = train[0]
    assert torch.all((x == 0) | (x == 1))


def test_subset_is_class_balanced():
    """N-CARS is stored sorted by class. An unshuffled subset would be one
    class only, and the model would score 100% for free."""
    train, _, _ = load_ncars(str(NCARS), representation="voxel_grid", subset=200)
    labels = [train[i][1] for i in range(0, len(train), 4)]
    assert 0 in labels and 1 in labels
    frac = sum(labels) / len(labels)
    assert 0.25 < frac < 0.75, f"subset is {frac:.0%} cars -- not balanced"


def test_dataloader_batches_for_both_models():
    train_cnn, _, _ = load_ncars(str(NCARS), representation="voxel_grid", subset=16)
    x, y = next(iter(DataLoader(train_cnn, batch_size=4)))
    assert x.shape == (4, 5, 100, 120) and y.shape == (4,)

    train_snn, _, _ = load_ncars(str(NCARS), representation="snn_spikes",
                                 num_steps=6, subset=16)
    x, _ = next(iter(DataLoader(train_snn, batch_size=4)))
    assert x.shape == (4, 6, 2, 100, 120)


def test_missing_root_raises_clearly():
    with pytest.raises(FileNotFoundError, match="could not find n-cars"):
        NCarsDataset(ROOT / "data" / "definitely-not-here", train=True)


def test_end_to_end_through_the_snn():
    """A real N-CARS batch must run through the actual model."""
    from src.models.classifier import SNNClassifier

    train, _, _ = load_ncars(str(NCARS), representation="snn_spikes",
                             num_steps=6, subset=8)
    x, _ = next(iter(DataLoader(train, batch_size=2)))
    model = SNNClassifier(in_channels=2, num_classes=2, num_blocks=3, num_steps=6)
    out = model(x.permute(1, 0, 2, 3, 4))
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_isinstance_of_event_tensor_dataset():
    train, _, _ = load_ncars(str(NCARS), subset=4)
    assert isinstance(train, EventTensorDataset)
