"""Tests for the training engine.

These use a tiny synthetic dataset so the whole file runs in seconds on a
laptop. The point is not to reach good accuracy -- it is to prove the loop is
wired correctly, so that when Rung 2 fails to converge you know the fault is in
the model or the data, not the plumbing.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.engine.train import (
    fit,
    measure_energy,
    pick_device,
    run_epoch,
)
from src.models.classifier import CNNClassifier, SNNClassifier

C, H, W, T = 2, 16, 16, 4
NUM_CLASSES = 2


def _separable_data(n: int = 64, snn: bool = False):
    """A trivially separable task: class 1 has many more events than class 0.

    Deliberately easy. If a model cannot learn this, the loop is broken.
    """
    y = torch.randint(0, NUM_CLASSES, (n,))
    density = torch.where(y == 1, 0.30, 0.02).view(n, 1, 1, 1)
    if snn:
        x = (torch.rand(n, T, C, H, W) < density.unsqueeze(1)).float()
    else:
        x = (torch.rand(n, C, H, W) < density).float()
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_pick_device_returns_a_device():
    assert isinstance(pick_device(), torch.device)
    assert pick_device("cpu").type == "cpu"


def test_eval_mode_does_not_change_weights():
    """No optimizer means no updates -- guards against an accidental step()."""
    model = CNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    before = [p.clone() for p in model.parameters()]
    run_epoch(model, _separable_data(snn=False), torch.device("cpu"))
    for p, b in zip(model.parameters(), before):
        assert torch.equal(p, b)


def test_train_mode_updates_weights():
    model = CNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = [p.clone() for p in model.parameters()]
    run_epoch(model, _separable_data(snn=False), torch.device("cpu"), opt)
    assert any(not torch.equal(p, b) for p, b in zip(model.parameters(), before))


def test_cnn_learns_separable_task():
    torch.manual_seed(0)
    model = CNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    loader = _separable_data(n=128, snn=False)
    result = fit(model, loader, loader, epochs=6, lr=5e-3, tag="test-cnn")
    assert result["best_accuracy"] > 0.85


def test_snn_learns_separable_task():
    """The end-to-end proof that surrogate gradients work through the whole
    stack: conv, BatchNorm, LIF, backprop-through-time, optimiser."""
    torch.manual_seed(0)
    model = SNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    loader = _separable_data(n=128, snn=True)
    result = fit(model, loader, loader, epochs=6, lr=5e-3, tag="test-snn")
    assert result["best_accuracy"] > 0.85


def test_snn_epoch_reports_firing_rate():
    model = SNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    res = run_epoch(model, _separable_data(snn=True), torch.device("cpu"))
    assert 0.0 < res.firing_rate < 1.0
    assert len(res.per_layer_rates) == 2


def test_cnn_epoch_reports_no_firing_rate():
    model = CNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    res = run_epoch(model, _separable_data(snn=False), torch.device("cpu"))
    assert res.firing_rate == 0.0
    assert res.per_layer_rates == {}


def test_sparsity_penalty_increases_reported_loss():
    """Sanity check that the regularisation term is actually applied."""
    torch.manual_seed(0)
    loader = _separable_data(snn=True)

    torch.manual_seed(0)
    plain = SNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    loss_plain = run_epoch(plain, loader, torch.device("cpu"), sparsity_lambda=0.0).loss

    torch.manual_seed(0)
    penalised = SNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    loss_pen = run_epoch(
        penalised, loader, torch.device("cpu"), sparsity_lambda=10.0
    ).loss

    assert loss_pen > loss_plain


def test_measure_energy_is_per_sample_and_normalised():
    """Energy must not scale with how many batches were measured."""
    model = SNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    loader = _separable_data(n=128, snn=True)
    few = measure_energy(model, loader, torch.device("cpu"), num_batches=2)
    many = measure_energy(model, loader, torch.device("cpu"), num_batches=8)
    assert few.total_synops > 0
    # Different batches have different sparsity, so allow slack -- but the two
    # must be the same order of magnitude, not 4x apart.
    assert 0.5 < many.total_synops / few.total_synops < 2.0


def test_measure_energy_picks_mode_from_model():
    cnn = CNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    snn = SNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    assert measure_energy(cnn, _separable_data(snn=False), torch.device("cpu")).mode == "dense"
    assert measure_energy(snn, _separable_data(snn=True), torch.device("cpu")).mode == "spiking"


def test_checkpoint_is_written(tmp_path):
    model = CNNClassifier(in_channels=C, num_classes=NUM_CLASSES, num_blocks=2)
    loader = _separable_data(snn=False)
    fit(model, loader, loader, epochs=2, checkpoint_dir=tmp_path, tag="ckpt")
    saved = tmp_path / "ckpt_best.pt"
    assert saved.exists()
    blob = torch.load(saved, weights_only=False)
    assert "model" in blob and "val_accuracy" in blob
