"""Tests for the Prophesee .dat reader.

Round-trips through synthetic files, so these run without the dataset. A
bit-packing error here would silently scramble coordinates and produce a
model that trains on noise -- which looks like "the task is hard", not "the
reader is broken".
"""

import numpy as np
import pytest

from src.data.prophesee import (
    STRUCTURED_DTYPE,
    pack_events,
    read_dat,
    unpack_events,
    write_dat,
)


def make_events(n: int = 200, seed: int = 0, width: int = 120, height: int = 100):
    rng = np.random.default_rng(seed)
    ev = np.empty(n, dtype=STRUCTURED_DTYPE)
    ev["x"] = rng.integers(0, width, n)
    ev["y"] = rng.integers(0, height, n)
    ev["t"] = np.sort(rng.integers(0, 100_000, n))
    ev["p"] = rng.integers(0, 2, n)
    return ev


def test_pack_unpack_round_trip():
    ev = make_events()
    back = unpack_events(pack_events(ev))
    for field in ("x", "y", "t", "p"):
        assert np.array_equal(back[field], ev[field]), f"{field} corrupted"


def test_file_round_trip(tmp_path):
    ev = make_events(500)
    path = tmp_path / "sample.dat"
    write_dat(path, ev)
    back = read_dat(path)
    assert len(back) == len(ev)
    for field in ("x", "y", "t", "p"):
        assert np.array_equal(back[field], ev[field])


def test_bit_layout_is_exact():
    """Hand-checked packing. x=1, y=2, p=1 must land in the documented bits."""
    ev = np.array([(1, 2, 12345, 1)], dtype=STRUCTURED_DTYPE)
    raw = pack_events(ev)
    expected = 1 | (2 << 14) | (1 << 28)
    assert raw["data"][0] == expected
    assert raw["t"][0] == 12345


def test_maximum_coordinates_survive():
    """14 bits each: coordinates up to 16383 must not overflow into each other.
    Real sensors are far smaller, but a mask error shows up here first."""
    ev = np.array([(16383, 16383, 7, 1)], dtype=STRUCTURED_DTYPE)
    back = unpack_events(pack_events(ev))
    assert back["x"][0] == 16383
    assert back["y"][0] == 16383
    assert back["p"][0] == 1


def test_polarity_stays_binary():
    ev = make_events(300)
    back = unpack_events(pack_events(ev))
    assert set(np.unique(back["p"])).issubset({0, 1})


def test_header_is_skipped(tmp_path):
    """Multiple header lines must not be read as event data."""
    ev = make_events(50)
    path = tmp_path / "hdr.dat"
    with open(path, "wb") as f:
        f.write(b"% Data file\n% Version 2\n% Width 120\n% Height 100\n")
        f.write(bytes([0, 8]))
        pack_events(ev).tofile(f)
    back = read_dat(path)
    assert len(back) == 50
    assert np.array_equal(back["x"], ev["x"])


def test_empty_file_returns_empty_array(tmp_path):
    path = tmp_path / "empty.dat"
    with open(path, "wb") as f:
        f.write(b"% nothing here\n")
    back = read_dat(path)
    assert len(back) == 0
    assert back.dtype == STRUCTURED_DTYPE


def test_unsupported_event_size_is_rejected(tmp_path):
    """Better a clear error than silently misreading every event."""
    path = tmp_path / "bad.dat"
    with open(path, "wb") as f:
        f.write(b"% header\n")
        f.write(bytes([0, 16]))  # ev_size 16, not supported
        f.write(b"\x00" * 64)
    with pytest.raises(ValueError, match="unsupported ev_size"):
        read_dat(path)


def test_timestamps_are_monotonic_after_read(tmp_path):
    ev = make_events(400)
    path = tmp_path / "t.dat"
    write_dat(path, ev)
    back = read_dat(path)
    assert np.all(np.diff(back["t"]) >= 0)


def test_output_is_compatible_with_event_tensor_dataset(tmp_path):
    """The reader's output must drop straight into the existing pipeline."""
    from src.data.datasets import EventTensorDataset

    ev = make_events(600, width=120, height=100)
    path = tmp_path / "s.dat"
    write_dat(path, ev)

    class OneFile:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return read_dat(path), 1

    ds = EventTensorDataset(OneFile(), sensor_size=(120, 100),
                            representation="snn_spikes", num_steps=8)
    x, y = ds[0]
    assert x.shape == (8, 2, 100, 120)
    assert y == 1
