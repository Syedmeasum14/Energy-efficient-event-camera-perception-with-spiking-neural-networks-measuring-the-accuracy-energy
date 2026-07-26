"""Reader for Prophesee `.dat` event files.

N-CARS, GEN1 and the 1 Mpx detection dataset all ship in this format, and
`tonic` does not support it -- so Rung 2 and Rung 3 both depend on this file.

THE FORMAT
----------
1. ASCII header lines, each starting with `%`, terminated by the first line
   that does not. Contains metadata such as the sensor geometry.
2. Two bytes: `ev_type` (uint8) and `ev_size` (uint8). `ev_size` is 8 for
   every dataset we use.
3. A flat array of 8-byte events, little-endian:

       bytes 0-3   uint32   timestamp, microseconds
       bytes 4-7   uint32   packed payload

   The payload packs three fields:

       bits  0-13   x        (14 bits)
       bits 14-27   y        (14 bits)
       bit     28   polarity (1 bit)

   So `x = data & 0x3FFF`, `y = (data >> 14) & 0x3FFF`, `p = (data >> 28) & 1`.

Everything is read with a single `np.fromfile` and unpacked with vectorised
bit operations -- a Python loop over a few million events is unusably slow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Bit layout of the 32-bit payload word.
X_MASK = 0x3FFF
Y_SHIFT = 14
Y_MASK = 0x3FFF
P_SHIFT = 28
P_MASK = 0x1

EVENT_DTYPE = np.dtype([("t", "<u4"), ("data", "<u4")])

# What the rest of the codebase expects: the same field names tonic uses, so
# EventTensorDataset consumes both without a special case.
STRUCTURED_DTYPE = np.dtype(
    [("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)]
)


def read_dat_header(f) -> dict:
    """Consume the `%`-prefixed header, returning parsed key/value pairs.

    Leaves the file positioned at the two type bytes that follow.
    """
    meta: dict[str, str] = {}
    while True:
        pos = f.tell()
        line = f.readline()
        if not line:
            break
        if not line.startswith(b"%"):
            # Overshot into the binary section -- rewind and stop.
            f.seek(pos)
            break
        text = line[1:].strip().decode("utf-8", errors="ignore")
        if " " in text:
            key, _, value = text.partition(" ")
            meta[key.strip()] = value.strip()
    return meta


def read_dat(path: str | Path) -> np.ndarray:
    """Read a Prophesee `.dat` file into a structured array.

    Returns a (N,) array with fields x, y, t, p -- p in {0, 1}, t in
    microseconds, sorted by time as stored.
    """
    path = Path(path)
    with open(path, "rb") as f:
        read_dat_header(f)

        type_bytes = f.read(2)
        if len(type_bytes) < 2:
            return np.empty(0, dtype=STRUCTURED_DTYPE)
        ev_size = type_bytes[1]
        if ev_size != 8:
            raise ValueError(
                f"{path}: unsupported ev_size {ev_size} (expected 8). "
                "This reader handles the standard 8-byte CD event layout only."
            )

        raw = np.fromfile(f, dtype=EVENT_DTYPE)

    return unpack_events(raw)


def unpack_events(raw: np.ndarray) -> np.ndarray:
    """Unpack the packed (t, data) pairs into x/y/t/p fields."""
    out = np.empty(raw.shape[0], dtype=STRUCTURED_DTYPE)
    data = raw["data"]
    out["x"] = (data & X_MASK).astype(np.int16)
    out["y"] = ((data >> Y_SHIFT) & Y_MASK).astype(np.int16)
    out["t"] = raw["t"].astype(np.int64)
    out["p"] = ((data >> P_SHIFT) & P_MASK).astype(np.int8)
    return out


def pack_events(events: np.ndarray) -> np.ndarray:
    """Inverse of unpack_events. Used by the tests to build synthetic files."""
    raw = np.empty(events.shape[0], dtype=EVENT_DTYPE)
    raw["t"] = events["t"].astype("<u4")
    raw["data"] = (
        (events["x"].astype("<u4") & X_MASK)
        | ((events["y"].astype("<u4") & Y_MASK) << Y_SHIFT)
        | ((events["p"].astype("<u4") & P_MASK) << P_SHIFT)
    )
    return raw


def write_dat(path: str | Path, events: np.ndarray, header: str = "Test file") -> None:
    """Write a `.dat` file. Exists so tests can round-trip without the dataset."""
    with open(path, "wb") as f:
        f.write(f"% {header}\n".encode())
        f.write(b"% end\n")
        f.write(bytes([0, 8]))  # ev_type=0 (CD), ev_size=8
        pack_events(events).tofile(f)
