"""Tests for the download helper. No network access -- uses a local file:// URL."""

import pytest

from src.data.download import NMNIST_FILES, fetch, md5sum


def _make_source(tmp_path, content=b"hello events"):
    src = tmp_path / "source.bin"
    src.write_bytes(content)
    return src, md5sum(src)


def test_md5sum_matches_hashlib(tmp_path):
    import hashlib

    src, digest = _make_source(tmp_path)
    assert digest == hashlib.md5(src.read_bytes()).hexdigest()


def test_fetch_downloads_and_verifies(tmp_path):
    src, digest = _make_source(tmp_path)
    dest = tmp_path / "out" / "copy.bin"
    fetch(src.as_uri(), dest, digest)
    assert dest.exists()
    assert dest.read_bytes() == src.read_bytes()


def test_fetch_skips_when_already_valid(tmp_path):
    src, digest = _make_source(tmp_path)
    dest = tmp_path / "copy.bin"
    fetch(src.as_uri(), dest, digest)
    mtime = dest.stat().st_mtime_ns

    fetch(src.as_uri(), dest, digest)  # second call must be a no-op
    assert dest.stat().st_mtime_ns == mtime


def test_fetch_redownloads_on_checksum_mismatch(tmp_path):
    """A truncated file from an interrupted run must not be trusted."""
    src, digest = _make_source(tmp_path)
    dest = tmp_path / "copy.bin"
    dest.write_bytes(b"truncated")

    fetch(src.as_uri(), dest, digest)
    assert dest.read_bytes() == src.read_bytes()


def test_fetch_raises_on_bad_checksum(tmp_path):
    src, _ = _make_source(tmp_path)
    dest = tmp_path / "copy.bin"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        fetch(src.as_uri(), dest, "0" * 32)
    assert not dest.exists()


def test_no_partial_file_left_behind(tmp_path):
    """A failed verification must not leave a .part file masquerading as data."""
    src, _ = _make_source(tmp_path)
    dest = tmp_path / "copy.bin"
    with pytest.raises(RuntimeError):
        fetch(src.as_uri(), dest, "0" * 32)
    assert list(tmp_path.glob("*.part")) == []


def test_retries_then_succeeds(tmp_path, monkeypatch):
    """A transient transport error must not abort the download."""
    import src.data.download as dl

    src, digest = _make_source(tmp_path)
    dest = tmp_path / "copy.bin"

    real_urlopen = dl.urllib.request.urlopen
    calls = {"n": 0}

    def flaky(request, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError("simulated reset")
        return real_urlopen(request, *args, **kwargs)

    monkeypatch.setattr(dl.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(dl.time, "sleep", lambda _: None)

    fetch(src.as_uri(), dest, digest, retry_delay=0)
    assert calls["n"] == 2
    assert dest.read_bytes() == src.read_bytes()


def test_gives_up_after_max_retries(tmp_path, monkeypatch):
    import src.data.download as dl

    src, digest = _make_source(tmp_path)
    monkeypatch.setattr(
        dl.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionResetError("always fails")),
    )
    monkeypatch.setattr(dl.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        fetch(src.as_uri(), tmp_path / "copy.bin", digest, max_retries=3, retry_delay=0)


def test_partial_file_is_kept_between_attempts(tmp_path, monkeypatch):
    """The whole point of resume: a failed attempt must not discard progress."""
    import src.data.download as dl

    src, digest = _make_source(tmp_path, content=b"x" * 5000)
    dest = tmp_path / "copy.bin"
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(b"x" * 2000)  # simulate an interrupted earlier attempt

    seen_ranges = []
    real_urlopen = dl.urllib.request.urlopen

    def record(request, *args, **kwargs):
        seen_ranges.append(request.headers.get("Range"))
        return real_urlopen(request, *args, **kwargs)

    monkeypatch.setattr(dl.urllib.request, "urlopen", record)
    fetch(src.as_uri(), dest, digest, retry_delay=0)

    # A Range header must have been sent for the surviving 2000 bytes.
    assert seen_ranges[0] == "bytes=2000-"


def test_nmnist_constants_are_well_formed():
    """Guards against a typo in a URL or checksum -- which would otherwise
    surface as an unintelligible parse error much later."""
    assert set(NMNIST_FILES) == {"train", "test"}
    for url, filename, md5 in NMNIST_FILES.values():
        assert url.startswith("https://")
        assert filename.endswith(".zip")
        assert len(md5) == 32 and all(c in "0123456789abcdef" for c in md5)
