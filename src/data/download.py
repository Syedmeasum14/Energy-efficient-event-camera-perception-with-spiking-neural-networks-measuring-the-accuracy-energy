"""Robust dataset downloading.

WHY THIS EXISTS
---------------
`tonic` downloads via `urllib` with its default User-Agent. Several dataset
hosts -- Mendeley in particular, which serves N-MNIST -- reject that agent with
HTTP 403. The URLs themselves are fine: fetched with a browser-like User-Agent
and redirect following, they return 200.

This module fetches the archive first, into exactly the path tonic expects.
tonic then finds a valid file, skips its own (broken) download, and extracts
normally. No monkey-patching, no forked dependency.

Dataset hosting rots constantly -- Mendeley changed their API, and the host
serving POKERDVS no longer resolves at all. Keeping the fetch logic here, with
checksums, means a broken URL is a one-line fix rather than an afternoon.
"""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

# A browser-like User-Agent. Mendeley and several S3-backed hosts 403 the
# default Python one.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# url, filename, md5 -- mirroring tonic's own constants so a mismatch is loud.
NMNIST_FILES = {
    "train": (
        "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
        "39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded",
        "train.zip",
        "20959b8e626244a1b502305a9e6e2031",
    ),
    "test": (
        "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
        "05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded",
        "test.zip",
        "69ca8762b2fe404d9b9bad1103e97832",
    ),
}


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    url: str,
    dest: Path,
    expected_md5: str | None = None,
    max_retries: int = 5,
    retry_delay: float = 3.0,
) -> Path:
    """Download `url` to `dest`, resuming and retrying on failure.

    These archives are ~1 GB from academic hosts that drop connections. A naive
    download that dies at 98% and restarts from zero is not a hypothetical --
    it happened while building this. So:

      * partial data goes to a `.part` file that survives a failed attempt
      * retries resume with an HTTP Range header rather than starting over
      * the checksum is verified BEFORE the final rename, so a truncated file
        is never mistaken for a complete one

    Verifying before skipping matters just as much: a half-finished download
    left by an interrupted run would otherwise be treated as valid, and the
    failure surfaces much later as an unintelligible parse error.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_md5 is None or md5sum(dest) == expected_md5:
            print(f"  {dest.name}: already present, skipping")
            return dest
        print(f"  {dest.name}: checksum mismatch, re-downloading")
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        # Resume from whatever survived the previous attempt.
        offset = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            print(f"  {dest.name}: resuming from {offset / 1e6:.0f} MB (attempt {attempt})")
        else:
            print(f"  {dest.name}: downloading (attempt {attempt})")

        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request) as response:
                # 206 == server honoured the Range request. Anything else means
                # it is sending the whole file, so start the .part over rather
                # than appending and silently corrupting it.
                mode = "ab" if response.status == 206 and offset else "wb"
                with open(tmp, mode) as out:
                    shutil.copyfileobj(response, out)
            break
        except Exception as exc:  # noqa: BLE001 - any transport error is retryable
            if attempt == max_retries:
                raise RuntimeError(
                    f"failed to download {dest.name} after {max_retries} attempts: {exc}"
                ) from exc
            print(f"  {dest.name}: {type(exc).__name__}: {exc} -- retrying")
            time.sleep(retry_delay)

    if expected_md5 is not None:
        actual = md5sum(tmp)
        if actual != expected_md5:
            tmp.unlink()
            raise RuntimeError(
                f"checksum mismatch for {dest.name}: got {actual}, expected {expected_md5}"
            )

    # Rename only after verification, so an interrupted run never leaves a file
    # that looks complete.
    tmp.rename(dest)
    print(f"  {dest.name}: done ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def ensure_nmnist(root: str | Path) -> Path:
    """Place N-MNIST archives where tonic expects them, then let tonic extract.

    tonic looks for `<root>/NMNIST/train.zip` and `<root>/NMNIST/test.zip`.
    """
    target = Path(root) / "NMNIST"
    print(f"ensuring N-MNIST in {target}")
    for split, (url, filename, md5) in NMNIST_FILES.items():
        fetch(url, target / filename, md5)
    return target


if __name__ == "__main__":
    import sys

    ensure_nmnist(sys.argv[1] if len(sys.argv) > 1 else "data/nmnist")
