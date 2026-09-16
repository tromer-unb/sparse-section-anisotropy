#!/usr/bin/env python3
"""
Download selected binary micro-CT ROI cubes from the Figshare+ rock dataset:

  Full scale, microscopically resolved tomographies of sandstone and carbonate
  rocks augmented by experimental porosity and permeability values
  Figshare+ article ID: 21375565

Default action:
  - download Edwards Brown (EdB-1), binary ROI-1 only
  - download Dataset_Information.xlsx if not already present
  - extract the .raw member into a clean directory
  - verify the expected 2500^3 uint8 byte count
  - sample a few planes and report the 0/1 fractions (without assuming which
    phase is pore until metadata / morphology is checked)

The downloader queries the Figshare API dynamically, so file IDs do not need to
be hard-coded.

Examples
--------
# Recommended first external-validation download:
python3 download_real_rocks.py --rock edb1 --rois 1

# Download all three binary ROIs for Edwards Brown:
python3 download_real_rocks.py --rock edb1 --rois 1 2 3

# Only list matching files, no download:
python3 download_real_rocks.py --rock edb1 --rois 1 --list-only

# Bentheimer + Edwards Brown, ROI-1 only:
python3 download_real_rocks.py --rock both --rois 1

# Delete compressed archive after successful extraction:
python3 download_real_rocks.py --rock edb1 --rois 1 --delete-archive
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np

ARTICLE_ID = 21375565
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
DEFAULT_OUT = Path.home() / "Isarael" / "pushforward" / "data" / "real_rocks"
RAW_SHAPE = (2500, 2500, 2500)
RAW_DTYPE = np.uint8
EXPECTED_RAW_BYTES = int(np.prod(RAW_SHAPE)) * np.dtype(RAW_DTYPE).itemsize

ROCKS = {
    "edb1": {
        "label": "Edwards Brown (EdB-1)",
        "prefixes": ("edb-1", "edb_1", "edb1"),
        "folder": "EdwardsBrown_EdB1",
    },
    "bentheimer": {
        "label": "Bentheimer (Kocurek 15A)",
        "prefixes": ("kocurek_15a", "kocurek-15a", "15a"),
        "folder": "Bentheimer_15A",
    },
}


def fetch_article_metadata() -> dict:
    print("=" * 78)
    print("QUERYING FIGSHARE+ DATASET")
    print("=" * 78)
    print(f"Article ID: {ARTICLE_ID}")
    print(f"API:        {API_URL}")
    req = Request(API_URL, headers={"User-Agent": "python-download-real-rocks/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.load(resp)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not query Figshare API: {exc}") from exc

    files = data.get("files", [])
    if not files:
        raise RuntimeError("Figshare API returned no files for this article.")

    print(f"Title:      {data.get('title', '(unknown)')}")
    print(f"Version:    {data.get('version', '(unknown)')}")
    print(f"Files:      {len(files)}")
    print()
    return data


def norm(s: str) -> str:
    return s.lower().replace(" ", "")


def match_binary_roi(files: list[dict], rock_key: str, roi: int) -> dict:
    info = ROCKS[rock_key]
    suffix = f"_binary_roi-{roi}.raw.tar.bz2"
    candidates = []

    for f in files:
        name = f.get("name", "")
        n = norm(name)
        prefix_ok = any(norm(p) in n for p in info["prefixes"])
        suffix_ok = suffix in n
        if prefix_ok and suffix_ok:
            candidates.append(f)

    if len(candidates) == 1:
        return candidates[0]

    # More permissive fallback in case punctuation in the repository differs.
    if not candidates:
        for f in files:
            name = f.get("name", "")
            n = norm(name)
            prefix_ok = any(norm(p) in n for p in info["prefixes"])
            tokens_ok = (
                "binary" in n
                and f"roi-{roi}" in n
                and n.endswith(".raw.tar.bz2")
            )
            if prefix_ok and tokens_ok:
                candidates.append(f)

    if len(candidates) != 1:
        names = [f.get("name", "") for f in candidates]
        raise RuntimeError(
            f"Expected exactly one binary ROI-{roi} file for {info['label']}; "
            f"found {len(candidates)}. Candidates: {names}"
        )
    return candidates[0]


def match_dataset_information(files: list[dict]) -> dict | None:
    exact = [f for f in files if f.get("name", "").lower() == "dataset_information.xlsx"]
    if len(exact) == 1:
        return exact[0]
    fallback = [
        f for f in files
        if "dataset_information" in f.get("name", "").lower()
        and f.get("name", "").lower().endswith(".xlsx")
    ]
    return fallback[0] if fallback else None


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{n} B"


def download_with_resume(url: str, destination: Path, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if expected_size and destination.exists() and destination.stat().st_size == expected_size:
        print(f"  already downloaded: {destination.name} ({human_bytes(expected_size)})")
        return

    curl = shutil.which("curl")
    if curl:
        cmd = [
            curl,
            "-L",
            "--fail",
            "--retry", "8",
            "--retry-delay", "3",
            "--connect-timeout", "30",
            "-C", "-",
            "-o", str(destination),
            url,
        ]
        print("  downloading with resumable curl...")
        print("  " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("  curl not found; using Python urllib (no automatic resume).")
        req = Request(url, headers={"User-Agent": "python-download-real-rocks/1.0"})
        with urlopen(req, timeout=120) as r, open(destination, "wb") as out:
            shutil.copyfileobj(r, out, length=8 * 1024 * 1024)

    if expected_size is not None:
        actual = destination.stat().st_size
        if actual != expected_size:
            raise RuntimeError(
                f"Downloaded size mismatch for {destination.name}: "
                f"expected {expected_size}, got {actual}. Re-run the script to resume."
            )
    print(f"  download complete: {destination} ({human_bytes(destination.stat().st_size)})")


def tar_members(archive: Path) -> list[str]:
    tar = shutil.which("tar")
    if not tar:
        raise RuntimeError("System 'tar' command not found.")
    p = subprocess.run(
        [tar, "-tjf", str(archive)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def extract_raw_clean(archive: Path, out_raw: Path) -> Path:
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    if out_raw.exists() and out_raw.stat().st_size == EXPECTED_RAW_BYTES:
        print(f"  raw already extracted and size-valid: {out_raw}")
        return out_raw

    members = tar_members(archive)
    raw_members = [m for m in members if m.lower().endswith(".raw")]
    if len(raw_members) != 1:
        raise RuntimeError(
            f"Expected one .raw member inside {archive.name}; found {len(raw_members)}: {raw_members[:10]}"
        )

    member = raw_members[0]
    print(f"  archive member: {member}")
    print(f"  extracting cleanly to: {out_raw}")
    print(f"  expected uncompressed raw size: {human_bytes(EXPECTED_RAW_BYTES)}")

    tar = shutil.which("tar")
    tmp = out_raw.with_suffix(out_raw.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    with open(tmp, "wb") as fout:
        subprocess.run(
            [tar, "-xOjf", str(archive), member],
            stdout=fout,
            check=True,
        )

    actual = tmp.stat().st_size
    if actual != EXPECTED_RAW_BYTES:
        raise RuntimeError(
            f"Extracted RAW has unexpected size: expected {EXPECTED_RAW_BYTES:,}, got {actual:,}."
        )
    tmp.replace(out_raw)
    print("  extraction complete and byte-count verified.")
    return out_raw


def validate_binary_raw(raw_path: Path, n_planes: int = 9) -> None:
    print("  validating binary values from sampled planes...")
    vol = np.memmap(raw_path, dtype=RAW_DTYPE, mode="r", shape=RAW_SHAPE, order="C")
    idx = np.linspace(125, RAW_SHAPE[0] - 126, n_planes, dtype=int)

    counts: dict[int, int] = {}
    sampled = 0

    # Sample axis-0 planes only for speed and sequential-ish access. This is
    # enough to verify stored values; it is not an exact whole-volume porosity.
    for k in idx:
        plane = np.asarray(vol[k, :, :])
        vals, cnt = np.unique(plane, return_counts=True)
        sampled += plane.size
        for v, c in zip(vals.tolist(), cnt.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(c)

    del vol
    vals_sorted = sorted(counts)
    print(f"  sampled planes: {idx.tolist()}")
    print(f"  sampled voxels: {sampled:,}")
    print(f"  distinct values: {vals_sorted}")
    for v in vals_sorted:
        print(f"    value {v}: {counts[v]:,}  fraction={counts[v]/sampled:.6f}")

    if not set(vals_sorted).issubset({0, 1}):
        print("  WARNING: sampled data contain values other than 0/1.")
    else:
        print("  binary storage check: PASS")
        print("  NOTE: phase identity (pore vs solid) is intentionally NOT inferred here.")


def metadata_download(files: list[dict], root: Path, list_only: bool) -> None:
    f = match_dataset_information(files)
    if f is None:
        print("WARNING: Dataset_Information.xlsx not found through API.")
        return
    dest = root / "metadata" / f["name"]
    size = int(f.get("size", 0)) or None
    print("=" * 78)
    print("DATASET METADATA")
    print("=" * 78)
    print(f"File: {f['name']}")
    print(f"ID:   {f.get('id')}")
    print(f"Size: {human_bytes(size) if size else '(unknown)'}")
    print(f"Path: {dest}")
    if not list_only:
        download_with_resume(f.get("download_url") or f"https://ndownloader.figshare.com/files/{f['id']}", dest, size)
    print()


def process_target(files: list[dict], rock_key: str, roi: int, root: Path,
                   list_only: bool, no_extract: bool, delete_archive: bool,
                   validate: bool) -> None:
    info = ROCKS[rock_key]
    f = match_binary_roi(files, rock_key, roi)
    name = f["name"]
    size = int(f.get("size", 0)) or None
    file_id = f.get("id")
    url = f.get("download_url") or f"https://ndownloader.figshare.com/files/{file_id}"

    roi_dir = root / info["folder"] / f"ROI{roi}"
    archive = roi_dir / name

    # Clean raw name is based on archive filename, independent of internal tar path.
    clean_raw_name = name[:-8] if name.lower().endswith(".tar.bz2") else (name + ".raw")
    raw_path = roi_dir / clean_raw_name

    print("=" * 78)
    print(f"{info['label']} — BINARY ROI-{roi}")
    print("=" * 78)
    print(f"Repository file: {name}")
    print(f"File ID:         {file_id}")
    print(f"Compressed size: {human_bytes(size) if size else '(unknown)'}")
    print(f"Download URL:    {url}")
    print(f"Archive path:    {archive}")
    print(f"RAW path:        {raw_path}")
    print(f"RAW expected:    {EXPECTED_RAW_BYTES:,} bytes ({human_bytes(EXPECTED_RAW_BYTES)})")

    if list_only:
        print("LIST ONLY: no download performed.\n")
        return

    download_with_resume(url, archive, size)

    if not no_extract:
        extract_raw_clean(archive, raw_path)
        if validate:
            validate_binary_raw(raw_path)
        if delete_archive:
            print(f"  deleting compressed archive: {archive}")
            archive.unlink()
    else:
        print("  extraction skipped (--no-extract)")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download binary ROI cubes for real-rock validation from Figshare+ article 21375565."
    )
    p.add_argument(
        "--rock",
        choices=("edb1", "bentheimer", "both"),
        default="edb1",
        help="Rock to download. Default: edb1",
    )
    p.add_argument(
        "--rois",
        nargs="+",
        type=int,
        choices=(1, 2, 3),
        default=[1],
        help="ROI numbers. Default: 1",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output root. Default: {DEFAULT_OUT}",
    )
    p.add_argument("--list-only", action="store_true", help="Show matched repository files but do not download.")
    p.add_argument("--no-metadata", action="store_true", help="Do not download Dataset_Information.xlsx.")
    p.add_argument("--no-extract", action="store_true", help="Download archive but do not extract the RAW.")
    p.add_argument("--delete-archive", action="store_true", help="Delete .tar.bz2 after successful extraction.")
    p.add_argument("--no-validate", action="store_true", help="Skip sampled binary-value validation after extraction.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.out.expanduser().resolve()
    print(f"Output root: {root}")
    print(f"Free space before download: {human_bytes(shutil.disk_usage(root.parent if root.parent.exists() else Path.home()).free)}")
    print()

    data = fetch_article_metadata()
    files = data["files"]

    if not args.no_metadata:
        metadata_download(files, root, args.list_only)

    rock_keys = ["edb1", "bentheimer"] if args.rock == "both" else [args.rock]
    for rock_key in rock_keys:
        for roi in sorted(set(args.rois)):
            process_target(
                files=files,
                rock_key=rock_key,
                roi=roi,
                root=root,
                list_only=args.list_only,
                no_extract=args.no_extract,
                delete_archive=args.delete_archive,
                validate=not args.no_validate,
            )

    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Output root: {root}")
    if not args.list_only:
        try:
            print(f"Free space after download: {human_bytes(shutil.disk_usage(root).free)}")
        except FileNotFoundError:
            pass
    print()
    print("Recommended next step:")
    print("  verify EdB-1 phase identity against Dataset_Information.xlsx / morphology")
    print("  before reusing the Bentheimer final-validation script.")


if __name__ == "__main__":
    main()
