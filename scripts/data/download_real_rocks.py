#!/usr/bin/env python3
"""Download binary micro-CT ROIs used by the real-rock validation.

Dataset
-------
Ferreira et al. (2023), Scientific Data 10:368.
Figshare+ DOI: 10.25452/figshare.plus.21375565
Article ID: 21375565

The script queries the Figshare API dynamically, downloads the requested
binary ROI archive, extracts its single RAW member, verifies the expected
2500^3 uint8 byte count, and optionally samples planes to verify binary values.
It deliberately does not infer the pore/solid phase identity from value
frequency alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

ARTICLE_ID = 21375565
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
RAW_SHAPE = (2500, 2500, 2500)
RAW_DTYPE = np.uint8
EXPECTED_RAW_BYTES = int(np.prod(RAW_SHAPE)) * np.dtype(RAW_DTYPE).itemsize
DEFAULT_OUT = Path("data/raw")

ROCKS = {
    "edb1": {
        "label": "Edwards Brown (EdB-1)",
        "prefixes": ("edb-1", "edb_1", "edb1"),
        "folder": "edwards_brown",
    },
    "bentheimer": {
        "label": "Bentheimer (Kocurek 15A)",
        "prefixes": ("kocurek_15a", "kocurek-15a", "15a"),
        "folder": "bentheimer",
    },
}


def fetch_metadata() -> dict:
    req = Request(API_URL, headers={"User-Agent": "sparse-section-anisotropy/0.1"})
    try:
        with urlopen(req, timeout=60) as response:
            data = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not query Figshare API: {exc}") from exc
    if not data.get("files"):
        raise RuntimeError("Figshare API returned no files.")
    return data


def normalized(name: str) -> str:
    return name.lower().replace(" ", "")


def find_roi(files: list[dict], rock: str, roi: int) -> dict:
    info = ROCKS[rock]
    matches = []
    for item in files:
        name = normalized(item.get("name", ""))
        prefix_ok = any(normalized(p) in name for p in info["prefixes"])
        roi_ok = "binary" in name and f"roi-{roi}" in name
        archive_ok = name.endswith(".raw.tar.bz2")
        if prefix_ok and roi_ok and archive_ok:
            matches.append(item)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one binary ROI-{roi} archive for {info['label']}; "
            f"found {len(matches)}: {[m.get('name') for m in matches]}"
        )
    return matches[0]


def download(item: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item.get("size", 0)) or None
    if expected and destination.exists() and destination.stat().st_size == expected:
        print(f"reuse: {destination}")
        return
    url = item.get("download_url") or f"https://ndownloader.figshare.com/files/{item['id']}"
    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [curl, "-L", "--fail", "--retry", "8", "-C", "-", "-o", str(destination), url],
            check=True,
        )
    else:
        req = Request(url, headers={"User-Agent": "sparse-section-anisotropy/0.1"})
        with urlopen(req, timeout=120) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if expected and destination.stat().st_size != expected:
        raise RuntimeError(f"Compressed download size mismatch for {destination.name}.")


def extract_raw(archive: Path, raw_path: Path) -> None:
    if raw_path.exists() and raw_path.stat().st_size == EXPECTED_RAW_BYTES:
        print(f"reuse: {raw_path}")
        return
    listed = subprocess.run(
        ["tar", "-tjf", str(archive)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    members = [m for m in listed if m.lower().endswith(".raw")]
    if len(members) != 1:
        raise RuntimeError(f"Expected one RAW member; found {members}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = raw_path.with_suffix(raw_path.suffix + ".part")
    with open(tmp, "wb") as output:
        subprocess.run(["tar", "-xOjf", str(archive), members[0]], stdout=output, check=True)
    if tmp.stat().st_size != EXPECTED_RAW_BYTES:
        raise RuntimeError(
            f"RAW byte count mismatch: expected {EXPECTED_RAW_BYTES}, got {tmp.stat().st_size}."
        )
    tmp.replace(raw_path)


def validate_binary(raw_path: Path) -> None:
    volume = np.memmap(raw_path, dtype=RAW_DTYPE, mode="r", shape=RAW_SHAPE, order="C")
    indices = np.linspace(125, RAW_SHAPE[0] - 126, 9, dtype=int)
    values: set[int] = set()
    for index in indices:
        values.update(int(v) for v in np.unique(np.asarray(volume[index])))
    del volume
    print(f"sampled stored values: {sorted(values)}")
    if not values.issubset({0, 1}):
        raise RuntimeError("Sampled RAW values are not binary 0/1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rock", choices=("bentheimer", "edb1", "both"), default="both")
    parser.add_argument("--rois", nargs="+", type=int, choices=(1, 2, 3), default=[1])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--delete-archive", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = fetch_metadata()
    print(f"dataset: {metadata.get('title', '(unknown)')}")
    print(f"DOI: 10.25452/figshare.plus.21375565")
    keys = ("bentheimer", "edb1") if args.rock == "both" else (args.rock,)
    for rock in keys:
        for roi in sorted(set(args.rois)):
            item = find_roi(metadata["files"], rock, roi)
            print(f"{ROCKS[rock]['label']} ROI-{roi}: {item['name']} ({item.get('size')} bytes)")
            if args.list_only:
                continue
            folder = args.out / ROCKS[rock]["folder"]
            archive = folder / item["name"]
            raw_name = item["name"][:-8] if item["name"].lower().endswith(".tar.bz2") else item["name"]
            raw_path = folder / raw_name
            download(item, archive)
            extract_raw(archive, raw_path)
            if not args.no_validate:
                validate_binary(raw_path)
            if args.delete_archive:
                archive.unlink(missing_ok=True)
            print(f"ready: {raw_path}")


if __name__ == "__main__":
    main()
