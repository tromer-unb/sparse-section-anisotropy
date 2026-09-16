"""Input helpers for binary 2D sections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_section(path: str | Path) -> np.ndarray:
    """Load a 2D NPY or common image file without guessing the pore phase."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("L"))

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2D image, found shape {arr.shape}")
    return arr


def pore_indicator(array: np.ndarray, pore_value: float | int | bool = 0) -> np.ndarray:
    """Convert an image to the internal convention 1=pore, 0=solid."""
    arr = np.asarray(array)
    if arr.dtype == bool:
        return arr if bool(pore_value) else ~arr
    return arr == pore_value
