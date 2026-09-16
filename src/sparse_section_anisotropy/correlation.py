"""Directional two-point correlation utilities.

The implementation follows the finite-domain, overlap-normalized estimator used
in the frozen paper scripts. Binary input uses 1 for pore and 0 for solid.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve


def autocorrelation_field(binary: np.ndarray, valid: np.ndarray | None = None):
    """Return normalized finite-domain 2D autocorrelation and its zero-lag index."""
    x = np.asarray(binary, dtype=float)
    if x.ndim != 2:
        raise ValueError("binary must be a 2D array")

    if valid is None:
        m = np.ones_like(x, dtype=float)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != x.shape:
            raise ValueError("valid mask must have the same shape as binary")
        m = valid.astype(float)

    if m.sum() == 0:
        raise ValueError("valid mask contains no valid pixels")

    mean = float((x * m).sum() / m.sum())
    f = (x - mean) * m

    numerator = fftconvolve(f, f[::-1, ::-1], mode="full")
    pairs = fftconvolve(m, m[::-1, ::-1], mode="full")

    cov = np.zeros_like(numerator, dtype=float)
    ok = pairs > 0.5
    cov[ok] = numerator[ok] / pairs[ok]

    center = (x.shape[0] - 1, x.shape[1] - 1)
    c0 = float(cov[center])
    if not np.isfinite(c0) or c0 <= 0:
        raise ValueError("zero-lag covariance is not positive; check the binary image")

    return cov / c0, center


def directional_acf(
    acf_field: np.ndarray,
    center: tuple[int, int],
    theta_deg: float,
    max_lag: int,
):
    """Sample the autocorrelation along an in-plane direction.

    theta=0 follows image axis 0 / the first column of the section basis.
    """
    theta = np.deg2rad(float(theta_deg))
    r = np.arange(int(max_lag) + 1, dtype=float)
    rr = center[0] + r * np.cos(theta)
    cc = center[1] + r * np.sin(theta)
    values = map_coordinates(
        np.asarray(acf_field, dtype=float),
        np.vstack([rr, cc]),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    values[0] = 1.0
    return r, values


def first_crossing_length(
    r: np.ndarray,
    c: np.ndarray,
    threshold: float = float(np.exp(-1.0)),
) -> float:
    """Return the linearly interpolated first threshold crossing or NaN."""
    r = np.asarray(r, dtype=float)
    c = np.asarray(c, dtype=float)
    if r.shape != c.shape or r.ndim != 1:
        raise ValueError("r and c must be 1D arrays with identical shape")

    for i in range(1, len(c)):
        if np.isfinite(c[i]) and c[i] <= threshold:
            x1, x2 = r[i - 1], r[i]
            y1, y2 = c[i - 1], c[i]
            if not (np.isfinite(y1) and np.isfinite(y2)):
                return float("nan")
            if np.isclose(y1, y2):
                return float(x2)
            return float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1))
    return float("nan")


def measure_directional_lengths(
    binary: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    theta_step_deg: float = 5.0,
    max_lag: int | None = None,
    threshold: float = float(np.exp(-1.0)),
):
    """Measure directional correlation lengths over theta in [0, 180) degrees."""
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError("binary must be 2D")
    if theta_step_deg <= 0 or theta_step_deg >= 180:
        raise ValueError("theta_step_deg must be in (0, 180)")

    if max_lag is None:
        max_lag = max(1, min(binary.shape) // 4)
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValueError("max_lag must be positive")

    field, center = autocorrelation_field(binary, valid)
    theta = np.arange(0.0, 180.0, float(theta_step_deg), dtype=float)
    ell = np.empty(theta.shape, dtype=float)

    for i, angle in enumerate(theta):
        r, c = directional_acf(field, center, angle, max_lag)
        ell[i] = first_crossing_length(r, c, threshold)

    return theta, ell
