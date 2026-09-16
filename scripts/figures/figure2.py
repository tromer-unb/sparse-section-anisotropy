#!/usr/bin/env python3
"""
High-quality Figure 2 for the paper.

Scientific message
------------------
A single oriented 2D section can detect anisotropy only when the preferred
3D direction has a component inside the observed plane. The figure compares
isotropic and anisotropic synthetic porous media, their directional
correlation functions, tensor fits, and an ensemble-level negative control.

Generates
---------
  - figure2_highquality.png
  - figure2_highquality.svg

Dependencies
------------
  python3 -m pip install numpy scipy matplotlib

This script is fully standalone and does not depend on figure1.py or any
other project script.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve


# -----------------------------------------------------------------------------
# Settings -- matched to the revised Figure 1
# -----------------------------------------------------------------------------
SEED = 30
N = 96
POROSITY = 0.27

SIGMA_ISO = (4.8, 4.8, 4.8)
SIGMA_ANISO = (4.8, 4.8, 1.8)

MAX_LAG = 28
THETA_STEP_DEG = 5
N_ENSEMBLE = 20
DPI = 800

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "figure2_highquality.png"
SVG_PATH = OUTDIR / "figure2_highquality.svg"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "font.weight": "bold",
    "axes.titlesize": 11.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.6,
    "svg.fonttype": "none",   # keep text editable in Inkscape
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Synthetic porous media
# -----------------------------------------------------------------------------
def generate_binary_medium(n, porosity, sigmas, seed):
    """Generate a thresholded anisotropic Gaussian random field."""
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((n, n, n))
    field = gaussian_filter(field, sigma=sigmas, mode="wrap")
    threshold = np.quantile(field, 1.0 - porosity)
    return field >= threshold


def central_section(volume, plane):
    """
    Return a central orthogonal section.

    plane='XZ' -> y = const, output axes are (X, Z)
    plane='XY' -> z = const, output axes are (X, Y)
    """
    c = volume.shape[0] // 2
    if plane.upper() == "XZ":
        return volume[:, c, :]
    if plane.upper() == "XY":
        return volume[:, :, c]
    raise ValueError("plane must be 'XZ' or 'XY'")


# -----------------------------------------------------------------------------
# Unbiased normalized autocorrelation field
# -----------------------------------------------------------------------------
def autocorrelation_field(binary, valid=None):
    """
    Compute the unbiased normalized 2D autocorrelation field using FFT
    convolution and explicit valid-pair counting.
    """
    if valid is None:
        valid = np.ones_like(binary, dtype=bool)

    x = binary.astype(float)
    m = valid.astype(float)
    mean = (x * m).sum() / m.sum()
    f = (x - mean) * m

    numerator = fftconvolve(f, f[::-1, ::-1], mode="full")
    pairs = fftconvolve(m, m[::-1, ::-1], mode="full")

    cov = np.zeros_like(numerator, dtype=float)
    ok = pairs > 0.5
    cov[ok] = numerator[ok] / pairs[ok]

    center = (binary.shape[0] - 1, binary.shape[1] - 1)
    c0 = cov[center]
    if not np.isfinite(c0) or c0 <= 0:
        raise RuntimeError("Invalid zero-lag covariance.")

    return cov / c0, center


def directional_acf(acf_field, center, theta_deg, max_lag=MAX_LAG):
    """
    Sample the 2D autocorrelation field along an arbitrary in-plane angle.

    theta=0 deg samples the first array axis.
    theta=90 deg samples the second array axis.
    """
    theta = np.deg2rad(theta_deg)
    r = np.arange(max_lag + 1, dtype=float)

    rr = center[0] + r * np.cos(theta)
    cc = center[1] + r * np.sin(theta)

    vals = map_coordinates(
        acf_field,
        np.vstack([rr, cc]),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    vals[0] = 1.0
    return r, vals


def correlation_length(r, c, threshold=np.exp(-1.0)):
    """First interpolated crossing of C(r)=1/e."""
    for i in range(1, len(c)):
        if c[i] <= threshold:
            x1, x2 = r[i - 1], r[i]
            y1, y2 = c[i - 1], c[i]
            if np.isclose(y1, y2):
                return float(x2)
            return float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1))
    return np.nan


def directional_lengths(binary, max_lag=MAX_LAG, theta_step=THETA_STEP_DEG):
    """Measure ell(theta) over [0, 180) degrees."""
    acf_field, center = autocorrelation_field(binary)
    theta = np.arange(0.0, 180.0, theta_step, dtype=float)
    ell = np.empty_like(theta)

    for i, th in enumerate(theta):
        r, c = directional_acf(acf_field, center, th, max_lag)
        ell[i] = correlation_length(r, c)

    return theta, ell, acf_field, center


# -----------------------------------------------------------------------------
# 2D tensor fit
# -----------------------------------------------------------------------------
def fit_q2d(theta_deg, ell):
    """
    Fit the symmetric tensor Q from 1/ell(theta)^2 = u^T Q u.

    The unconstrained least-squares result is projected to the nearest
    positive-definite matrix by flooring its eigenvalues. This is adequate
    for the clean synthetic proof-of-concept used in Figure 2.
    """
    theta = np.deg2rad(theta_deg)
    good = np.isfinite(ell) & (ell > 0)
    theta = theta[good]
    ell = ell[good]

    c = np.cos(theta)
    s = np.sin(theta)
    A = np.column_stack([c * c, 2.0 * c * s, s * s])
    y = 1.0 / (ell * ell)

    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    Q = np.array([
        [coef[0], coef[1]],
        [coef[1], coef[2]],
    ], dtype=float)

    w, V = np.linalg.eigh(Q)
    floor = max(1e-8, 1e-6 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    Q = V @ np.diag(w) @ V.T
    return Q


def ell_from_q2d(Q, theta_deg):
    theta = np.deg2rad(theta_deg)
    u = np.vstack([np.cos(theta), np.sin(theta)])
    val = np.einsum("ij,ji->i", u.T @ Q, u)
    return 1.0 / np.sqrt(val)


def anisotropy_ratio(Q):
    """A = ell_max / ell_min."""
    eigenvalues = np.linalg.eigvalsh(Q)
    return float(np.sqrt(eigenvalues.max() / eigenvalues.min()))


def tensor_summary(binary):
    theta, ell, acf_field, center = directional_lengths(binary)
    Q = fit_q2d(theta, ell)
    return {
        "theta": theta,
        "ell": ell,
        "Q": Q,
        "A": anisotropy_ratio(Q),
        "acf_field": acf_field,
        "center": center,
    }


# -----------------------------------------------------------------------------
# Ensemble calculations for the negative control panel
# -----------------------------------------------------------------------------
def ensemble_anisotropy(n_ensemble=N_ENSEMBLE):
    """
    Compare in-plane and out-of-plane detectability across independent
    realizations. The imposed short-correlation direction is Z.

    XZ contains Z -> anisotropy should be detected.
    XY excludes Z -> anisotropy should largely disappear.
    """
    iso_xz = np.empty(n_ensemble)
    aniso_xz = np.empty(n_ensemble)
    iso_xy = np.empty(n_ensemble)
    aniso_xy = np.empty(n_ensemble)

    for seed in range(n_ensemble):
        vol_iso = generate_binary_medium(N, POROSITY, SIGMA_ISO, seed)
        vol_aniso = generate_binary_medium(N, POROSITY, SIGMA_ANISO, seed)

        iso_xz[seed] = tensor_summary(central_section(vol_iso, "XZ"))["A"]
        aniso_xz[seed] = tensor_summary(central_section(vol_aniso, "XZ"))["A"]
        iso_xy[seed] = tensor_summary(central_section(vol_iso, "XY"))["A"]
        aniso_xy[seed] = tensor_summary(central_section(vol_aniso, "XY"))["A"]

    return iso_xz, aniso_xz, iso_xy, aniso_xy


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
def panel_label(ax, label, x=0.01, y=0.99):
    text_fn = getattr(ax, "text2D", ax.text)
    text_fn(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.12",
            fc="white",
            ec="none",
            alpha=0.88,
        ),
    )


def show_binary(ax, image, title):
    ax.imshow(image.T, origin="lower", cmap="gray", interpolation="nearest")
    ax.set_title(title, pad=8, fontweight="bold")
    ax.set_xlabel("First in-plane axis", fontweight="bold")
    ax.set_ylabel("Second in-plane axis", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def polar_tensor_panel(ax, summary, title):
    theta = summary["theta"]
    ell = summary["ell"]
    Q = summary["Q"]

    good = np.isfinite(ell)
    theta_rad = np.deg2rad(theta[good])
    ell_good = ell[good]

    theta_full = np.concatenate([theta_rad, theta_rad + np.pi])
    ell_full = np.concatenate([ell_good, ell_good])

    theta_fine = np.linspace(0.0, 180.0, 721)
    ell_fit = ell_from_q2d(Q, theta_fine)
    fit_rad = np.deg2rad(theta_fine)
    fit_full_th = np.concatenate([fit_rad, fit_rad + np.pi])
    fit_full_r = np.concatenate([ell_fit, ell_fit])

    ax.scatter(theta_full, ell_full, s=11, label="Measured")
    ax.plot(fit_full_th, fit_full_r, label="Tensor fit")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(135)
    ax.set_title(title, pad=18, fontweight="bold")
    ax.text(
        0.50,
        -0.12,
        f"A$_{{2D}}$ = {summary['A']:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        fontweight="bold",
    )


def ensemble_panel(ax, data):
    iso_xz, aniso_xz, iso_xy, aniso_xy = data
    datasets = [iso_xz, aniso_xz, iso_xy, aniso_xy]
    labels = ["Iso\nXZ", "Aniso\nXZ", "Iso\nXY", "Aniso\nXY"]
    positions = np.arange(1, 5)

    bp = ax.boxplot(
        datasets,
        positions=positions,
        widths=0.52,
        patch_artist=False,
        showfliers=False,
        medianprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )

    # deterministic horizontal jitter for visibility
    rng = np.random.default_rng(1234)
    for x, vals in zip(positions, datasets):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full_like(vals, x, dtype=float) + jitter,
            vals,
            s=13,
            alpha=0.62,
            edgecolors="none",
        )

    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel(r"Anisotropy ratio $A_{2D}$", fontweight="bold")
    ax.set_title("In-plane detectability across realizations", pad=8, fontweight="bold")
    ax.set_xlim(0.5, 4.5)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Representative realization
    vol_iso = generate_binary_medium(N, POROSITY, SIGMA_ISO, SEED)
    vol_aniso = generate_binary_medium(N, POROSITY, SIGMA_ANISO, SEED)

    img_iso_xz = central_section(vol_iso, "XZ")
    img_aniso_xz = central_section(vol_aniso, "XZ")

    iso_summary = tensor_summary(img_iso_xz)
    aniso_summary = tensor_summary(img_aniso_xz)

    # Curves used in panel (c)
    curves = {}
    for name, summary in (("Iso", iso_summary), ("Aniso", aniso_summary)):
        for th, axis_name in ((0, "X"), (90, "Z")):
            curves[(name, axis_name)] = directional_acf(
                summary["acf_field"],
                summary["center"],
                th,
                MAX_LAG,
            )

    # Ensemble-level negative control
    ensemble = ensemble_anisotropy(N_ENSEMBLE)

    # -------------------------------------------------------------------------
    # Figure layout: 2 x 3
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(8.8, 7.3))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[1.0, 1.02],
        hspace=0.44,
        wspace=0.42,
    )

    # (a) isotropic section
    ax_a = fig.add_subplot(gs[0, 0])
    show_binary(ax_a, img_iso_xz, "Isotropic XZ section")
    panel_label(ax_a, "(a)")

    # (b) anisotropic section
    ax_b = fig.add_subplot(gs[0, 1])
    show_binary(ax_b, img_aniso_xz, "Anisotropic XZ section")
    panel_label(ax_b, "(b)")

    # (c) directional ACF comparison
    ax_c = fig.add_subplot(gs[0, 2])
    line_styles = {
        ("Iso", "X"): "-",
        ("Iso", "Z"): "--",
        ("Aniso", "X"): "-.",
        ("Aniso", "Z"): ":",
    }
    for key in (("Iso", "X"), ("Iso", "Z"), ("Aniso", "X"), ("Aniso", "Z")):
        r, c = curves[key]
        ax_c.plot(r, c, linestyle=line_styles[key], label=f"{key[0]} {key[1]}")

    ax_c.axhline(np.exp(-1.0), linestyle="--", linewidth=1.0, label="1/e")
    ax_c.axhline(0.0, linestyle=":", linewidth=0.8)
    ax_c.set_xlim(0, MAX_LAG)
    ax_c.set_ylim(-0.12, 1.04)
    ax_c.set_xlabel("Separation r (pixels)", fontweight="bold")
    ax_c.set_ylabel("Autocorrelation", fontweight="bold")
    ax_c.set_title("Directional correlation decay", pad=8, fontweight="bold")
    ax_c.legend(frameon=False, ncol=1, handlelength=2.4)
    panel_label(ax_c, "(c)")

    # (d) isotropic tensor response
    ax_d = fig.add_subplot(gs[1, 0], projection="polar")
    polar_tensor_panel(ax_d, iso_summary, "Isotropic angular response")
    panel_label(ax_d, "(d)")

    # (e) anisotropic tensor response
    ax_e = fig.add_subplot(gs[1, 1], projection="polar")
    polar_tensor_panel(ax_e, aniso_summary, "Anisotropic angular response")
    panel_label(ax_e, "(e)")

    # (f) ensemble negative control
    ax_f = fig.add_subplot(gs[1, 2])
    ensemble_panel(ax_f, ensemble)
    panel_label(ax_f, "(f)")

    # No inter-panel arrows: panels (a)--(c) are comparisons, not a process sequence.

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.965)

    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Reproducibility summary printed to the terminal
    iso_xz, aniso_xz, iso_xy, aniso_xy = ensemble

    print("Figure 2 generated successfully.")
    print(f"PNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")
    print()
    print("Representative section:")
    print(f"  A2D isotropic XZ   = {iso_summary['A']:.4f}")
    print(f"  A2D anisotropic XZ = {aniso_summary['A']:.4f}")
    print()
    print("Ensemble mean +/- SD:")
    for name, vals in (
        ("Iso XZ", iso_xz),
        ("Aniso XZ", aniso_xz),
        ("Iso XY", iso_xy),
        ("Aniso XY", aniso_xy),
    ):
        print(f"  {name:9s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()