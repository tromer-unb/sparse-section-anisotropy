#!/usr/bin/env python3
"""
High-quality Figure 3 for the paper.

Scientific message
------------------
The section-level pore-correlation tensor recovers an unknown in-plane
principal direction without being given that direction a priori and obeys
the expected tensor rotation law under rigid rotations of the same porous
microstructure.

Generates
---------
  - figure3_highquality.png
  - figure3_highquality.svg
  - figure3_blind_rotation_results.csv

Dependencies
------------
  python3 -m pip install numpy scipy matplotlib

This script is fully standalone. It does not depend on figure1.py,
figure2.py, PoreSpy, TACC data, or any project module.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, rotate, map_coordinates
from scipy.signal import fftconvolve


# -----------------------------------------------------------------------------
# Settings -- matched to Figures 1 and 2
# -----------------------------------------------------------------------------
SEED = 41
POROSITY = 0.27

# Generate on a larger canvas and crop after rotation.  This avoids artificial
# corner filling in the analyzed region for arbitrary rotation angles.
N_CANVAS = 256
N_CROP = 160

# Base anisotropy.  Array coordinates are (row=y, column=x), hence the first
# Gaussian sigma controls y and the second controls x.  The long correlation
# axis is horizontal before rotation.
SIGMA_MINOR = 2.1
SIGMA_MAJOR = 6.0

EXAMPLE_ANGLE_DEG = 37.0
MAX_LAG = 42
THETA_STEP_DEG = 5
N_ENSEMBLE = 60
N_EQUIV_ANGLES = 13
DPI = 800

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "figure3_highquality.png"
SVG_PATH = OUTDIR / "figure3_highquality.svg"
CSV_PATH = OUTDIR / "figure3_blind_rotation_results.csv"

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
    "svg.fonttype": "none",   # editable text in Inkscape
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Synthetic 2D porous section with controlled orientation
# -----------------------------------------------------------------------------
def generate_base_section(seed, n=N_CANVAS, porosity=POROSITY):
    """
    Generate a thresholded anisotropic Gaussian random field whose long
    correlation axis is horizontal (0 deg) before rotation.
    """
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((n, n))
    field = gaussian_filter(
        field,
        sigma=(SIGMA_MINOR, SIGMA_MAJOR),
        mode="wrap",
    )
    threshold = np.quantile(field, 1.0 - porosity)
    return field >= threshold


def rotate_and_crop(binary, angle_deg, crop_n=N_CROP, order=1):
    """
    Actively rotate a binary section counter-clockwise and extract a central
    crop that is fully supported by the original canvas.

    Linear interpolation followed by a 0.5 threshold is the primary setting;
    the large parent canvas ensures the analyzed crop is not contaminated by
    the rotated image corners.
    """
    sampled = rotate(
        binary.astype(float),
        angle=-float(angle_deg),
        reshape=False,
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

    n = sampled.shape[0]
    start = (n - crop_n) // 2
    stop = start + crop_n
    crop = sampled[start:stop, start:stop]

    valid = np.isfinite(crop)
    if valid.mean() < 0.999:
        raise RuntimeError(
            "Central crop contains unsupported pixels. Increase N_CANVAS "
            "or decrease N_CROP."
        )

    return crop >= 0.5


# -----------------------------------------------------------------------------
# Unbiased normalized autocorrelation field
# -----------------------------------------------------------------------------
def autocorrelation_field(binary):
    x = binary.astype(float)
    m = np.ones_like(x, dtype=float)
    mean = x.mean()
    f = x - mean

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
    Sample the 2D autocorrelation field along an angle measured
    counter-clockwise from the horizontal image axis.
    """
    theta = np.deg2rad(theta_deg)
    r = np.arange(max_lag + 1, dtype=float)

    # array coordinates are (row=y, col=x)
    rr = center[0] + r * np.sin(theta)
    cc = center[1] + r * np.cos(theta)

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
    acf_field, center = autocorrelation_field(binary)
    theta = np.arange(0.0, 180.0, theta_step, dtype=float)
    ell = np.empty_like(theta)

    for i, th in enumerate(theta):
        r, c = directional_acf(acf_field, center, th, max_lag)
        ell[i] = correlation_length(r, c)

    return theta, ell, acf_field, center


# -----------------------------------------------------------------------------
# 2D tensor fit and orientation extraction
# -----------------------------------------------------------------------------
def fit_q2d(theta_deg, ell):
    """Fit 1/ell(theta)^2 = u^T Q u and project Q to SPD."""
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
    return V @ np.diag(w) @ V.T


def ell_from_q2d(Q, theta_deg):
    theta = np.deg2rad(theta_deg)
    u = np.vstack([np.cos(theta), np.sin(theta)])
    val = np.einsum("ij,ji->i", u.T @ Q, u)
    return 1.0 / np.sqrt(val)


def principal_summary(Q):
    """
    Return principal lengths, anisotropy ratio, and the orientation of the
    longest correlation axis in [0, 180) degrees.
    """
    w, V = np.linalg.eigh(Q)  # ascending eigenvalues
    lengths = 1.0 / np.sqrt(w)

    idx_long = int(np.argmax(lengths))
    vec = V[:, idx_long]  # coordinates are (x, y)
    angle = np.rad2deg(np.arctan2(vec[1], vec[0])) % 180.0

    A = float(lengths.max() / lengths.min())
    return lengths, A, float(angle)


def axial_error_deg(a_deg, b_deg):
    """Smallest difference between undirected axes, in degrees."""
    return float(abs((a_deg - b_deg + 90.0) % 180.0 - 90.0))


def align_axis_to_reference(angle_deg, reference_deg):
    """Return the 180-deg-equivalent angle nearest to a reference angle."""
    candidates = np.array([angle_deg - 180.0, angle_deg, angle_deg + 180.0])
    return float(candidates[np.argmin(np.abs(candidates - reference_deg))])


def determinant_normalize(Q):
    det = np.linalg.det(Q)
    if det <= 0:
        raise RuntimeError("Q must be positive definite.")
    return Q / np.sqrt(det)


def rotation_matrix(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=float)


# -----------------------------------------------------------------------------
# One blind recovery
# -----------------------------------------------------------------------------
def analyze_rotated_section(seed, angle_deg):
    base = generate_base_section(seed)
    img = rotate_and_crop(base, angle_deg)
    theta, ell, acf_field, center = directional_lengths(img)
    Q = fit_q2d(theta, ell)
    lengths, A, recovered = principal_summary(Q)
    return {
        "image": img,
        "theta": theta,
        "ell": ell,
        "acf_field": acf_field,
        "acf_center": center,
        "Q": Q,
        "lengths": lengths,
        "A": A,
        "recovered_angle": recovered,
    }


# -----------------------------------------------------------------------------
# Ensemble blind-angle experiment
# -----------------------------------------------------------------------------
def blind_rotation_ensemble(n_cases=N_ENSEMBLE, seed0=5000):
    rng = np.random.default_rng(20260907)

    # Avoid exact 0/180 boundary in the scatter visualization; the estimator
    # itself is axial and valid over the full [0, 180) interval.
    true_angles = rng.uniform(7.0, 173.0, size=n_cases)
    recovered = np.empty(n_cases)
    anisotropy = np.empty(n_cases)
    errors = np.empty(n_cases)

    for i, alpha in enumerate(true_angles):
        result = analyze_rotated_section(seed0 + i, alpha)
        recovered[i] = result["recovered_angle"]
        anisotropy[i] = result["A"]
        errors[i] = axial_error_deg(recovered[i], alpha)

    recovered_aligned = np.array([
        align_axis_to_reference(r, t)
        for r, t in zip(recovered, true_angles)
    ])

    return true_angles, recovered, recovered_aligned, anisotropy, errors


# -----------------------------------------------------------------------------
# Rotation-law experiment on the same realization
# -----------------------------------------------------------------------------
def rotation_equivariance(seed=SEED):
    angles = np.linspace(0.0, 165.0, N_EQUIV_ANGLES)
    tensors = []
    recovered_angles = []

    for alpha in angles:
        result = analyze_rotated_section(seed, alpha)
        tensors.append(result["Q"])
        recovered_angles.append(result["recovered_angle"])

    tensors = np.asarray(tensors)
    recovered_angles = np.asarray(recovered_angles)

    Q0 = determinant_normalize(tensors[0])
    predicted = []
    errors = []

    for alpha, Q in zip(angles, tensors):
        R = rotation_matrix(alpha)
        Q_pred = R @ Q0 @ R.T
        Q_obs = determinant_normalize(Q)

        predicted.append(Q_pred)
        errors.append(
            np.linalg.norm(Q_obs - Q_pred, ord="fro")
            / np.linalg.norm(Q_pred, ord="fro")
        )

    return angles, tensors, np.asarray(predicted), np.asarray(errors), recovered_angles


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
def panel_label(ax, label, x=0.01, y=0.99):
    text_fn = getattr(ax, "text2D", ax.text)
    text_fn(
        x, y, label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.12",
            fc="white",
            ec="none",
            alpha=0.85,
        ),
    )


def add_axis_line(ax, angle_deg, length, style="-", label=None, linewidth=2.0):
    n = N_CROP
    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0
    a = np.deg2rad(angle_deg)
    dx = 0.5 * length * np.cos(a)
    dy = 0.5 * length * np.sin(a)
    ax.plot(
        [cx - dx, cx + dx],
        [cy - dy, cy + dy],
        linestyle=style,
        linewidth=linewidth,
        label=label,
    )


def normalized_tensor_components(Qs):
    Qn = np.array([determinant_normalize(Q) for Q in Qs])
    return Qn[:, 0, 0], Qn[:, 0, 1], Qn[:, 1, 1]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Representative blind test
    # ------------------------------------------------------------------
    example = analyze_rotated_section(SEED, EXAMPLE_ANGLE_DEG)
    example_error = axial_error_deg(example["recovered_angle"], EXAMPLE_ANGLE_DEG)

    # ------------------------------------------------------------------
    # Blind-angle ensemble
    # ------------------------------------------------------------------
    (
        true_angles,
        recovered_raw,
        recovered_aligned,
        ensemble_A,
        ensemble_errors,
    ) = blind_rotation_ensemble()

    # ------------------------------------------------------------------
    # Rotation law on the same realization
    # ------------------------------------------------------------------
    eq_angles, eq_Q, eq_Q_pred, eq_errors, eq_recovered = rotation_equivariance()
    obs_xx, obs_xy, obs_yy = normalized_tensor_components(eq_Q)
    pred_xx = eq_Q_pred[:, 0, 0]
    pred_xy = eq_Q_pred[:, 0, 1]
    pred_yy = eq_Q_pred[:, 1, 1]

    # Save ensemble results for reproducibility
    table = np.column_stack([
        true_angles,
        recovered_raw,
        recovered_aligned,
        ensemble_errors,
        ensemble_A,
    ])
    np.savetxt(
        CSV_PATH,
        table,
        delimiter=",",
        header="true_angle_deg,recovered_angle_deg,recovered_aligned_deg,axial_error_deg,A_2D",
        comments="",
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 7.6))
    gs = GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.45,
        wspace=0.38,
    )

    # (a) Blind example
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(example["image"], origin="lower", cmap="gray", interpolation="nearest")
    add_axis_line(
        ax_a,
        EXAMPLE_ANGLE_DEG,
        length=0.72 * N_CROP,
        style="--",
        label="Ground truth",
        linewidth=1.7,
    )
    add_axis_line(
        ax_a,
        example["recovered_angle"],
        length=0.58 * N_CROP,
        style="-",
        label="Recovered",
        linewidth=2.2,
    )
    ax_a.set_title("Blind orientation recovery", pad=8, fontweight="bold")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.legend(frameon=False, loc="upper right")
    ax_a.text(
        0.03, 0.06,
        f"true = {EXAMPLE_ANGLE_DEG:.1f}$^\\circ$\n"
        f"recovered = {example['recovered_angle']:.1f}$^\\circ$\n"
        f"error = {example_error:.1f}$^\\circ$",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82),
    )
    panel_label(ax_a, "(a)")

    # (b) Angular response of the blind example
    ax_b = fig.add_subplot(gs[0, 1], projection="polar")
    good = np.isfinite(example["ell"])
    th = np.deg2rad(example["theta"][good])
    ell = example["ell"][good]
    th_full = np.concatenate([th, th + np.pi])
    ell_full = np.concatenate([ell, ell])

    th_fine_deg = np.linspace(0.0, 180.0, 721)
    ell_fit = ell_from_q2d(example["Q"], th_fine_deg)
    th_fine = np.deg2rad(th_fine_deg)
    fit_full_th = np.concatenate([th_fine, th_fine + np.pi])
    fit_full_ell = np.concatenate([ell_fit, ell_fit])

    ax_b.scatter(th_full, ell_full, s=13, label="Measured")
    ax_b.plot(fit_full_th, fit_full_ell, label="Fitted tensor")
    ax_b.set_theta_zero_location("E")
    ax_b.set_theta_direction(1)
    ax_b.set_rlabel_position(135)
    ax_b.set_title("Angular response of the blind case", pad=18, fontweight="bold")
    ax_b.legend(loc="lower left", bbox_to_anchor=(-0.05, -0.18), frameon=False)
    panel_label(ax_b, "(b)", x=-0.02, y=1.06)

    # (c) Tensor component rotation law
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.scatter(eq_angles, obs_xx, s=20, label=r"$q_{xx}$ measured")
    ax_c.scatter(eq_angles, obs_xy, s=20, label=r"$q_{xy}$ measured")
    ax_c.scatter(eq_angles, obs_yy, s=20, label=r"$q_{yy}$ measured")
    ax_c.plot(eq_angles, pred_xx, linestyle="--", linewidth=1.2)
    ax_c.plot(eq_angles, pred_xy, linestyle="--", linewidth=1.2)
    ax_c.plot(eq_angles, pred_yy, linestyle="--", linewidth=1.2)
    ax_c.set_xlabel("Applied rotation (deg)", fontweight="bold")
    ax_c.set_ylabel("Normalized tensor component", fontweight="bold")
    ax_c.set_title("Tensor components under rotation", pad=8, fontweight="bold")
    ax_c.legend(frameon=False, fontsize=7.2, ncol=1)
    panel_label(ax_c, "(c)")

    # (d) True vs recovered orientation
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.scatter(true_angles, recovered_aligned, s=22, alpha=0.82)
    lo = min(true_angles.min(), recovered_aligned.min()) - 5.0
    hi = max(true_angles.max(), recovered_aligned.max()) + 5.0
    ax_d.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="1:1")
    ax_d.set_xlim(lo, hi)
    ax_d.set_ylim(lo, hi)
    ax_d.set_xlabel("True principal orientation (deg)", fontweight="bold")
    ax_d.set_ylabel("Recovered orientation (deg)", fontweight="bold")
    ax_d.set_title("Blind recovery across realizations", pad=8, fontweight="bold")
    ax_d.legend(frameon=False, loc="upper left")
    panel_label(ax_d, "(d)")

    # (e) Angular-error distribution
    ax_e = fig.add_subplot(gs[1, 1])
    bins = np.arange(0.0, max(5.0, np.ceil(ensemble_errors.max()) + 1.0) + 1e-9, 1.0)
    ax_e.hist(ensemble_errors, bins=bins, edgecolor="white", linewidth=0.7)
    median_err = float(np.median(ensemble_errors))
    p95_err = float(np.percentile(ensemble_errors, 95))
    ax_e.axvline(median_err, linestyle="--", linewidth=1.3, label="Median")
    ax_e.axvline(p95_err, linestyle=":", linewidth=1.5, label="95th percentile")
    ax_e.set_xlabel("Absolute axial error (deg)", fontweight="bold")
    ax_e.set_ylabel("Number of realizations", fontweight="bold")
    ax_e.set_title("Orientation-error distribution", pad=8, fontweight="bold")
    ax_e.legend(frameon=False)
    ax_e.text(
        0.97, 0.94,
        f"median = {median_err:.2f}$^\\circ$\n"
        f"95th = {p95_err:.2f}$^\\circ$",
        transform=ax_e.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        fontweight="bold",
    )
    panel_label(ax_e, "(e)")

    # (f) Rotation-equivariance error
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.plot(eq_angles, 100.0 * eq_errors, marker="o", markersize=4.2)
    ax_f.axhline(100.0 * np.median(eq_errors), linestyle="--", linewidth=1.2, label="Median")
    ax_f.set_xlabel("Applied rotation (deg)", fontweight="bold")
    ax_f.set_ylabel("Equivariance error (%)", fontweight="bold")
    ax_f.set_title("Rotation equivariance of $Q_{2D}$", pad=8, fontweight="bold")
    ax_f.legend(frameon=False)
    ax_f.text(
        0.97, 0.94,
        f"median = {100*np.median(eq_errors):.2f}%\n"
        f"max = {100*np.max(eq_errors):.2f}%",
        transform=ax_f.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        fontweight="bold",
    )
    panel_label(ax_f, "(f)")

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.96)
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Figure 3 generated successfully.")
    print(f"PNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"CSV: {CSV_PATH}")
    print()
    print("Representative blind case")
    print(f"  true angle       = {EXAMPLE_ANGLE_DEG:.3f} deg")
    print(f"  recovered angle  = {example['recovered_angle']:.3f} deg")
    print(f"  axial error      = {example_error:.3f} deg")
    print(f"  A_2D             = {example['A']:.3f}")
    print()
    print("Blind-angle ensemble")
    print(f"  n                = {N_ENSEMBLE}")
    print(f"  median error     = {np.median(ensemble_errors):.3f} deg")
    print(f"  95th percentile  = {np.percentile(ensemble_errors, 95):.3f} deg")
    print(f"  mean A_2D        = {np.mean(ensemble_A):.3f}")
    print()
    print("Rotation equivariance")
    print(f"  median error     = {100*np.median(eq_errors):.3f}%")
    print(f"  maximum error    = {100*np.max(eq_errors):.3f}%")


if __name__ == "__main__":
    main()