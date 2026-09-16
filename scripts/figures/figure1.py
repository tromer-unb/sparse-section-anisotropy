#!/usr/bin/env python3
"""
High-quality Figure 1 for the paper.
Generates:
  - figure1_highquality.png
  - figure1_highquality.svg

Dependencies:
  python3 -m pip install numpy scipy matplotlib
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
# Settings
# -----------------------------------------------------------------------------
SEED = 17
N = 96
POROSITY = 0.27
SIGMA_XYZ = (4.8, 4.8, 1.8)
SECTION_ANGLE_DEG = 48.0
MAX_LAG = 28
THETA_STEP_DEG = 5
DPI = 800

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "figure1_highquality.png"
SVG_PATH = OUTDIR / "figure1_highquality.svg"

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
    "legend.fontsize": 8.2,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Synthetic porous medium
# -----------------------------------------------------------------------------
def generate_binary_medium(n=N, porosity=POROSITY, sigmas=SIGMA_XYZ, seed=SEED):
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((n, n, n))
    field = gaussian_filter(field, sigma=sigmas, mode="wrap")
    threshold = np.quantile(field, 1.0 - porosity)
    return field >= threshold


# -----------------------------------------------------------------------------
# Oriented section extraction
# -----------------------------------------------------------------------------
def extract_section(volume, theta_deg, size=None, order=1):
    if size is None:
        size = min(volume.shape)

    theta = np.deg2rad(theta_deg)
    center = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0

    u = np.arange(size, dtype=float) - (size - 1.0) / 2.0
    v = np.arange(size, dtype=float) - (size - 1.0) / 2.0
    U, V = np.meshgrid(u, v, indexing="ij")

    X = center[0] + U
    Y = center[1] + V * np.cos(theta)
    Z = center[2] + V * np.sin(theta)

    valid = (
        (X >= 0) & (X <= volume.shape[0] - 1) &
        (Y >= 0) & (Y <= volume.shape[1] - 1) &
        (Z >= 0) & (Z <= volume.shape[2] - 1)
    )

    coords = np.array([X, Y, Z])
    sampled = map_coordinates(
        volume.astype(float),
        coords,
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

    binary = sampled >= 0.5
    binary[~valid] = False
    return binary, valid


# -----------------------------------------------------------------------------
# Unbiased normalized autocorrelation field
# -----------------------------------------------------------------------------
def autocorrelation_field(binary, valid=None):
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
    return cov / c0, center


def directional_acf(acf_field, center, theta_deg, max_lag=MAX_LAG):
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
    for i in range(1, len(c)):
        if c[i] <= threshold:
            x1, x2 = r[i - 1], r[i]
            y1, y2 = c[i - 1], c[i]
            if np.isclose(y1, y2):
                return float(x2)
            return float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1))
    return np.nan


# -----------------------------------------------------------------------------
# Tensor fit in 2D
# -----------------------------------------------------------------------------
def fit_q2d(theta_deg, ell):
    theta = np.deg2rad(theta_deg)
    good = np.isfinite(ell) & (ell > 0)
    theta = theta[good]
    ell = ell[good]

    c = np.cos(theta)
    s = np.sin(theta)
    A = np.column_stack([c * c, 2.0 * c * s, s * s])
    y = 1.0 / (ell * ell)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    Q = np.array([[coef[0], coef[1]], [coef[1], coef[2]]], dtype=float)

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


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
def panel_label(ax, label, x=0.01, y=0.99):
    text_fn = getattr(ax, "text2D", ax.text)
    text_fn(x, y, label, transform=ax.transAxes,
            fontsize=15, fontweight="bold", ha="left", va="top", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def draw_cube(ax, n):
    edges = [
        ((0, 0, 0), (n, 0, 0)), ((0, n, 0), (n, n, 0)),
        ((0, 0, n), (n, 0, n)), ((0, n, n), (n, n, n)),
        ((0, 0, 0), (0, n, 0)), ((n, 0, 0), (n, n, 0)),
        ((0, 0, n), (0, n, n)), ((n, 0, n), (n, n, n)),
        ((0, 0, 0), (0, 0, n)), ((n, 0, 0), (n, 0, n)),
        ((0, n, 0), (0, n, n)), ((n, n, 0), (n, n, n)),
    ]
    for p0, p1 in edges:
        ax.plot(*zip(p0, p1), linewidth=0.8, alpha=0.65, color="black")


def draw_section_planes(ax, n):
    c = n / 2.0
    g = np.linspace(0, n, 2)
    X, Y = np.meshgrid(g, g)
    Z = np.full_like(X, c)
    ax.plot_surface(X, Y, Z, alpha=0.14, linewidth=0)

    X2, Z2 = np.meshgrid(g, g)
    Y2 = np.full_like(X2, c)
    ax.plot_surface(X2, Y2, Z2, alpha=0.12, linewidth=0)

    th = np.deg2rad(SECTION_ANGLE_DEG)
    uu = np.linspace(-n * 0.46, n * 0.46, 2)
    vv = np.linspace(-n * 0.46, n * 0.46, 2)
    U, V = np.meshgrid(uu, vv)
    XO = c + U
    YO = c + V * np.cos(th)
    ZO = c + V * np.sin(th)
    ax.plot_surface(XO, YO, ZO, alpha=0.20, linewidth=0)


def sparse_pore_scatter(ax, volume, n_points=850, seed=123):
    rng = np.random.default_rng(seed)
    coords = np.argwhere(volume)
    if len(coords) > n_points:
        coords = coords[rng.choice(len(coords), size=n_points, replace=False)]
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1.4, alpha=0.08, color="black")


def add_direction_arrows(ax, img):
    h, w = img.shape
    cx, cy = w * 0.50, h * 0.50
    length = 0.28 * min(h, w)
    for ang in (0, 45, 90, 135):
        th = np.deg2rad(ang)
        dx = length * np.cos(th)
        dy = -length * np.sin(th)
        ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5, mutation_scale=14))
        tx = cx + 1.15 * dx
        ty = cy + 1.15 * dy
        ax.text(tx, ty, rf"{ang}$^\circ$", ha="center", va="center",
                fontsize=8.0, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.75))


def make_ellipsoid_from_lengths(lengths, n_u=48, n_v=30):
    a, b, c = lengths
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    U, V = np.meshgrid(u, v)
    X = a * np.cos(U) * np.sin(V)
    Y = b * np.sin(U) * np.sin(V)
    Z = c * np.cos(V)
    return X, Y, Z


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    volume = generate_binary_medium()
    section, valid = extract_section(volume, SECTION_ANGLE_DEG, order=1)
    acf_field, acf_center = autocorrelation_field(section, valid)

    theta_all = np.arange(0, 180, THETA_STEP_DEG, dtype=float)
    ell_all = []
    curves = {}
    for th in theta_all:
        r, c = directional_acf(acf_field, acf_center, th, MAX_LAG)
        ell_all.append(correlation_length(r, c))
    ell_all = np.asarray(ell_all)

    for th in (0, 45, 90, 135):
        curves[th] = directional_acf(acf_field, acf_center, th, MAX_LAG)

    Q2 = fit_q2d(theta_all, ell_all)
    theta_fine = np.linspace(0, 180, 721)
    ell_fit = ell_from_q2d(Q2, theta_fine)

    # Conceptual 3D output ellipsoid for the workflow figure
    ell3 = np.asarray(SIGMA_XYZ, dtype=float)
    ell3 = 1.0 + 0.55 * ell3 / ell3.min()

    fig = plt.figure(figsize=(8.8, 7.3))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 0.98], hspace=0.42, wspace=0.65)

    # (a)
    ax_a = fig.add_subplot(gs[0, 0:2], projection="3d")
    sparse_pore_scatter(ax_a, volume)
    draw_cube(ax_a, N)
    draw_section_planes(ax_a, N)
    ax_a.set_xlim(0, N); ax_a.set_ylim(0, N); ax_a.set_zlim(0, N)
    ax_a.set_box_aspect((1, 1, 1))
    ax_a.view_init(elev=23, azim=-55)
    ax_a.set_xlabel("x", labelpad=-7, fontweight="bold")
    ax_a.set_ylabel("y", labelpad=-7, fontweight="bold")
    ax_a.set_zlabel("z", labelpad=-5, fontweight="bold")
    ax_a.set_xticks([]); ax_a.set_yticks([]); ax_a.set_zticks([])
    ax_a.set_title("Sparse oriented sections", pad=6, fontweight="bold")
    ax_a.grid(False)
    for axis in (ax_a.xaxis, ax_a.yaxis, ax_a.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
        except Exception:
            pass
    panel_label(ax_a, "(a)")

    # (b)
    ax_b = fig.add_subplot(gs[0, 2:4])
    shown = np.where(valid, section.astype(float), np.nan)
    ax_b.imshow(shown.T, origin="lower", cmap="gray", interpolation="nearest")
    add_direction_arrows(ax_b, section)
    ax_b.set_title("Representative oriented section", pad=8, fontweight="bold")
    ax_b.set_xlabel(r"$\xi_1$", fontweight="bold")
    ax_b.set_ylabel(r"$\xi_2$", fontweight="bold")
    ax_b.set_xticks([]); ax_b.set_yticks([])
    panel_label(ax_b, "(b)")

    # (c)
    ax_c = fig.add_subplot(gs[0, 4:6])
    for th, (r, c) in curves.items():
        ax_c.plot(r, c, label=rf"{th}$^\circ$")
    ax_c.axhline(np.exp(-1), linestyle="--", linewidth=1.1, label="threshold")
    ax_c.axhline(0, linestyle=":", linewidth=0.9)
    ax_c.set_xlim(0, MAX_LAG)
    ax_c.set_ylim(-0.12, 1.04)
    ax_c.set_xlabel("Separation r (pixels)", fontweight="bold")
    ax_c.set_ylabel("Autocorrelation", fontweight="bold")
    ax_c.set_title("Directional autocorrelation", pad=8, fontweight="bold")
    ax_c.legend(frameon=False, ncol=2, handlelength=2.2, columnspacing=0.9)
    panel_label(ax_c, "(c)")

    # (d)
    ax_d = fig.add_subplot(gs[1, 0:3], projection="polar")
    good = np.isfinite(ell_all)
    theta_rad = np.deg2rad(theta_all[good])
    ell_good = ell_all[good]
    theta_full = np.concatenate([theta_rad, theta_rad + np.pi])
    ell_full = np.concatenate([ell_good, ell_good])
    fit_rad = np.deg2rad(theta_fine)
    fit_full_th = np.concatenate([fit_rad, fit_rad + np.pi])
    fit_full_r = np.concatenate([ell_fit, ell_fit])
    ax_d.scatter(theta_full, ell_full, s=14, label="Measured")
    ax_d.plot(fit_full_th, fit_full_r, label="Fitted tensor")
    ax_d.set_theta_zero_location("E")
    ax_d.set_theta_direction(1)
    ax_d.set_rlabel_position(135)
    ax_d.set_title("Angular response and fitted 2D tensor", pad=18, fontweight="bold")
    ax_d.legend(loc="lower left", bbox_to_anchor=(0.00, -0.18), frameon=False)
    panel_label(ax_d, "(d)")

    # (e)
    ax_e = fig.add_subplot(gs[1, 3:6], projection="3d")
    XE, YE, ZE = make_ellipsoid_from_lengths(ell3)
    ax_e.plot_wireframe(XE, YE, ZE, rstride=2, cstride=3, linewidth=0.7, alpha=0.95, color="black")
    m = float(np.max(ell3) * 1.22)
    ax_e.plot([-m, m], [0, 0], [0, 0], linewidth=1.0, color="black")
    ax_e.plot([0, 0], [-m, m], [0, 0], linewidth=1.0, color="black")
    ax_e.plot([0, 0], [0, 0], [-m, m], linewidth=1.0, color="black")
    ax_e.set_xlim(-m, m); ax_e.set_ylim(-m, m); ax_e.set_zlim(-m, m)
    ax_e.set_box_aspect((1, 1, 1))
    ax_e.view_init(elev=22, azim=-52)
    ax_e.set_xticks([]); ax_e.set_yticks([]); ax_e.set_zticks([])
    ax_e.set_xlabel("e1", labelpad=-6, fontweight="bold")
    ax_e.set_ylabel("e2", labelpad=-6, fontweight="bold")
    ax_e.set_zlabel("e3", labelpad=-4, fontweight="bold")
    ax_e.set_title("Recovered 3D anisotropy ellipsoid", pad=6, fontweight="bold")
    ax_e.grid(False)
    for axis in (ax_e.xaxis, ax_e.yaxis, ax_e.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
        except Exception:
            pass
    ax_e.text2D(0.5, -0.08, "Section tensors are combined to recover the 3D anisotropy descriptor",
                transform=ax_e.transAxes, ha="center", va="top", fontsize=9.0, fontweight="bold")
    panel_label(ax_e, "(e)", x=0.01, y=0.99)

    # pipeline arrows between top panels
    for x0, x1 in ((0.323, 0.357), (0.652, 0.686)):
        arrow = patches.FancyArrowPatch(
            (x0, 0.735), (x1, 0.735),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.1,
            color="black",
        )
        fig.add_artist(arrow)

    fig.text(0.50, 0.485,
             "Directional correlation lengths are summarized by a 2D tensor and then combined across sections.",
             ha="center", va="center", fontsize=9.4, fontweight="bold")

    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.06, top=0.965)
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Figure 1 generated successfully.")
    print(f"PNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")


if __name__ == "__main__":
    main()