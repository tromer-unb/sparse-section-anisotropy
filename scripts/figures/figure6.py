#!/usr/bin/env python3
"""
Figure 6 — Real-rock validation: Bentheimer sandstone vs Edwards Brown carbonate.

This script DOES NOT recompute any 3D reference, ACF, section measurement, or
Monte-Carlo reconstruction. It reads the frozen final CSV outputs committed under
results/figure6 and extracts one representative 512x512 raw slice per rock.

Expected inputs
---------------
Bentheimer:
  results/figure6/bentheimer_final_subvolumes.csv
  results/figure6/bentheimer_final_summary.csv
  $SPARSE_SECTION_DATA_DIR/Bentheimer_15A/ROI1/kocurek_15a_..._binary_ROI-1.raw

Edwards Brown:
  results/figure6/edb1_final_subvolumes.csv
  results/figure6/edb1_final_summary.csv
  $SPARSE_SECTION_DATA_DIR/EdwardsBrown_EdB1/ROI1/edb-1_..._binary_ROI-1.raw

Outputs
-------
  figure6_real_rock_comparison.png
  figure6_real_rock_comparison.svg
  figure6_real_rock_comparison_summary.csv
  figure6_real_rock_comparison_report.txt

Dependencies
------------
  python3 -m pip install numpy matplotlib

Run
---
  python3 figure6_real_rock_comparison.py
"""

from pathlib import Path
import os
import csv
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


# =============================================================================
# PROJECT-RELATIVE PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "figure6"

# Raw volumes are intentionally not tracked in Git. By default they are expected
# under data/raw, matching the directory layout created by download_real_rocks.py.
# Override the root without editing this script:
#   SPARSE_SECTION_DATA_DIR=/path/to/real_rocks python scripts/figures/figure6.py
DATA_ROOT = Path(os.environ.get(
    "SPARSE_SECTION_DATA_DIR",
    str(REPO_ROOT / "data" / "raw"),
)).expanduser().resolve()

BENTHEIMER_RAW = DATA_ROOT / "Bentheimer_15A" / "ROI1" / (
    "kocurek_15a_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw"
)
EDB1_RAW = DATA_ROOT / "EdwardsBrown_EdB1" / "ROI1" / (
    "edb-1_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw"
)

BENT_SUBVOL = RESULTS_DIR / "bentheimer_final_subvolumes.csv"
BENT_SUMMARY = RESULTS_DIR / "bentheimer_final_summary.csv"
EDB_SUBVOL = RESULTS_DIR / "edb1_final_subvolumes.csv"
EDB_SUMMARY = RESULTS_DIR / "edb1_final_summary.csv"

OUTDIR = REPO_ROOT / "outputs" / "figure6"
OUTDIR.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUTDIR / "figure6_real_rock_comparison.png"
SVG_PATH = OUTDIR / "figure6_real_rock_comparison.svg"
CSV_PATH = OUTDIR / "figure6_real_rock_comparison_summary.csv"
REPORT_PATH = OUTDIR / "figure6_real_rock_comparison_report.txt"


# =============================================================================
# FROZEN EXPERIMENTAL CONSTANTS
# =============================================================================

RAW_SHAPE = (2500, 2500, 2500)
RAW_DTYPE = np.uint8
SUBVOL_N = 512
VOXEL_SIZE_UM = 2.25
TOTAL_SECTIONS = (3, 9, 15, 21)

# Reference calibration established on Bentheimer and used as the frozen protocol.
# Do not interpret this as an independently calibrated EdB-1 uncertainty.
REF_PROTOCOL_P95_BENTHEIMER = 0.01706

DPI = 800

# Grayscale / print-friendly styles.
BENT_STYLE = dict(color="black", marker="o", linestyle="-", linewidth=1.9)
EDB_STYLE = dict(color="0.45", marker="s", linestyle="--", linewidth=1.9)


# =============================================================================
# PAPER STYLE
# =============================================================================

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


# =============================================================================
# IO / UTILITIES
# =============================================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n  {path}\n\n"
            "Edit the USER PATHS block at the top of the script if your files "
            "are in a different directory."
        )


def read_csv(path):
    require_file(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=np.nan):
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def as_int(row, key, default=0):
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def pct(x):
    return 100.0 * np.asarray(x, dtype=float)


def qsummary(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(median=np.nan, q1=np.nan, q3=np.nan, min=np.nan, max=np.nan)
    return {
        "median": float(np.median(x)),
        "q1": float(np.percentile(x, 25)),
        "q3": float(np.percentile(x, 75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=11.5, fontweight="bold",
        clip_on=False,
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.9)


def parse_final_data(subvol_path, summary_path, rock_name):
    sub = read_csv(subvol_path)
    summ = read_csv(summary_path)

    if len(sub) != 27:
        raise RuntimeError(
            f"{rock_name}: expected 27 subvolumes in {subvol_path.name}, found {len(sub)}"
        )

    # Subvolume table.
    sub_rows = []
    for r in sub:
        sub_rows.append({
            "id": r["subvolume_id"],
            "grid_i": as_int(r, "grid_i"),
            "grid_j": as_int(r, "grid_j"),
            "grid_k": as_int(r, "grid_k"),
            "center": (
                as_int(r, "center_a0"),
                as_int(r, "center_a1"),
                as_int(r, "center_a2"),
            ),
            "phi": as_float(r, "porosity"),
            "A_ref": as_float(r, "A_ref"),
            "eta_ref": as_float(r, "seed_eta_median"),
            "major_gap": as_float(r, "major_gap"),
        })

    # Summary table indexed by total section count.
    by_total = {n: [] for n in TOTAL_SECTIONS}
    for r in summ:
        total = as_int(r, "total_sections")
        if total not in by_total:
            continue
        by_total[total].append({
            "id": r["subvolume_id"],
            "E_med": as_float(r, "E_median"),
            "E_q1": as_float(r, "E_q1"),
            "E_q3": as_float(r, "E_q3"),
            "A_ref": as_float(r, "A_ref"),
            "A_sparse_med": as_float(r, "A_sparse_median"),
            "A_err": as_float(r, "abs_A_error"),
            "phi_rms": as_float(r, "phi_rms_median"),
            "eta_sparse": as_float(r, "eta_sparse_median"),
            "valid_fraction": as_float(r, "valid_fraction", default=1.0),
        })

    for total in TOTAL_SECTIONS:
        if len(by_total[total]) != 27:
            raise RuntimeError(
                f"{rock_name}: expected 27 summary rows for total_sections={total}, "
                f"found {len(by_total[total])}"
            )

    return {"name": rock_name, "sub": sub_rows, "by_total": by_total}


def aggregate_rock(data):
    out = {}
    for total in TOTAL_SECTIONS:
        rows = data["by_total"][total]
        E = np.array([r["E_med"] for r in rows], dtype=float)
        Aerr = np.array([r["A_err"] for r in rows], dtype=float)
        pos_iqr = np.array([r["E_q3"] - r["E_q1"] for r in rows], dtype=float)
        phi_rms = np.array([r["phi_rms"] for r in rows], dtype=float)
        valid = np.array([r["valid_fraction"] for r in rows], dtype=float)

        out[total] = {
            "E": qsummary(E),
            "Aerr": qsummary(Aerr),
            "position_iqr": qsummary(pos_iqr),
            "phi_rms": qsummary(phi_rms),
            "pct_E_le_10": float(100.0 * np.mean(E <= 0.10)),
            "pct_E_le_15": float(100.0 * np.mean(E <= 0.15)),
            "pct_E_le_20": float(100.0 * np.mean(E <= 0.20)),
            "valid_fraction_median": float(np.nanmedian(valid)),
            "valid_fraction_min": float(np.nanmin(valid)),
        }
    return out


def representative_subvolume(data):
    """Choose the subvolume whose porosity is closest to the spatial median."""
    phi = np.array([r["phi"] for r in data["sub"]], dtype=float)
    med = np.median(phi)
    idx = int(np.argmin(np.abs(phi - med)))
    return data["sub"][idx]


def load_representative_slice(raw_path, subrow):
    require_file(raw_path)
    expected = int(np.prod(RAW_SHAPE) * np.dtype(RAW_DTYPE).itemsize)
    actual = raw_path.stat().st_size
    if actual != expected:
        raise RuntimeError(
            f"Unexpected RAW size for {raw_path.name}: {actual:,} bytes; "
            f"expected {expected:,}."
        )

    c0, c1, c2 = subrow["center"]
    h = SUBVOL_N // 2
    s1 = slice(c1 - h, c1 + h)
    s2 = slice(c2 - h, c2 + h)

    raw = np.memmap(raw_path, dtype=RAW_DTYPE, mode="r", shape=RAW_SHAPE, order="C")
    img = np.asarray(raw[c0, s1, s2], dtype=np.uint8).copy()
    del raw
    if img.shape != (SUBVOL_N, SUBVOL_N):
        raise RuntimeError(f"Representative slice has unexpected shape: {img.shape}")
    return img


def add_scale_bar(ax, image_size, length_um=500.0):
    length_px = length_um / VOXEL_SIZE_UM
    x0 = image_size * 0.07
    y0 = image_size * 0.91
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", linewidth=3.2,
            solid_capstyle="butt")
    ax.text(
        x0 + 0.5 * length_px,
        y0 - image_size * 0.035,
        f"{int(length_um)} µm",
        color="white",
        ha="center", va="top",
        fontsize=7.8, fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.12", fc="black", ec="none", alpha=0.55),
    )


def plot_median_iqr(ax, agg, style, label, metric, scale=1.0):
    x = np.array(TOTAL_SECTIONS, dtype=float)
    med = np.array([agg[n][metric]["median"] for n in TOTAL_SECTIONS]) * scale
    q1 = np.array([agg[n][metric]["q1"] for n in TOTAL_SECTIONS]) * scale
    q3 = np.array([agg[n][metric]["q3"] for n in TOTAL_SECTIONS]) * scale
    yerr = np.vstack([med - q1, q3 - med])
    ax.errorbar(
        x, med, yerr=yerr,
        capsize=3.0, capthick=1.0,
        markersize=5.5,
        label=label,
        **style,
    )
    return med, q1, q3


# =============================================================================
# REPORT / CSV
# =============================================================================

def write_summary_csv(bent_agg, edb_agg):
    fields = [
        "rock", "total_sections",
        "E_median", "E_q1", "E_q3",
        "Aerr_median", "Aerr_q1", "Aerr_q3",
        "position_IQR_median", "position_IQR_q1", "position_IQR_q3",
        "phi_rms_median",
        "pct_E_le_10", "pct_E_le_15", "pct_E_le_20",
        "valid_fraction_median", "valid_fraction_min",
    ]
    rows = []
    for rock, agg in [("Bentheimer", bent_agg), ("Edwards Brown", edb_agg)]:
        for total in TOTAL_SECTIONS:
            a = agg[total]
            rows.append({
                "rock": rock,
                "total_sections": total,
                "E_median": a["E"]["median"],
                "E_q1": a["E"]["q1"],
                "E_q3": a["E"]["q3"],
                "Aerr_median": a["Aerr"]["median"],
                "Aerr_q1": a["Aerr"]["q1"],
                "Aerr_q3": a["Aerr"]["q3"],
                "position_IQR_median": a["position_iqr"]["median"],
                "position_IQR_q1": a["position_iqr"]["q1"],
                "position_IQR_q3": a["position_iqr"]["q3"],
                "phi_rms_median": a["phi_rms"]["median"],
                "pct_E_le_10": a["pct_E_le_10"],
                "pct_E_le_15": a["pct_E_le_15"],
                "pct_E_le_20": a["pct_E_le_20"],
                "valid_fraction_median": a["valid_fraction_median"],
                "valid_fraction_min": a["valid_fraction_min"],
            })

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_report(bent, edb, bent_agg, edb_agg, bent_rep, edb_rep):
    bphi = qsummary([r["phi"] for r in bent["sub"]])
    ephi = qsummary([r["phi"] for r in edb["sub"]])
    bA = qsummary([r["A_ref"] for r in bent["sub"]])
    eA = qsummary([r["A_ref"] for r in edb["sub"]])
    beta = qsummary([r["eta_ref"] for r in bent["sub"]])
    eeta = qsummary([r["eta_ref"] for r in edb["sub"]])

    b3, b21 = bent_agg[3]["E"]["median"], bent_agg[21]["E"]["median"]
    e3, e21 = edb_agg[3]["E"]["median"], edb_agg[21]["E"]["median"]
    bred = 100.0 * (1.0 - b21 / b3)
    ered = 100.0 * (1.0 - e21 / e3)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("FIGURE 6 — REAL-ROCK COMPARISON\n")
        f.write("=" * 78 + "\n\n")
        f.write("Inputs are FINAL validation CSVs; no heavy computation is repeated.\n")
        f.write("Both rocks use 27 non-overlapping 512^3 spatial regions and the same\n")
        f.write("frozen sparse-section protocol (3, 9, 15, 21 total sections).\n\n")

        f.write("Bentheimer\n")
        f.write(f"  porosity median/range: {100*bphi['median']:.3f}% / "
                f"[{100*bphi['min']:.3f}, {100*bphi['max']:.3f}]%\n")
        f.write(f"  A_ref median/range:    {bA['median']:.4f} / "
                f"[{bA['min']:.4f}, {bA['max']:.4f}]\n")
        f.write(f"  seed-level eta median: {100*beta['median']:.3f}%\n")
        f.write(f"  E_Q 3 -> 21 sections:  {100*b3:.3f}% -> {100*b21:.3f}% "
                f"({bred:.1f}% reduction)\n")
        f.write(f"  representative slice:  {bent_rep['id']} center={bent_rep['center']}\n\n")

        f.write("Edwards Brown (EdB-1)\n")
        f.write(f"  porosity median/range: {100*ephi['median']:.3f}% / "
                f"[{100*ephi['min']:.3f}, {100*ephi['max']:.3f}]%\n")
        f.write(f"  A_ref median/range:    {eA['median']:.4f} / "
                f"[{eA['min']:.4f}, {eA['max']:.4f}]\n")
        f.write(f"  seed-level eta median: {100*eeta['median']:.3f}%\n")
        f.write(f"  E_Q 3 -> 21 sections:  {100*e3:.3f}% -> {100*e21:.3f}% "
                f"({ered:.1f}% reduction)\n")
        f.write(f"  representative slice:  {edb_rep['id']} center={edb_rep['center']}\n\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 78 + "\n")
        f.write("Spatial replication reduces tensor reconstruction error and position\n")
        f.write("sensitivity in both lithologies. The carbonate remains more difficult,\n")
        f.write("consistent with its much stronger spatial heterogeneity and higher\n")
        f.write("reference-model residual.\n")
        f.write("The 1.706% p95 reference number is a Bentheimer calibration of the frozen\n")
        f.write("50k x 5-seed protocol; it is not presented as an independent EdB-1\n")
        f.write("uncertainty estimate.\n")


# =============================================================================
# FIGURE
# =============================================================================

def make_figure(bent, edb, bent_agg, edb_agg, bent_img, edb_img,
                bent_rep, edb_rep):
    fig = plt.figure(figsize=(10.8, 7.9))
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.08, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.42, wspace=0.36,
    )

    # ------------------------------------------------------------------ (a)
    ax_container = fig.add_subplot(gs[0, 0])
    ax_container.axis("off")
    ax_container.set_title("Representative real microstructures", pad=8)
    panel_label(ax_container, "(a)", x=-0.08, y=1.04)

    # Two inset axes inside panel (a).
    ax_b = ax_container.inset_axes([0.00, 0.08, 0.48, 0.83])
    ax_e = ax_container.inset_axes([0.52, 0.08, 0.48, 0.83])

    for ax, img, title, rep in [
        (ax_b, bent_img, "Bentheimer", bent_rep),
        (ax_e, edb_img, "Edwards Brown", edb_rep),
    ]:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9.5, pad=4)
        ax.text(
            0.03, 0.04,
            rf"$\phi={100*rep['phi']:.1f}\%$" + "\n" + rf"$A_{{ref}}={rep['A_ref']:.2f}$",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=7.7, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.55),
        )
        add_scale_bar(ax, SUBVOL_N, length_um=500.0)

    ax_container.text(
        0.5, -0.02,
        r"Representative $a_1$-$a_2$ sections from median-porosity spatial regions",
        transform=ax_container.transAxes,
        ha="center", va="top",
        fontsize=8.0, fontweight="bold",
    )

    # ------------------------------------------------------------------ (b)
    ax = fig.add_subplot(gs[0, 1])
    bphi = np.array([r["phi"] for r in bent["sub"]])
    bA = np.array([r["A_ref"] for r in bent["sub"]])
    ephi = np.array([r["phi"] for r in edb["sub"]])
    eA = np.array([r["A_ref"] for r in edb["sub"]])
    ax.scatter(100*bphi, bA, s=31, facecolors="white", edgecolors="black",
               linewidths=1.0, label="Bentheimer", zorder=3)
    ax.scatter(100*ephi, eA, s=31, color="0.45", marker="s",
               label="Edwards Brown", zorder=2)
    ax.set_xlabel("Subvolume porosity (%)")
    ax.set_ylabel(r"Reference anisotropy $A_{ref}$")
    ax.set_title("Spatial microstructural heterogeneity")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    panel_label(ax, "(b)")

    # ------------------------------------------------------------------ (c)
    ax = fig.add_subplot(gs[0, 2])
    plot_median_iqr(ax, bent_agg, BENT_STYLE, "Bentheimer", "E", scale=100.0)
    plot_median_iqr(ax, edb_agg, EDB_STYLE, "Edwards Brown", "E", scale=100.0)
    ax.axhline(
        100*REF_PROTOCOL_P95_BENTHEIMER,
        color="0.7", linewidth=1.0, linestyle=":", zorder=0,
    )
    ax.text(
        21.3, 100*REF_PROTOCOL_P95_BENTHEIMER,
        "1.71% ref. protocol\ncalibration (Bent.)",
        ha="right", va="bottom", fontsize=6.9, color="0.4", fontweight="bold",
    )
    ax.set_xticks(TOTAL_SECTIONS)
    ax.set_xlabel("Total number of sections")
    ax.set_ylabel(r"Spatial median tensor error $E_Q$ (%)")
    ax.set_title("3D tensor recovery improves with replication")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(bottom=0)
    clean_axis(ax)
    panel_label(ax, "(c)")

    # ------------------------------------------------------------------ (d)
    ax = fig.add_subplot(gs[1, 0])
    plot_median_iqr(
        ax, bent_agg, BENT_STYLE, "Bentheimer", "position_iqr", scale=100.0
    )
    plot_median_iqr(
        ax, edb_agg, EDB_STYLE, "Edwards Brown", "position_iqr", scale=100.0
    )
    ax.set_xticks(TOTAL_SECTIONS)
    ax.set_xlabel("Total number of sections")
    ax.set_ylabel(r"Within-region $E_Q$ IQR (pp)")
    ax.set_title("Position sensitivity decreases")
    ax.set_ylim(bottom=0)
    clean_axis(ax)
    panel_label(ax, "(d)")

    # ------------------------------------------------------------------ (e)
    ax = fig.add_subplot(gs[1, 1])
    plot_median_iqr(ax, bent_agg, BENT_STYLE, "Bentheimer", "Aerr", scale=1.0)
    plot_median_iqr(ax, edb_agg, EDB_STYLE, "Edwards Brown", "Aerr", scale=1.0)
    ax.set_xticks(TOTAL_SECTIONS)
    ax.set_xlabel("Total number of sections")
    ax.set_ylabel(r"Median $|A_{sparse}-A_{ref}|$")
    ax.set_title("Anisotropy magnitude stabilizes")
    ax.set_ylim(bottom=0)
    clean_axis(ax)
    panel_label(ax, "(e)")

    # ------------------------------------------------------------------ (f)
    ax = fig.add_subplot(gs[1, 2])
    x = np.array(TOTAL_SECTIONS, dtype=float)
    b10 = np.array([bent_agg[n]["pct_E_le_10"] for n in TOTAL_SECTIONS])
    e10 = np.array([edb_agg[n]["pct_E_le_10"] for n in TOTAL_SECTIONS])
    b20 = np.array([bent_agg[n]["pct_E_le_20"] for n in TOTAL_SECTIONS])
    e20 = np.array([edb_agg[n]["pct_E_le_20"] for n in TOTAL_SECTIONS])

    ax.plot(x, b10, color="black", marker="o", linestyle="-", linewidth=1.8,
            label=r"Bent., $E_Q\leq10\%$")
    ax.plot(x, b20, color="black", marker="^", linestyle=":", linewidth=1.6,
            label=r"Bent., $E_Q\leq20\%$")
    ax.plot(x, e10, color="0.45", marker="s", linestyle="--", linewidth=1.8,
            label=r"EdB-1, $E_Q\leq10\%$")
    ax.plot(x, e20, color="0.45", marker="v", linestyle="-.", linewidth=1.6,
            label=r"EdB-1, $E_Q\leq20\%$")
    ax.set_xticks(TOTAL_SECTIONS)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Total number of sections")
    ax.set_ylabel("Subvolumes meeting threshold (%)")
    ax.set_title("Accuracy depends on lithology")
    ax.legend(frameon=False, loc="lower right", fontsize=7.1)
    clean_axis(ax)
    panel_label(ax, "(f)")

    # Global caption-like footnote.
    fig.text(
        0.5, 0.012,
        r"27 non-overlapping $512^3$ regions per rock; identical frozen section protocol. "
        r"Error bars show the spatial IQR of region-level medians. "
        r"Rank-deficient EdB-1 draws are excluded from $E_Q$ percentiles and tracked separately.",
        ha="center", va="bottom",
        fontsize=7.8, fontweight="bold",
    )

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.085, top=0.955)
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("FIGURE 6 — REAL-ROCK COMPARISON")
    print("=" * 78)
    print("This script reads final CSVs only; it does not rerun the heavy analysis.\n")

    for p in [BENT_SUBVOL, BENT_SUMMARY, EDB_SUBVOL, EDB_SUMMARY,
              BENTHEIMER_RAW, EDB1_RAW]:
        print(f"checking: {p}")
        require_file(p)

    print("\nLoading final Bentheimer results...")
    bent = parse_final_data(BENT_SUBVOL, BENT_SUMMARY, "Bentheimer")
    print("Loading final Edwards Brown results...")
    edb = parse_final_data(EDB_SUBVOL, EDB_SUMMARY, "Edwards Brown")

    bent_agg = aggregate_rock(bent)
    edb_agg = aggregate_rock(edb)

    bent_rep = representative_subvolume(bent)
    edb_rep = representative_subvolume(edb)

    print(
        f"Bentheimer representative region: {bent_rep['id']}  "
        f"phi={100*bent_rep['phi']:.3f}%  Aref={bent_rep['A_ref']:.4f}"
    )
    print(
        f"EdB-1 representative region:      {edb_rep['id']}  "
        f"phi={100*edb_rep['phi']:.3f}%  Aref={edb_rep['A_ref']:.4f}"
    )

    print("\nReading representative RAW slices via memmap...")
    bent_img = load_representative_slice(BENTHEIMER_RAW, bent_rep)
    edb_img = load_representative_slice(EDB1_RAW, edb_rep)

    print("Computing comparison summaries...")
    write_summary_csv(bent_agg, edb_agg)
    write_report(bent, edb, bent_agg, edb_agg, bent_rep, edb_rep)

    print("Generating Figure 6...")
    make_figure(
        bent, edb, bent_agg, edb_agg,
        bent_img, edb_img, bent_rep, edb_rep,
    )

    print("\n" + "=" * 78)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 78)
    for rock, agg in [("Bentheimer", bent_agg), ("Edwards Brown", edb_agg)]:
        print(rock)
        for n in TOTAL_SECTIONS:
            a = agg[n]
            print(
                f"  total={n:2d}  E_Q={100*a['E']['median']:6.2f}% "
                f"IQR=[{100*a['E']['q1']:.2f},{100*a['E']['q3']:.2f}]%  "
                f"|dA|={a['Aerr']['median']:.4f}  "
                f"position-IQR={100*a['position_iqr']['median']:.2f} pp  "
                f"E<=10/20%={a['pct_E_le_10']:.0f}/{a['pct_E_le_20']:.0f}%"
            )
        print()

    print("FILES GENERATED")
    print("=" * 78)
    for p in [PNG_PATH, SVG_PATH, CSV_PATH, REPORT_PATH]:
        print(p)


if __name__ == "__main__":
    main()