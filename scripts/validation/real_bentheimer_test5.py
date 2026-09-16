#!/usr/bin/env python3
"""
Bentheimer real-rock validation: test 5 (FINAL calibration test)
================================================================

Consensus-size convergence using ONLY the cached 50,000-point reference tensors
from Test 4.  This script does NOT open the 15 GB RAW file and does NOT repeat
any 3D directional-correlation calculation.

Purpose
-------
Test how the Monte Carlo uncertainty of the 3D reference tensor decreases when
we average multiple independent 50k-point seeds:

    Q_ref^(m) = consensus(Q_1, ..., Q_m)

for

    m = 1, 2, 3, 5, 7, 10 seeds.

The five Test-4 subvolumes are used:

    G11, G05, G00, G18, G04.

The consensus definition is intentionally identical to Test 4:

    Euclidean mean in Q-space -> projection back to SPD.

There is still no analytic ground truth for a real rock.  Therefore this script
reports THREE complementary diagnostics:

1) Bootstrap-to-full-consensus deviation

       E_full = ||Q~_boot - Q~_10||_F / ||Q~_10||_F

   where Q_10 is the consensus of all ten available 50k seeds and Q~ denotes
   determinant normalization.  This is a finite-sample stability diagnostic;
   it is NOT called an exact error to truth.

2) Independent bootstrap-pair disagreement

       D_pair = symmetric distance(Q_boot_A, Q_boot_B)

   where A and B are independently resampled m-seed consensuses.

3) Single-consensus uncertainty proxy

       U_pair = D_pair / sqrt(2)

   Under an approximately independent, equal-variance small-error model,
   D_pair/sqrt(2) estimates the scale of one consensus around the latent mean.
   This approximation is explicitly labeled as a proxy, not an exact theorem
   for the nonlinear tensor metric.

The script also tracks anisotropy-ratio stability and geometric correlation
length stability.

Practical decision rule
-----------------------
A recommended seed count is selected as the SMALLEST tested m satisfying BOTH:

    pooled p95(U_pair) <= TARGET_P95_POOLED
    worst-subvolume p95(U_pair) <= TARGET_P95_WORST

If no tested m satisfies both thresholds, the script recommends m=10 (the
largest supported by the existing cache) and reports its measured uncertainty.
No additional calibration test is required: the remaining step is then a final
production run using the chosen consensus size.

Required input
--------------
Directory from Test 4:

    bentheimer_test4_cache/
        G11_seed00.json ... G11_seed09.json
        G05_seed00.json ... G05_seed09.json
        G00_seed00.json ... G00_seed09.json
        G18_seed00.json ... G18_seed09.json
        G04_seed00.json ... G04_seed09.json

Each JSON must contain point_results["50000"]["Q"].

Outputs
-------
    bentheimer_test5_consensus_convergence.png
    bentheimer_test5_consensus_convergence.svg
    bentheimer_test5_report.txt
    bentheimer_test5_summary.csv
    bentheimer_test5_subvolume_summary.csv
    bentheimer_test5_bootstrap.csv
    bentheimer_test5_exact_subsets.csv

Dependencies
------------
    python3 -m pip install numpy matplotlib

Run
---
    python3 real_bentheimer_test5_consensus_convergence.py
"""

from pathlib import Path
from itertools import combinations
import csv
import json
import math
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# USER SETTINGS
# =============================================================================

# Test-4 cache folder.  The script first tries the folder beside this script,
# then the current working directory.
CACHE_DIR_NAME = "bentheimer_test4_cache"

SUBVOLUMES = ("G11", "G05", "G00", "G18", "G04")
N_AVAILABLE_SEEDS = 10
N_POINTS = 50_000

CONSENSUS_SIZES = (1, 2, 3, 5, 7, 10)

# Bootstrap replicates per subvolume and consensus size.
# 3000 is already stable enough for p95 diagnostics and runs quickly because
# only cached 3x3 tensors are used.
N_BOOTSTRAP = 3000
SEED = 20260908

# Practical decision thresholds for the pair-based single-consensus uncertainty
# proxy U_pair = D_pair/sqrt(2).
TARGET_P95_POOLED = 0.020   # 2%
TARGET_P95_WORST = 0.030    # 3%

DPI = 800


# =============================================================================
# OUTPUTS / STYLE
# =============================================================================

OUTDIR = Path(__file__).resolve().parent
PNG_PATH = OUTDIR / "bentheimer_test5_consensus_convergence.png"
SVG_PATH = OUTDIR / "bentheimer_test5_consensus_convergence.svg"
REPORT_PATH = OUTDIR / "bentheimer_test5_report.txt"
SUMMARY_CSV_PATH = OUTDIR / "bentheimer_test5_summary.csv"
SUBVOL_SUMMARY_CSV_PATH = OUTDIR / "bentheimer_test5_subvolume_summary.csv"
BOOTSTRAP_CSV_PATH = OUTDIR / "bentheimer_test5_bootstrap.csv"
EXACT_SUBSETS_CSV_PATH = OUTDIR / "bentheimer_test5_exact_subsets.csv"

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
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# TENSOR UTILITIES -- kept consistent with Test 4
# =============================================================================

def project_spd(Q):
    Q = np.asarray(Q, dtype=float)
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    floor = max(1e-12, 1e-8 * np.max(np.abs(w)))
    w = np.maximum(w, floor)
    return V @ np.diag(w) @ V.T


def normalize_det_spd(Q):
    Q = project_spd(Q)
    return Q / np.linalg.det(Q) ** (1.0 / Q.shape[0])


def principal_lengths_and_axes(Q):
    w, V = np.linalg.eigh(project_spd(Q))
    w = np.maximum(w, 1e-14)
    ell = 1.0 / np.sqrt(w)
    order = np.argsort(ell)[::-1]
    return ell[order], V[:, order]


def anisotropy_ratio(Q):
    ell, _ = principal_lengths_and_axes(Q)
    return float(ell[0] / ell[-1])


def geometric_correlation_length(Q):
    Q = project_spd(Q)
    return float(np.linalg.det(Q) ** (-1.0 / 6.0))


def tensor_shape_error(Q_est, Q_ref):
    Qe = normalize_det_spd(Q_est)
    Qr = normalize_det_spd(Q_ref)
    return float(
        np.linalg.norm(Qe - Qr, ord="fro") /
        np.linalg.norm(Qr, ord="fro")
    )


def tensor_pair_error(Q1, Q2):
    A = normalize_det_spd(Q1)
    B = normalize_det_spd(Q2)
    den = 0.5 * (
        np.linalg.norm(A, ord="fro") + np.linalg.norm(B, ord="fro")
    )
    return float(np.linalg.norm(A - B, ord="fro") / den)


def consensus_Q(Q_list):
    """Euclidean mean in Q-space, projected back to SPD -- same as Test 4."""
    return project_spd(np.mean(np.stack(Q_list, axis=0), axis=0))


# =============================================================================
# BASIC STATISTICS
# =============================================================================

def percentile_summary(values):
    x = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(x)),
        "q1": float(np.percentile(x, 25)),
        "q3": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def find_cache_dir():
    candidates = [
        OUTDIR / CACHE_DIR_NAME,
        Path.cwd() / CACHE_DIR_NAME,
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p.resolve()

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Test-4 cache directory not found.\n"
        "Expected a folder named 'bentheimer_test4_cache'.\n"
        f"Tried:\n{tried}\n\n"
        "Place this script in the same directory that contains the Test-4 "
        "cache folder, or edit CACHE_DIR_NAME at the top of the script."
    )


# =============================================================================
# LOAD TEST-4 CACHE
# =============================================================================

def load_cached_tensors(cache_dir):
    data = {}

    print("=" * 78)
    print("LOADING TEST-4 CACHE")
    print("=" * 78)
    print(f"Cache directory: {cache_dir}")
    print(f"Required point count: {N_POINTS:,}")
    print(f"Required seeds/subvolume: {N_AVAILABLE_SEEDS}")
    print()

    for sid in SUBVOLUMES:
        rows = []
        phi_values = []

        for seed_idx in range(N_AVAILABLE_SEEDS):
            path = cache_dir / f"{sid}_seed{seed_idx:02d}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing cache file:\n{path}\n\n"
                    "Test 5 requires all ten Test-4 seed caches for each "
                    "selected subvolume."
                )

            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            sid_file = obj.get("subvolume", {}).get("id")
            if sid_file != sid:
                raise RuntimeError(
                    f"Cache identity mismatch in {path.name}: "
                    f"expected {sid}, found {sid_file}."
                )

            pr = obj.get("point_results", {}).get(str(N_POINTS))
            if pr is None:
                raise RuntimeError(
                    f"{path.name} does not contain point_results['{N_POINTS}']."
                )

            Q = np.asarray(pr.get("Q"), dtype=float)
            if Q.shape != (3, 3) or not np.all(np.isfinite(Q)):
                raise RuntimeError(f"Invalid 3x3 Q tensor in {path.name}.")

            rows.append(project_spd(Q))
            phi_values.append(float(obj.get("phi_sub", np.nan)))

        Qs = np.stack(rows, axis=0)
        phi = float(np.nanmedian(phi_values))
        Q_full = consensus_Q(list(Qs))

        data[sid] = {
            "Qs": Qs,
            "phi": phi,
            "Q_full": Q_full,
            "A_full": anisotropy_ratio(Q_full),
            "ell_geom_full": geometric_correlation_length(Q_full),
        }

        print(
            f"{sid}: loaded {len(Qs)} seeds  "
            f"phi={100*phi:.3f}%  "
            f"A_10={data[sid]['A_full']:.4f}  "
            f"ell_geom_10={data[sid]['ell_geom_full']:.3f} px"
        )

    print()
    return data


# =============================================================================
# EXACT UNIQUE-SUBSET SENSITIVITY
# =============================================================================

def exact_subset_analysis(data):
    """
    Enumerate all unique subsets of the ten available seeds.

    These subset-to-full-consensus deviations are finite-sample sensitivity
    diagnostics.  Because the subset contributes to the 10-seed target, they
    are NOT treated as independent uncertainty estimates.
    """
    rows = []

    print("=" * 78)
    print("EXACT UNIQUE-SUBSET SENSITIVITY")
    print("=" * 78)

    for sid in SUBVOLUMES:
        Qs = data[sid]["Qs"]
        Q_full = data[sid]["Q_full"]
        A_full = data[sid]["A_full"]
        ellg_full = data[sid]["ell_geom_full"]

        for m in CONSENSUS_SIZES:
            combos = list(combinations(range(N_AVAILABLE_SEEDS), m))

            # m=10 has exactly one combination and therefore zero deviation
            # from the full 10-seed consensus by construction.
            for combo in combos:
                Qc = consensus_Q([Qs[i] for i in combo])
                rows.append({
                    "subvolume": sid,
                    "m": int(m),
                    "indices": " ".join(str(i) for i in combo),
                    "E_to_full": tensor_shape_error(Qc, Q_full),
                    "abs_dA": abs(anisotropy_ratio(Qc) - A_full),
                    "rel_ell_geom_error": abs(
                        geometric_correlation_length(Qc) - ellg_full
                    ) / ellg_full,
                })

        print(f"{sid}: exact subsets completed")

    print()
    return rows


# =============================================================================
# BOOTSTRAP CONSENSUS CONVERGENCE
# =============================================================================

def bootstrap_analysis(data):
    rows = []
    total_blocks = len(SUBVOLUMES) * len(CONSENSUS_SIZES)
    block = 0

    print("=" * 78)
    print("BOOTSTRAP CONSENSUS CONVERGENCE")
    print("=" * 78)
    print(f"Replicates / subvolume / m: {N_BOOTSTRAP:,}")
    print("Sampling: with replacement from the ten independent 50k seed tensors")
    print()

    t0 = time.time()

    for sidx, sid in enumerate(SUBVOLUMES):
        Qs = data[sid]["Qs"]
        Q_full = data[sid]["Q_full"]
        A_full = data[sid]["A_full"]
        ellg_full = data[sid]["ell_geom_full"]

        for m in CONSENSUS_SIZES:
            block += 1
            rng = np.random.default_rng(
                SEED + 100_000 * (sidx + 1) + 1000 * int(m)
            )

            E_full_values = []
            pair_values = []
            pair_proxy_values = []
            dA_values = []
            dL_values = []

            for rep in range(N_BOOTSTRAP):
                idx_a = rng.integers(0, N_AVAILABLE_SEEDS, size=m)
                idx_b = rng.integers(0, N_AVAILABLE_SEEDS, size=m)

                Qa = consensus_Q([Qs[i] for i in idx_a])
                Qb = consensus_Q([Qs[i] for i in idx_b])

                E_full = tensor_shape_error(Qa, Q_full)
                D_pair = tensor_pair_error(Qa, Qb)
                U_pair = D_pair / math.sqrt(2.0)
                dA = abs(anisotropy_ratio(Qa) - A_full)
                dL = abs(
                    geometric_correlation_length(Qa) - ellg_full
                ) / ellg_full

                E_full_values.append(E_full)
                pair_values.append(D_pair)
                pair_proxy_values.append(U_pair)
                dA_values.append(dA)
                dL_values.append(dL)

                rows.append({
                    "subvolume": sid,
                    "replicate": int(rep + 1),
                    "m": int(m),
                    "E_to_full": float(E_full),
                    "pair_disagreement": float(D_pair),
                    "single_consensus_proxy": float(U_pair),
                    "abs_dA": float(dA),
                    "rel_ell_geom_error": float(dL),
                })

            E = percentile_summary(E_full_values)
            D = percentile_summary(pair_values)
            U = percentile_summary(pair_proxy_values)
            dA_s = percentile_summary(dA_values)
            dL_s = percentile_summary(dL_values)

            elapsed = time.time() - t0
            print(
                f"[{block:02d}/{total_blocks:02d}] {sid} m={m:2d}  "
                f"Efull med={100*E['median']:.2f}% p95={100*E['p95']:.2f}%  "
                f"Upair med={100*U['median']:.2f}% p95={100*U['p95']:.2f}%  "
                f"|dA| med={dA_s['median']:.4f}  "
                f"scale med={100*dL_s['median']:.2f}%  "
                f"elapsed={elapsed:.1f}s"
            )

    print()
    return rows


# =============================================================================
# SUMMARIES / RECOMMENDATION
# =============================================================================

def summarize_bootstrap(rows):
    subvol_rows = []
    global_rows = []

    # Per-subvolume summaries.
    for sid in SUBVOLUMES:
        for m in CONSENSUS_SIZES:
            rr = [r for r in rows if r["subvolume"] == sid and r["m"] == m]

            E = percentile_summary([r["E_to_full"] for r in rr])
            D = percentile_summary([r["pair_disagreement"] for r in rr])
            U = percentile_summary([r["single_consensus_proxy"] for r in rr])
            dA = percentile_summary([r["abs_dA"] for r in rr])
            dL = percentile_summary([r["rel_ell_geom_error"] for r in rr])

            subvol_rows.append({
                "subvolume": sid,
                "m": int(m),
                "E_median": E["median"],
                "E_q1": E["q1"],
                "E_q3": E["q3"],
                "E_p95": E["p95"],
                "pair_median": D["median"],
                "pair_p95": D["p95"],
                "proxy_median": U["median"],
                "proxy_q1": U["q1"],
                "proxy_q3": U["q3"],
                "proxy_p95": U["p95"],
                "abs_dA_median": dA["median"],
                "abs_dA_p95": dA["p95"],
                "rel_ell_geom_median": dL["median"],
                "rel_ell_geom_p95": dL["p95"],
            })

    # Global pooled summaries.  Bootstrap replicates are descriptive resamples,
    # not independent physical rock replicates; the report states this clearly.
    for m in CONSENSUS_SIZES:
        rr = [r for r in rows if r["m"] == m]

        E = percentile_summary([r["E_to_full"] for r in rr])
        D = percentile_summary([r["pair_disagreement"] for r in rr])
        U = percentile_summary([r["single_consensus_proxy"] for r in rr])
        dA = percentile_summary([r["abs_dA"] for r in rr])
        dL = percentile_summary([r["rel_ell_geom_error"] for r in rr])

        sv_proxy_p95 = [
            r["proxy_p95"] for r in subvol_rows if r["m"] == m
        ]

        global_rows.append({
            "m": int(m),
            "E_median": E["median"],
            "E_q1": E["q1"],
            "E_q3": E["q3"],
            "E_p95": E["p95"],
            "pair_median": D["median"],
            "pair_p95": D["p95"],
            "proxy_median": U["median"],
            "proxy_q1": U["q1"],
            "proxy_q3": U["q3"],
            "proxy_p95": U["p95"],
            "worst_subvolume_proxy_p95": float(max(sv_proxy_p95)),
            "abs_dA_median": dA["median"],
            "abs_dA_p95": dA["p95"],
            "rel_ell_geom_median": dL["median"],
            "rel_ell_geom_p95": dL["p95"],
        })

    recommended = None
    for r in global_rows:
        if (
            r["proxy_p95"] <= TARGET_P95_POOLED
            and r["worst_subvolume_proxy_p95"] <= TARGET_P95_WORST
        ):
            recommended = int(r["m"])
            break

    if recommended is None:
        recommended = int(max(CONSENSUS_SIZES))
        threshold_met = False
    else:
        threshold_met = True

    return subvol_rows, global_rows, recommended, threshold_met


# =============================================================================
# CSV / REPORT
# =============================================================================

def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_report(data, global_rows, subvol_rows, recommended, threshold_met, cache_dir):
    rec = next(r for r in global_rows if r["m"] == recommended)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Bentheimer real-rock reference consensus convergence: Test 5\n")
        f.write("=" * 78 + "\n\n")
        f.write("This analysis uses ONLY cached Test-4 50k-point tensors.\n")
        f.write(f"Cache directory: {cache_dir}\n")
        f.write(f"Subvolumes: {', '.join(SUBVOLUMES)}\n")
        f.write(f"Independent cached seeds/subvolume: {N_AVAILABLE_SEEDS}\n")
        f.write(f"Points/direction per cached seed: {N_POINTS}\n")
        f.write(f"Bootstrap replicates/subvolume/m: {N_BOOTSTRAP}\n")
        f.write(f"Consensus sizes: {CONSENSUS_SIZES}\n\n")

        f.write("Terminology\n")
        f.write("-----------\n")
        f.write(
            "Q_full is the consensus of all ten available 50k-point seeds and is "
            "a practical finite-sample anchor, not analytic ground truth.\n"
        )
        f.write(
            "U_pair = D_pair/sqrt(2) is a single-consensus uncertainty proxy "
            "under an approximately independent equal-variance small-error model.\n"
        )
        f.write(
            "Bootstrap replicates are resampling diagnostics and are not independent "
            "physical rock realizations.\n\n"
        )

        f.write("Ten-seed consensus properties\n")
        f.write("-----------------------------\n")
        for sid in SUBVOLUMES:
            d = data[sid]
            f.write(
                f"{sid}: phi={100*d['phi']:.3f}%  "
                f"A_10={d['A_full']:.6f}  "
                f"ell_geom_10={d['ell_geom_full']:.6f} px\n"
            )
        f.write("\n")

        f.write("Global bootstrap convergence\n")
        f.write("----------------------------\n")
        for r in global_rows:
            f.write(
                f"m={r['m']:2d}  "
                f"E_full med={100*r['E_median']:.3f}% p95={100*r['E_p95']:.3f}%  "
                f"U_pair med={100*r['proxy_median']:.3f}% "
                f"p95={100*r['proxy_p95']:.3f}%  "
                f"worst-sv p95={100*r['worst_subvolume_proxy_p95']:.3f}%  "
                f"|dA| med={r['abs_dA_median']:.5f}  "
                f"scale med={100*r['rel_ell_geom_median']:.3f}%\n"
            )

        f.write("\nPractical decision\n")
        f.write("------------------\n")
        f.write(f"Pooled p95 target: <= {100*TARGET_P95_POOLED:.2f}%\n")
        f.write(f"Worst-subvolume p95 target: <= {100*TARGET_P95_WORST:.2f}%\n")
        f.write(f"Recommended consensus size: m={recommended}\n")
        f.write(f"Threshold pair satisfied: {threshold_met}\n")
        f.write(
            f"At m={recommended}: pooled U_pair p95="
            f"{100*rec['proxy_p95']:.3f}%, worst-subvolume U_pair p95="
            f"{100*rec['worst_subvolume_proxy_p95']:.3f}%.\n"
        )
        if not threshold_met:
            f.write(
                "The requested thresholds were not both reached.  m=10 is still "
                "recommended as the maximum supported by the existing cache; its "
                "measured uncertainty should be carried explicitly into the final "
                "validation instead of launching another calibration test.\n"
            )
        else:
            f.write(
                "The selected m is the smallest tested consensus size satisfying "
                "both practical uncertainty thresholds.\n"
            )


# =============================================================================
# FIGURE
# =============================================================================

def panel_label(ax, label, x=0.01, y=0.99):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )


def build_figure(global_rows, subvol_rows, data, recommended, threshold_met):
    mvals = np.array([r["m"] for r in global_rows], dtype=int)

    fig = plt.figure(figsize=(10.8, 7.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)

    # ------------------------------------------------------------------ (a)
    ax = fig.add_subplot(gs[0, 0])
    med = np.array([100*r["proxy_median"] for r in global_rows])
    q1 = np.array([100*r["proxy_q1"] for r in global_rows])
    q3 = np.array([100*r["proxy_q3"] for r in global_rows])
    p95 = np.array([100*r["proxy_p95"] for r in global_rows])
    ax.fill_between(mvals, q1, q3, alpha=0.20, label="IQR")
    ax.plot(mvals, med, marker="o", label="Median")
    ax.plot(mvals, p95, marker="s", linestyle="--", label="95th percentile")
    ax.axhline(100*TARGET_P95_POOLED, color="black", linestyle=":", linewidth=1.1,
               label=f"Pooled target ({100*TARGET_P95_POOLED:.0f}%)")
    ax.set_xlabel("Independent 50k seeds in consensus, m")
    ax.set_ylabel(r"$U_{pair}=D_{pair}/\sqrt{2}$ (%)")
    ax.set_title("Consensus tensor uncertainty proxy")
    ax.set_xticks(mvals)
    ax.legend(frameon=False, fontsize=7.7)
    panel_label(ax, "(a)")

    # ------------------------------------------------------------------ (b)
    ax = fig.add_subplot(gs[0, 1])
    ef_med = np.array([100*r["E_median"] for r in global_rows])
    ef_q1 = np.array([100*r["E_q1"] for r in global_rows])
    ef_q3 = np.array([100*r["E_q3"] for r in global_rows])
    ef_p95 = np.array([100*r["E_p95"] for r in global_rows])
    ax.fill_between(mvals, ef_q1, ef_q3, alpha=0.20, label="IQR")
    ax.plot(mvals, ef_med, marker="o", label="Median")
    ax.plot(mvals, ef_p95, marker="s", linestyle="--", label="95th percentile")
    ax.set_xlabel("Independent 50k seeds in consensus, m")
    ax.set_ylabel(r"$E(Q_m,Q_{10})$ (%)")
    ax.set_title("Stability relative to 10-seed anchor")
    ax.set_xticks(mvals)
    ax.legend(frameon=False, fontsize=7.7)
    panel_label(ax, "(b)")

    # ------------------------------------------------------------------ (c)
    ax = fig.add_subplot(gs[0, 2])
    dA_med = np.array([r["abs_dA_median"] for r in global_rows])
    dA_p95 = np.array([r["abs_dA_p95"] for r in global_rows])
    ax.plot(mvals, dA_med, marker="o", label="Median")
    ax.plot(mvals, dA_p95, marker="s", linestyle="--", label="95th percentile")
    ax.set_xlabel("Independent 50k seeds in consensus, m")
    ax.set_ylabel(r"$|A_m-A_{10}|$")
    ax.set_title("Anisotropy-ratio stability")
    ax.set_xticks(mvals)
    ax.legend(frameon=False)
    panel_label(ax, "(c)")

    # ------------------------------------------------------------------ (d)
    ax = fig.add_subplot(gs[1, 0])
    dl_med = np.array([100*r["rel_ell_geom_median"] for r in global_rows])
    dl_p95 = np.array([100*r["rel_ell_geom_p95"] for r in global_rows])
    ax.plot(mvals, dl_med, marker="o", label="Median")
    ax.plot(mvals, dl_p95, marker="s", linestyle="--", label="95th percentile")
    ax.set_xlabel("Independent 50k seeds in consensus, m")
    ax.set_ylabel("Relative geometric-scale difference (%)")
    ax.set_title("Correlation-scale stability")
    ax.set_xticks(mvals)
    ax.legend(frameon=False)
    panel_label(ax, "(d)")

    # ------------------------------------------------------------------ (e)
    ax = fig.add_subplot(gs[1, 1])
    for sid in SUBVOLUMES:
        rr = sorted(
            [r for r in subvol_rows if r["subvolume"] == sid],
            key=lambda r: r["m"],
        )
        ax.plot(
            [r["m"] for r in rr],
            [100*r["proxy_p95"] for r in rr],
            marker="o",
            label=sid,
        )
    ax.axhline(100*TARGET_P95_WORST, color="black", linestyle=":", linewidth=1.1,
               label=f"Per-volume target ({100*TARGET_P95_WORST:.0f}%)")
    ax.axvline(recommended, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Independent 50k seeds in consensus, m")
    ax.set_ylabel(r"Subvolume p95($U_{pair}$) (%)")
    ax.set_title("Across-rock-location stability")
    ax.set_xticks(mvals)
    ax.legend(frameon=False, fontsize=7.3, ncol=2)
    panel_label(ax, "(e)")

    # ------------------------------------------------------------------ (f)
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rec = next(r for r in global_rows if r["m"] == recommended)
    Avals = [data[sid]["A_full"] for sid in SUBVOLUMES]

    status = "thresholds satisfied" if threshold_met else "maximum supported m"
    lines = [
        "FINAL reference-calibration decision",
        "",
        f"Recommended consensus: m = {recommended}",
        f"Decision: {status}",
        "",
        f"Pooled p95 uncertainty proxy: {100*rec['proxy_p95']:.2f}%",
        f"Worst-location p95 proxy: {100*rec['worst_subvolume_proxy_p95']:.2f}%",
        f"Median |ΔA|: {rec['abs_dA_median']:.4f}",
        f"Median scale difference: {100*rec['rel_ell_geom_median']:.2f}%",
        "",
        f"10-seed A range: {min(Avals):.3f}–{max(Avals):.3f}",
        "",
        "Next step: freeze this reference",
        "protocol and run final validation.",
        "No further calibration test required.",
    ]
    ax.text(
        0.04, 0.96,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.7,
        fontweight="bold",
        linespacing=1.35,
    )
    panel_label(ax, "(f)")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.075, top=0.96)
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if max(CONSENSUS_SIZES) > N_AVAILABLE_SEEDS:
        raise ValueError("CONSENSUS_SIZES cannot exceed N_AVAILABLE_SEEDS.")

    t0 = time.time()
    cache_dir = find_cache_dir()
    data = load_cached_tensors(cache_dir)

    exact_rows = exact_subset_analysis(data)
    bootstrap_rows = bootstrap_analysis(data)

    subvol_rows, global_rows, recommended, threshold_met = summarize_bootstrap(
        bootstrap_rows
    )

    print("=" * 78)
    print("CONSENSUS-CONVERGENCE SUMMARY")
    print("=" * 78)
    print(
        "Primary uncertainty proxy: U_pair = independent bootstrap-pair "
        "disagreement / sqrt(2)"
    )
    print()

    for r in global_rows:
        print(
            f"m={r['m']:2d}  "
            f"Efull med={100*r['E_median']:6.3f}% p95={100*r['E_p95']:6.3f}%  "
            f"Upair med={100*r['proxy_median']:6.3f}% "
            f"p95={100*r['proxy_p95']:6.3f}%  "
            f"worst-sv p95={100*r['worst_subvolume_proxy_p95']:6.3f}%  "
            f"|dA| med={r['abs_dA_median']:.5f}  "
            f"scale med={100*r['rel_ell_geom_median']:.3f}%"
        )

    rec = next(r for r in global_rows if r["m"] == recommended)

    print()
    print("=" * 78)
    print("FINAL PRACTICAL DECISION")
    print("=" * 78)
    print(f"Pooled p95 target:          <= {100*TARGET_P95_POOLED:.2f}%")
    print(f"Worst-subvolume p95 target: <= {100*TARGET_P95_WORST:.2f}%")
    print(f"Recommended consensus size: m={recommended}")
    print(f"Both thresholds satisfied:  {threshold_met}")
    print(f"Measured pooled p95 proxy:   {100*rec['proxy_p95']:.3f}%")
    print(
        f"Measured worst-sv p95:       "
        f"{100*rec['worst_subvolume_proxy_p95']:.3f}%"
    )
    if not threshold_met:
        print(
            "NOTE: thresholds were not both reached. Use m=10 as the practical "
            "maximum supported by the current cache and carry the measured "
            "reference uncertainty into the final error bars. Do NOT launch "
            "another calibration test."
        )
    else:
        print(
            "Decision frozen: use this m for the final multi-seed 50k reference "
            "protocol. No further calibration test is required."
        )
    print()

    write_csv(EXACT_SUBSETS_CSV_PATH, exact_rows)
    write_csv(BOOTSTRAP_CSV_PATH, bootstrap_rows)
    write_csv(SUBVOL_SUMMARY_CSV_PATH, subvol_rows)
    write_csv(SUMMARY_CSV_PATH, global_rows)

    write_report(
        data, global_rows, subvol_rows,
        recommended, threshold_met, cache_dir,
    )

    build_figure(
        global_rows, subvol_rows, data,
        recommended, threshold_met,
    )

    print("=" * 78)
    print("FILES GENERATED")
    print("=" * 78)
    print(PNG_PATH)
    print(SVG_PATH)
    print(REPORT_PATH)
    print(SUMMARY_CSV_PATH)
    print(SUBVOL_SUMMARY_CSV_PATH)
    print(BOOTSTRAP_CSV_PATH)
    print(EXACT_SUBSETS_CSV_PATH)
    print()
    print(f"Total wall time: {time.time()-t0:.1f} s")
    print("Bentheimer Test 5 completed successfully.")


if __name__ == "__main__":
    main()