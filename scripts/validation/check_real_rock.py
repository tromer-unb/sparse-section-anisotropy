#!/usr/bin/env python3
"""
Initial validation of the real Bentheimer micro-CT binary volume.

The script:
  1. checks the RAW file size;
  2. opens it with numpy.memmap (no full-volume RAM loading);
  3. samples several planes throughout the volume;
  4. reports the binary values present and their fractions;
  5. shows both possible pore-fraction conventions;
  6. exports the three central orthogonal slices;
  7. optionally computes exact full-volume phase fractions in chunks.

Outputs:
  - bentheimer_check.png
  - bentheimer_check.svg
  - bentheimer_check_report.txt

Dependencies:
  python3 -m pip install numpy matplotlib
"""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

RAW_FILE = Path(
    "kocurek_15a_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw"
)

SHAPE = (2500, 2500, 2500)

DTYPE = np.uint8

# Number of approximately equally spaced planes sampled along each axis.
N_SAMPLE_PLANES = 12

# False = fast diagnostic sampling only.
# True  = also scans the complete 15.625 GB file sequentially.
EXACT_FULL_VOLUME = False

# Number of voxels processed per block in exact mode.
CHUNK_VOXELS = 100_000_000

DPI = 800


# =============================================================================
# OUTPUTS
# =============================================================================

OUTDIR = Path(__file__).resolve().parent

PNG_PATH = OUTDIR / "bentheimer_check.png"
SVG_PATH = OUTDIR / "bentheimer_check.svg"
REPORT_PATH = OUTDIR / "bentheimer_check_report.txt"


# =============================================================================
# STYLE
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
    "axes.linewidth": 0.9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# HELPERS
# =============================================================================

def human_bytes(n):
    """Return human-readable file size."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)

    for unit in units:
        if value < 1024.0:
            return f"{value:.3f} {unit}"
        value /= 1024.0

    return f"{value:.3f} PiB"


def check_file_size(path, shape, dtype):
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual = path.stat().st_size

    print("=" * 72)
    print("RAW FILE CHECK")
    print("=" * 72)

    print(f"File:     {path}")
    print(f"Shape:    {shape}")
    print(f"dtype:    {np.dtype(dtype)}")
    print(f"Expected: {expected} bytes ({human_bytes(expected)})")
    print(f"Actual:   {actual} bytes ({human_bytes(actual)})")

    if actual != expected:
        raise RuntimeError(
            "\nRAW size does not match the requested shape/dtype.\n"
            f"Expected {expected} bytes but found {actual} bytes."
        )

    print("Status:   PERFECT MATCH")
    print()

    return actual


def sample_plane_indices(n, count):
    """Approximately equally spaced internal planes."""
    return np.linspace(
        int(0.05 * n),
        int(0.95 * n),
        count,
        dtype=int,
    )


def sampled_histogram(volume):
    """
    Count uint8 values using selected complete 2D planes.

    This reads only a small fraction of the entire 15.6 GB volume.
    """
    hist = np.zeros(256, dtype=np.int64)

    nx, ny, nz = volume.shape

    xs = sample_plane_indices(nx, N_SAMPLE_PLANES)
    ys = sample_plane_indices(ny, N_SAMPLE_PLANES)
    zs = sample_plane_indices(nz, N_SAMPLE_PLANES)

    print("=" * 72)
    print("SAMPLING PLANES")
    print("=" * 72)

    print("X planes:", xs)
    print("Y planes:", ys)
    print("Z planes:", zs)
    print()

    for i in xs:
        plane = np.asarray(volume[i, :, :])
        hist += np.bincount(
            plane.ravel(),
            minlength=256,
        )

    for j in ys:
        plane = np.asarray(volume[:, j, :])
        hist += np.bincount(
            plane.ravel(),
            minlength=256,
        )

    for k in zs:
        plane = np.asarray(volume[:, :, k])
        hist += np.bincount(
            plane.ravel(),
            minlength=256,
        )

    return hist


def exact_histogram(volume):
    """
    Sequential full-volume histogram.

    RAM use remains small, but the entire ~15.6 GB file must be read.
    """
    flat = volume.reshape(-1)

    hist = np.zeros(256, dtype=np.int64)

    total = flat.size

    print("=" * 72)
    print("FULL-VOLUME SCAN")
    print("=" * 72)

    for start in range(0, total, CHUNK_VOXELS):
        end = min(start + CHUNK_VOXELS, total)

        chunk = np.asarray(flat[start:end])

        hist += np.bincount(
            chunk,
            minlength=256,
        )

        pct = 100.0 * end / total

        print(
            f"\rProcessed: {pct:6.2f}%",
            end="",
            flush=True,
        )

    print("\n")

    return hist


def summarize_histogram(hist, title):
    total = hist.sum()

    values = np.where(hist > 0)[0]

    print("=" * 72)
    print(title)
    print("=" * 72)

    print(f"Sampled voxels: {total:,}")
    print(f"Distinct values: {values.tolist()}")
    print()

    fractions = {}

    for value in values:
        fraction = hist[value] / total
        fractions[int(value)] = float(fraction)

        print(
            f"value {value:3d}: "
            f"{hist[value]:15,d} voxels  "
            f"fraction = {fraction:.6f} "
            f"({100*fraction:.3f}%)"
        )

    print()

    if len(values) == 2:

        v0, v1 = values

        print("BINARY CONVENTION CHECK")
        print("-----------------------")

        print(
            f"If value {v0} = pore:"
            f"  porosity = {fractions[int(v0)]:.6f}"
        )

        print(
            f"If value {v1} = pore:"
            f"  porosity = {fractions[int(v1)]:.6f}"
        )

        minority = min(
            fractions,
            key=fractions.get,
        )

        print()
        print(
            "Minority phase in sampled data:"
            f" value {minority}"
        )

        print(
            "IMPORTANT: minority phase is only a hint; "
            "it is not used automatically as the pore convention."
        )

    print()

    return values, fractions


def save_report(
    path,
    file_size,
    hist,
    values,
    fractions,
    exact=False,
):
    with open(path, "w", encoding="utf-8") as f:

        f.write("Bentheimer real-rock RAW validation\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"RAW file: {RAW_FILE}\n")
        f.write(f"Shape: {SHAPE}\n")
        f.write(f"dtype: {np.dtype(DTYPE)}\n")
        f.write(f"File size: {file_size} bytes\n")
        f.write(
            f"Expected size: "
            f"{int(np.prod(SHAPE))*np.dtype(DTYPE).itemsize} bytes\n"
        )
        f.write(f"Exact full-volume scan: {exact}\n\n")

        f.write("Observed values\n")
        f.write("-" * 40 + "\n")

        total = hist.sum()

        for value in values:
            f.write(
                f"value {int(value)}: "
                f"{hist[int(value)]} voxels, "
                f"fraction={fractions[int(value)]:.8f}\n"
            )

        if len(values) == 2:

            v0, v1 = values

            f.write("\nPossible pore conventions\n")
            f.write("-" * 40 + "\n")

            f.write(
                f"If {int(v0)} = pore: "
                f"phi={fractions[int(v0)]:.8f}\n"
            )

            f.write(
                f"If {int(v1)} = pore: "
                f"phi={fractions[int(v1)]:.8f}\n"
            )

        f.write(
            "\nAxis-order note:\n"
            "The cube dimensions alone do not establish the physical "
            "X/Y/Z orientation of the stored RAW axes. Metadata should be "
            "checked before assigning geological directions.\n"
        )


def central_slices(volume):
    ix = volume.shape[0] // 2
    iy = volume.shape[1] // 2
    iz = volume.shape[2] // 2

    xy = np.asarray(volume[:, :, iz])
    xz = np.asarray(volume[:, iy, :])
    yz = np.asarray(volume[ix, :, :])

    return xy, xz, yz


def plot_slices(volume, values, fractions):
    xy, xz, yz = central_slices(volume)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.0, 3.8),
    )

    images = [xy, xz, yz]

    titles = [
        "Central XY slice",
        "Central XZ slice",
        "Central YZ slice",
    ]

    for ax, image, title in zip(
        axes,
        images,
        titles,
    ):
        ax.imshow(
            image.T,
            origin="lower",
            cmap="gray",
            interpolation="nearest",
        )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    if len(values) == 2:

        v0, v1 = values

        subtitle = (
            f"Observed binary values: "
            f"{int(v0)} ({100*fractions[int(v0)]:.2f}%)  |  "
            f"{int(v1)} ({100*fractions[int(v1)]:.2f}%)"
        )

        fig.suptitle(
            subtitle,
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )

    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        bottom=0.04,
        top=0.86,
        wspace=0.08,
    )

    fig.savefig(
        PNG_PATH,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        SVG_PATH,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            f"\nRAW file not found:\n{RAW_FILE.resolve()}\n\n"
            "Place this script in the same directory as the RAW file "
            "or edit RAW_FILE at the top of the script."
        )

    file_size = check_file_size(
        RAW_FILE,
        SHAPE,
        DTYPE,
    )

    print("=" * 72)
    print("OPENING WITH MEMMAP")
    print("=" * 72)

    volume = np.memmap(
        RAW_FILE,
        dtype=DTYPE,
        mode="r",
        shape=SHAPE,
        order="C",
    )

    print("Memmap created successfully.")
    print("No full-volume RAM allocation was performed.")
    print()

    # ---------------------------------------------------------
    # Fast sampled analysis
    # ---------------------------------------------------------

    hist = sampled_histogram(volume)

    values, fractions = summarize_histogram(
        hist,
        "SAMPLED PHASE HISTOGRAM",
    )

    # ---------------------------------------------------------
    # Optional exact full scan
    # ---------------------------------------------------------

    if EXACT_FULL_VOLUME:

        hist = exact_histogram(volume)

        values, fractions = summarize_histogram(
            hist,
            "EXACT FULL-VOLUME PHASE HISTOGRAM",
        )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    plot_slices(
        volume,
        values,
        fractions,
    )

    save_report(
        REPORT_PATH,
        file_size,
        hist,
        values,
        fractions,
        exact=EXACT_FULL_VOLUME,
    )

    print("=" * 72)
    print("FILES GENERATED")
    print("=" * 72)

    print(PNG_PATH)
    print(SVG_PATH)
    print(REPORT_PATH)

    print()
    print("Validation complete.")


if __name__ == "__main__":
    main()