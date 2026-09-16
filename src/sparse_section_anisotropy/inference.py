"""Core tensor inference utilities for sparse oriented 2D sections.

The functions in this module operate on already-measured directional
correlation lengths. They intentionally separate image measurement from the
six-parameter 3D inverse problem so the inference stage can be tested and
reused independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SectionMeasurements:
    """Directional measurements from one oriented 2D section.

    Parameters
    ----------
    basis:
        Orthonormal 3x2 matrix whose columns span the section in the ambient
        3D coordinate system.
    theta_deg:
        In-plane direction angles in degrees, measured relative to basis[:, 0].
    correlation_lengths:
        Positive correlation lengths corresponding to theta_deg.
    """

    basis: np.ndarray
    theta_deg: np.ndarray
    correlation_lengths: np.ndarray


def project_spd(matrix: np.ndarray, relative_floor: float = 1e-8) -> np.ndarray:
    """Symmetrize a matrix and project it to the SPD cone."""
    q = np.asarray(matrix, dtype=float)
    q = 0.5 * (q + q.T)
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    floor = max(1e-12, relative_floor * np.max(np.abs(eigenvalues)))
    eigenvalues = np.maximum(eigenvalues, floor)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def design_rows(vectors: np.ndarray) -> np.ndarray:
    """Return linear rows for the six independent entries of a symmetric Q."""
    v = np.asarray(vectors, dtype=float)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.column_stack((x*x, y*y, z*z, 2*x*y, 2*x*z, 2*y*z))


def _qvec_to_matrix(q: np.ndarray) -> np.ndarray:
    return np.array(
        [[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]],
        dtype=float,
    )


def assemble_inverse_problem(
    sections: Iterable[SectionMeasurements],
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble A q = y with y = 1/ell^2 from oriented sections."""
    rows = []
    values = []

    for section in sections:
        basis = np.asarray(section.basis, dtype=float)
        theta = np.asarray(section.theta_deg, dtype=float)
        ell = np.asarray(section.correlation_lengths, dtype=float)

        if basis.shape != (3, 2):
            raise ValueError("Each section basis must have shape (3, 2).")
        if theta.shape != ell.shape:
            raise ValueError("theta_deg and correlation_lengths must have equal shape.")
        if not np.allclose(basis.T @ basis, np.eye(2), atol=1e-8):
            raise ValueError("Section basis columns must be orthonormal.")

        good = np.isfinite(ell) & (ell > 0)
        if not np.any(good):
            continue

        angle = np.deg2rad(theta[good])
        u = np.vstack((np.cos(angle), np.sin(angle)))
        v = (basis @ u).T
        rows.append(design_rows(v))
        values.append(1.0 / np.square(ell[good]))

    if not rows:
        raise ValueError("No valid directional measurements were supplied.")

    return np.vstack(rows), np.concatenate(values)


def infer_q3d(
    sections: Iterable[SectionMeasurements],
) -> tuple[np.ndarray, dict[str, float]]:
    """Infer the 3D SPD correlation tensor from sparse oriented sections.

    Returns
    -------
    Q:
        3x3 SPD tensor.
    diagnostics:
        Dictionary containing rank, condition number, number of observations,
        and normalized least-squares residual in y = 1/ell^2 space.
    """
    A, y = assemble_inverse_problem(sections)
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    if rank < 6:
        raise ValueError(
            f"The aggregate section geometry is not fully observable: rank={rank} < 6."
        )

    q, *_ = np.linalg.lstsq(A, y, rcond=None)
    Q = project_spd(_qvec_to_matrix(q))

    singular_values = np.linalg.svd(A, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1])
    residual = A @ np.array([Q[0,0], Q[1,1], Q[2,2], Q[0,1], Q[0,2], Q[1,2]]) - y
    relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(y))

    diagnostics = {
        "rank": float(rank),
        "condition_number": condition,
        "n_observations": float(len(y)),
        "relative_y_residual": relative_residual,
    }
    return Q, diagnostics


def principal_summary(Q: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return principal correlation lengths, axes, and anisotropy ratio."""
    Q = project_spd(Q)
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    lengths = 1.0 / np.sqrt(eigenvalues)
    order = np.argsort(lengths)[::-1]
    lengths = lengths[order]
    axes = eigenvectors[:, order]
    return {
        "principal_lengths": lengths,
        "principal_axes": axes,
        "anisotropy_ratio": float(lengths[0] / lengths[-1]),
    }
