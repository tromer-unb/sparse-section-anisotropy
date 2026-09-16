"""Linear inverse problem for the 3D pore-correlation tensor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any

import numpy as np


@dataclass(frozen=True)
class TensorFit:
    Q: np.ndarray | None
    rank: int
    condition_number: float
    design_matrix: np.ndarray
    target: np.ndarray
    measurement_count: int


def project_spd(Q: np.ndarray, relative_floor: float = 1e-7) -> np.ndarray:
    """Symmetrize and project a matrix to SPD by eigenvalue flooring."""
    Q = np.asarray(Q, dtype=float)
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    floor = max(1e-12, relative_floor * float(np.max(np.abs(w))))
    w = np.maximum(w, floor)
    return V @ np.diag(w) @ V.T


def _validate_basis(B: np.ndarray, atol: float = 1e-7) -> np.ndarray:
    B = np.asarray(B, dtype=float)
    if B.shape != (3, 2):
        raise ValueError("section basis must have shape (3, 2)")
    if not np.allclose(B.T @ B, np.eye(2), atol=atol):
        raise ValueError("section basis columns must be orthonormal")
    return B


def design_rows_for_section(B: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    """Construct the six-column symmetric-tensor design block."""
    B = _validate_basis(B)
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    u = np.vstack([np.cos(theta), np.sin(theta)])
    v = B @ u
    vx, vy, vz = v
    return np.column_stack(
        [
            vx * vx,
            vy * vy,
            vz * vz,
            2.0 * vx * vy,
            2.0 * vx * vz,
            2.0 * vy * vz,
        ]
    )


def qvec_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (6,):
        raise ValueError("q must contain six independent symmetric-tensor components")
    return np.array(
        [
            [q[0], q[3], q[4]],
            [q[3], q[1], q[5]],
            [q[4], q[5], q[2]],
        ],
        dtype=float,
    )


def assemble_inverse_problem(measurements: Iterable[Mapping[str, Any]]):
    """Assemble A q = y from section bases, angles, and correlation lengths."""
    A_blocks = []
    y_blocks = []

    for m in measurements:
        B = _validate_basis(np.asarray(m["basis"], dtype=float))
        theta = np.asarray(m["theta_deg"], dtype=float)
        ell = np.asarray(m["length"], dtype=float)
        if theta.shape != ell.shape:
            raise ValueError("theta_deg and length must have identical shapes")

        good = np.isfinite(ell) & (ell > 0)
        if not np.any(good):
            continue
        A_blocks.append(design_rows_for_section(B, theta[good]))
        y_blocks.append(1.0 / (ell[good] * ell[good]))

    if not A_blocks:
        raise ValueError("no finite positive correlation-length measurements")

    return np.vstack(A_blocks), np.concatenate(y_blocks)


def infer_tensor(measurements: Iterable[Mapping[str, Any]]) -> TensorFit:
    """Fit Q from oriented section measurements.

    If the global design matrix has rank < 6, Q is returned as None rather than
    silently computing a pseudoinverse tensor.
    """
    A, y = assemble_inverse_problem(measurements)
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    s = np.linalg.svd(A, compute_uv=False)
    cond = float(np.inf if s[-1] <= 0 else s[0] / s[-1])

    if rank < 6:
        return TensorFit(None, rank, cond, A, y, int(len(y)))

    q, *_ = np.linalg.lstsq(A, y, rcond=None)
    Q = project_spd(qvec_to_matrix(q))
    return TensorFit(Q, rank, cond, A, y, int(len(y)))


def principal_properties(Q: np.ndarray) -> dict:
    """Return eigensystem, principal lengths, and anisotropy ratio."""
    Q = project_spd(Q)
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    lengths = 1.0 / np.sqrt(eigenvalues)
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "principal_lengths": lengths,
        "anisotropy_ratio": float(np.max(lengths) / np.min(lengths)),
        "geometric_mean_length": float(np.linalg.det(Q) ** (-1.0 / 6.0)),
    }


def determinant_normalized(Q: np.ndarray) -> np.ndarray:
    Q = project_spd(Q)
    return Q / np.linalg.det(Q) ** (1.0 / 3.0)
