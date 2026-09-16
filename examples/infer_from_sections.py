#!/usr/bin/env python3
"""Minimal example: recover a 3D correlation tensor from oriented 2D sections.

This example starts after directional correlation lengths have been measured
from binary 2D images. It generates exact synthetic measurements from a known
3D SPD tensor and then recovers that tensor using the same inverse geometry
used in the manuscript.
"""

import numpy as np

from sparse_section_anisotropy import SectionMeasurements, infer_q3d, principal_summary


def section_measurements(Q, B, theta_deg):
    theta = np.deg2rad(theta_deg)
    u = np.vstack((np.cos(theta), np.sin(theta)))
    v = (B @ u).T
    inv_ell2 = np.einsum("ni,ij,nj->n", v, Q, v)
    ell = 1.0 / np.sqrt(inv_ell2)
    return SectionMeasurements(B, theta_deg, ell)


def main():
    # Known SPD tensor in the ambient coordinate system.
    R = np.array(
        [
            [0.8660254, -0.5, 0.0],
            [0.5, 0.8660254, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    principal_lengths = np.array([8.0, 5.0, 3.0])
    Q_true = R @ np.diag(1.0 / principal_lengths**2) @ R.T

    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    bases = [
        np.column_stack((ex, ey)),
        np.column_stack((ex, ez)),
        np.column_stack((ey, ez)),
    ]

    theta_deg = np.arange(0.0, 180.0, 5.0)
    sections = [section_measurements(Q_true, B, theta_deg) for B in bases]

    Q_hat, diagnostics = infer_q3d(sections)
    summary = principal_summary(Q_hat)

    rel_error = np.linalg.norm(Q_hat - Q_true) / np.linalg.norm(Q_true)

    np.set_printoptions(precision=6, suppress=True)
    print("Q_true:\n", Q_true)
    print("\nQ_recovered:\n", Q_hat)
    print(f"\nrelative tensor error = {rel_error:.3e}")
    print("diagnostics =", diagnostics)
    print("principal lengths =", summary["principal_lengths"])
    print("anisotropy ratio =", summary["anisotropy_ratio"])


if __name__ == "__main__":
    main()
