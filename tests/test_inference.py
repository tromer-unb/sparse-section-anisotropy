import numpy as np

from sparse_section_anisotropy import SectionMeasurements, infer_q3d


def _measure(Q, B, theta_deg):
    theta = np.deg2rad(theta_deg)
    u = np.vstack((np.cos(theta), np.sin(theta)))
    v = (B @ u).T
    y = np.einsum("ni,ij,nj->n", v, Q, v)
    return SectionMeasurements(B, theta_deg, 1.0 / np.sqrt(y))


def test_exact_recovery_from_three_orthogonal_planes():
    Q = np.array(
        [
            [0.030, 0.004, -0.002],
            [0.004, 0.050, 0.003],
            [-0.002, 0.003, 0.090],
        ]
    )
    assert np.all(np.linalg.eigvalsh(Q) > 0)

    e = np.eye(3)
    bases = [
        np.column_stack((e[:, 0], e[:, 1])),
        np.column_stack((e[:, 0], e[:, 2])),
        np.column_stack((e[:, 1], e[:, 2])),
    ]
    theta = np.arange(0.0, 180.0, 5.0)
    sections = [_measure(Q, B, theta) for B in bases]

    Q_hat, diagnostics = infer_q3d(sections)

    assert diagnostics["rank"] == 6.0
    assert np.allclose(Q_hat, Q, rtol=1e-10, atol=1e-12)


def test_two_orthogonal_planes_are_rank_deficient():
    Q = np.diag([0.03, 0.05, 0.09])
    e = np.eye(3)
    theta = np.arange(0.0, 180.0, 5.0)
    sections = [
        _measure(Q, np.column_stack((e[:, 0], e[:, 1])), theta),
        _measure(Q, np.column_stack((e[:, 0], e[:, 2])), theta),
    ]

    try:
        infer_q3d(sections)
    except ValueError as exc:
        assert "rank=5" in str(exc)
    else:
        raise AssertionError("Two orthogonal sections should not identify all six components.")
