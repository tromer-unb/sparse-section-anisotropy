import numpy as np

from sparse_section_anisotropy.tensor import infer_tensor, principal_properties


B_XY = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
B_XZ = np.array([[1, 0], [0, 0], [0, 1]], dtype=float)
B_YZ = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)


def exact_measurement(Q, B, step=5.0):
    theta = np.arange(0.0, 180.0, step)
    th = np.deg2rad(theta)
    u = np.vstack([np.cos(th), np.sin(th)])
    v = B @ u
    inv_ell2 = np.einsum("in,ij,jn->n", v, Q, v)
    return {"basis": B, "theta_deg": theta, "length": 1.0 / np.sqrt(inv_ell2)}


def test_rank_hierarchy_for_three_orthogonal_planes():
    Q = np.array([[0.8, 0.07, -0.04], [0.07, 1.3, 0.09], [-0.04, 0.09, 2.1]])
    ms = [exact_measurement(Q, B) for B in (B_XY, B_XZ, B_YZ)]
    assert infer_tensor(ms[:1]).rank == 3
    assert infer_tensor(ms[:2]).rank == 5
    assert infer_tensor(ms).rank == 6


def test_exact_tensor_recovery():
    Q = np.array([[0.8, 0.07, -0.04], [0.07, 1.3, 0.09], [-0.04, 0.09, 2.1]])
    fit = infer_tensor([exact_measurement(Q, B) for B in (B_XY, B_XZ, B_YZ)])
    assert fit.Q is not None
    np.testing.assert_allclose(fit.Q, Q, rtol=1e-10, atol=1e-10)
    props = principal_properties(fit.Q)
    assert props["anisotropy_ratio"] > 1.0
