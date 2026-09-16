import numpy as np

from sparse_section_anisotropy.correlation import (
    autocorrelation_field,
    first_crossing_length,
)


def test_first_crossing_is_linearly_interpolated():
    r = np.array([0.0, 1.0, 2.0])
    c = np.array([1.0, 0.5, 0.2])
    value = first_crossing_length(r, c, threshold=0.4)
    assert np.isclose(value, 4.0 / 3.0)


def test_no_crossing_is_censored():
    r = np.arange(4.0)
    c = np.array([1.0, 0.9, 0.8, 0.7])
    assert np.isnan(first_crossing_length(r, c, threshold=np.exp(-1.0)))


def test_autocorrelation_zero_lag_is_one():
    binary = np.zeros((32, 32), dtype=bool)
    binary[8:24, 10:22] = True
    acf, center = autocorrelation_field(binary)
    assert np.isclose(acf[center], 1.0)
