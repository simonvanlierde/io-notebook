import numpy as np
import pytest

from io_leontief import (
    apply_scenario,
    emissions_by_region,
    footprint,
    harness_checks,
    leontief_output,
    make_economy,
)


def test_synthetic_economy_is_productive():
    A, _, _ = make_economy()
    assert A.sum(axis=0).max() < 1
    assert np.all(np.linalg.solve(np.eye(len(A)) - A, np.ones(len(A))) > 0)


def test_leontief_identity():
    A, y, _ = make_economy()
    x = leontief_output(A, y)
    residual = np.abs((np.eye(len(A)) - A) @ x - y).max() / np.abs(y).max()
    assert residual < 1e-14


def test_zero_percent_control_equals_baseline():
    A, y, b = make_economy()
    A_0, y_0 = apply_scenario(A, y, coefficient=(4, 3), demand_sector=2)
    assert footprint(b, A_0, y_0) == footprint(b, A, y)


def test_hand_computed_two_by_two():
    # I - A = [[0.8, -0.3], [-0.1, 0.6]], det = 0.45
    # L = [[0.6, 0.3], [0.1, 0.8]] / 0.45; y = (10, 20) -> x = (12, 17) / 0.45
    A = np.array([[0.2, 0.3], [0.1, 0.4]])
    y = np.array([10.0, 20.0])
    np.testing.assert_allclose(leontief_output(A, y), [12 / 0.45, 17 / 0.45], rtol=1e-14)
    b = np.array([1.0, 0.5])
    assert footprint(b, A, y) == pytest.approx((12 + 8.5) / 0.45, rel=1e-14)


def test_emissions_by_region_sums_region_blocks():
    b = np.ones(6)
    x = np.arange(6.0)
    np.testing.assert_array_equal(emissions_by_region(b, x, n_regions=2), [3.0, 12.0])


@pytest.mark.parametrize(("coef_pct", "demand_pct"), [(0, 0), (-50, 0), (0, 50), (30, -20), (50, 50)])
def test_harness_checks_pass_across_scenarios(coef_pct, demand_pct):
    A, y, b = make_economy()
    A_s, y_s = apply_scenario(A, y, (4, 3), coef_pct, 0, demand_pct)
    checks = harness_checks(A, y, b, A_s, y_s, coef_pct, demand_pct)
    assert all(ok for ok, _ in checks.values()), checks


def test_scenario_does_not_mutate_baseline():
    A, y, _ = make_economy()
    A_copy, y_copy = A.copy(), y.copy()
    apply_scenario(A, y, (4, 3), -50, 0, 50)
    np.testing.assert_array_equal(A, A_copy)
    np.testing.assert_array_equal(y, y_copy)


def test_harness_flags_a_wrong_effect_sign():
    A, y, b = make_economy()
    A_s, y_s = apply_scenario(A, y, (4, 3), -50, 0, 0)
    checks = harness_checks(A, y, b, A_s, y_s, coefficient_pct=50)  # claims a raise, applies a cut
    assert checks["effect sign as expected"][0] is False
