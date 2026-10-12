import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from io_leontief import (
    apply_scenario,
    emissions_by_region,
    footprint,
    harness_checks,
    leontief_output,
    low_rank_factors,
    make_economy,
    woodbury_output,
)


def test_synthetic_economy_is_productive():
    A, _, _ = make_economy()
    assert A.sum(axis=0).max() < 1
    assert np.all(np.linalg.solve(np.eye(len(A)) - A, np.ones(len(A))) > 0)


def test_notebook_runs_as_a_script():
    subprocess.run([sys.executable, "io_leontief.py"], cwd=Path(__file__).parent.parent, check=True)


def test_hand_computed_two_by_two():
    # I - A = [[0.8, -0.3], [-0.1, 0.6]], det = 0.45
    # L = [[0.6, 0.3], [0.1, 0.8]] / 0.45; y = (10, 20) -> x = (12, 17) / 0.45
    A = np.array([[0.2, 0.3], [0.1, 0.4]])
    y = np.array([10.0, 20.0])
    np.testing.assert_allclose(leontief_output(A, y), [12 / 0.45, 17 / 0.45], rtol=1e-14)
    e = np.array([1.0, 0.5])
    assert footprint(e, A, y) == pytest.approx((12 + 8.5) / 0.45, rel=1e-14)


def test_emissions_by_region_sums_region_blocks():
    e = np.ones(6)
    x = np.arange(6.0)
    np.testing.assert_array_equal(emissions_by_region(e, x, n_regions=2), [3.0, 12.0])


@pytest.mark.parametrize(("coef_pct", "demand_pct"), [(0, 0), (-50, 0), (0, 50), (30, -20), (50, 50)])
def test_harness_checks_pass_across_scenarios(coef_pct, demand_pct):
    A, y, e = make_economy()
    A_s, y_s = apply_scenario(A, y, (4, 3), coef_pct, 0, demand_pct)
    checks = harness_checks(A, y, e, A_s, y_s, coef_pct, demand_pct)
    assert all(ok for ok, _ in checks.values()), checks


def test_scenario_does_not_mutate_baseline():
    A, y, _ = make_economy()
    A_copy, y_copy = A.copy(), y.copy()
    apply_scenario(A, y, (4, 3), -50, 0, 50)
    np.testing.assert_array_equal(A, A_copy)
    np.testing.assert_array_equal(y, y_copy)


def test_harness_flags_a_wrong_effect_sign():
    A, y, e = make_economy()
    A_s, y_s = apply_scenario(A, y, (4, 3), -50, 0, 0)
    checks = harness_checks(A, y, e, A_s, y_s, coefficient_pct=50)  # claims a raise, applies a cut
    assert checks["effect sign as expected"][0] is False


def test_woodbury_hand_computed_two_by_two():
    # A[0, 1] 0.3 -> 0.2: I - A_s = [[0.8, -0.2], [-0.1, 0.6]], det = 0.46
    # L_s = [[0.6, 0.2], [0.1, 0.8]] / 0.46; y = (10, 20) -> x = (10, 17) / 0.46
    A = np.array([[0.2, 0.3], [0.1, 0.4]])
    A_s = np.array([[0.2, 0.2], [0.1, 0.4]])
    U, V = low_rank_factors(A, A_s)
    assert U.shape == (2, 1)
    L = np.linalg.inv(np.eye(2) - A)
    np.testing.assert_allclose(woodbury_output(L, U, V, np.array([10.0, 20.0])), [10 / 0.46, 17 / 0.46], rtol=1e-14)


def test_woodbury_factors_rebuild_the_scenario():
    A, y, _ = make_economy()
    A_s, _ = apply_scenario(A, y, (4, 3), -50)
    A_s[:, 7] *= 0.9  # a second changed column: rank 2
    U, V = low_rank_factors(A, A_s)
    assert U.shape == (len(A), 2)
    np.testing.assert_allclose(A + U @ V.T, A_s, rtol=0, atol=1e-15)


@pytest.mark.parametrize("columns", [[], [3], [3, 7], [0, 4, 8], list(range(9))])
def test_woodbury_equals_full_resolve(columns):
    A, y, _ = make_economy()
    A_s = A.copy()
    A_s[:, columns] *= 0.6
    y_s = y * 1.1
    U, V = low_rank_factors(A, A_s)
    assert U.shape[1] == len(columns)
    x_w = woodbury_output(np.linalg.inv(np.eye(len(A)) - A), U, V, y_s)
    x_s = leontief_output(A_s, y_s)
    assert np.abs(x_w - x_s).max() / np.abs(x_s).max() < 1e-12


@pytest.mark.parametrize(("coef_pct", "demand_pct"), [(0, 0), (-50, 0), (0, 50), (30, -20), (50, 50)])
def test_harness_woodbury_check_passes(coef_pct, demand_pct):
    A, y, e = make_economy()
    A_s, y_s = apply_scenario(A, y, (4, 3), coef_pct, 0, demand_pct)
    ok, detail = harness_checks(A, y, e, A_s, y_s, coef_pct, demand_pct)["Woodbury update equals re-solve"]
    assert ok, detail
    assert detail.startswith(f"rank {int(coef_pct != 0)},")
