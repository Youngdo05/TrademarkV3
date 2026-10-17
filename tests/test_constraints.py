import cvxpy as cp
import numpy as np
import pytest

from selected.constraints import ExposureRow, ExposureRows, attach_exposure_limits


def project(target, rows, indices=None):
    x = cp.Variable(len(target))
    ceilings, (active, _) = attach_exposure_limits(x, rows, indices)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x-np.asarray(target))),
        [x >= 0, cp.sum(x) <= 1, *ceilings])
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    assert problem.status == cp.OPTIMAL
    return np.asarray(x.value), active


def test_name_and_group_limits_match_a_hand_solved_allocation():
    rows = ExposureRows(3, (ExposureRow((1, 0, 0), 0.25),
        ExposureRow((1, 1, 0), 0.50), ExposureRow((0, 0, 0), 0.0)))
    weights, active = project([0.8, 0.4, 0.2], rows)
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.20], atol=1e-7)
    np.testing.assert_array_equal(active, [0, 1])
    assert weights[0]+weights[1] <= 0.50+1e-8
    assert weights.sum() <= 1+1e-8


def test_geared_exposure_is_distinct_from_capital_weight():
    rows = ExposureRows(2, (ExposureRow((3, 1), 0.60),))
    weights, _ = project([0.4, 0.0], rows)
    np.testing.assert_allclose(weights, [0.2, 0.0], atol=1e-7)
    assert 3*weights[0]+weights[1] <= 0.60+1e-8


def test_selected_columns_preserve_the_original_asset_mapping():
    rows = ExposureRows(3, (ExposureRow((1, 0, 2), 0.60),))
    weights, _ = project([0.4, 0.2], rows, indices=np.array([2, 0]))
    np.testing.assert_allclose(weights, [0.24, 0.12], atol=1e-7)
    assert 2*weights[0]+weights[1] <= 0.60+1e-8


def test_zero_rows_do_not_create_a_spurious_constraint():
    rows = ExposureRows(2, (ExposureRow((0, 0), 0.0),))
    weights, active = project([0.2, 0.3], rows)
    np.testing.assert_allclose(weights, [0.2, 0.3], atol=1e-7)
    assert active.size == 0


def test_sparse_cache_is_immutable_and_refreshes_when_rows_change():
    rows = ExposureRows(2, (ExposureRow((1, 0), 0.4),))
    first, caps = rows.sparse_geometry()
    assert rows.sparse_geometry()[0] is first
    assert not first.data.flags.writeable and not caps.flags.writeable
    rows.constraints = (ExposureRow((0, 1), 0.3),)
    changed, new_caps = rows.sparse_geometry()
    assert changed is not first
    np.testing.assert_array_equal(changed.toarray(), [[0, 1]])
    np.testing.assert_array_equal(new_caps, [0.3])
    np.testing.assert_array_equal(caps, [0.4])


@pytest.mark.parametrize("indices", [[0, 0], [0, 3], [0.0, 1.0]])
def test_public_adapter_rejects_ambiguous_or_invalid_column_indices(indices):
    rows = ExposureRows(2, (ExposureRow((1, 1), 0.6),))
    with pytest.raises(ValueError, match="unique and aligned"):
        attach_exposure_limits(cp.Variable(2), rows, indices)


def test_public_adapter_rejects_an_impossible_zero_row_cap():
    with pytest.raises(ValueError, match="nonnegative"):
        ExposureRows(2, (ExposureRow((0, 0), -0.1),))
