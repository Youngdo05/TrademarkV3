import numpy as np
import pandas as pd
import pytest

from selected.common import IntegrityError
from selected.modeling import ModelFitError, OutcomeLedger, fit_diagnostics, fit_lgbm


def outcome(available, version, value=0.1, status="RESOLVED"):
    return dict(session="2019-01-02", security_id="SYNTH", maturity_session="2019-02-01",
        available_session=available, version=version, status=status, y_excess=value,
        evidence_basis="synthetic", evidence_ref="generated-test-fixture")


def test_historical_snapshot_preserves_revisions_and_ignores_future_poison():
    records = pd.DataFrame([outcome("2019-02-04", "v1"), outcome("2019-03-01", "v2", 0.2),
        outcome("2019-04-01", "v3", None, "UNRESOLVED_TERMS"), outcome("2019-05-01", "v4", -0.5)])
    keys = records.iloc[:1][["session", "security_id", "maturity_session"]]
    ledger = OutcomeLedger(records)
    for cutoff, expected in (("2019-02-01", None), ("2019-02-28", 0.1),
                            ("2019-03-01", 0.2), ("2019-04-30", None), ("2019-05-01", -0.5)):
        snapshot = ledger.snapshot(keys, cutoff)
        assert bool(snapshot.resolved.iloc[0]) == (expected is not None)
        if expected is not None:
            assert snapshot.y_excess.iloc[0] == expected
    poisoned = pd.concat([records, pd.DataFrame([outcome("2021-12-31", "future", "INVALID", "POISON")])])
    pd.testing.assert_frame_equal(ledger.snapshot(keys, "2019-05-01"),
        OutcomeLedger(poisoned).snapshot(keys, "2019-05-01"))


@pytest.mark.parametrize("changes", [
    {"y_excess": np.inf}, {"available_session": "2019-01-15"},
    {"evidence_ref": ""}, {"status": "UNRESOLVED_TERMS"},
])
def test_visible_invalid_evidence_fails_instead_of_silently_dropping_the_row(changes):
    record = outcome("2019-02-04", "v1")
    keys = pd.DataFrame([record])[["session", "security_id", "maturity_session"]]
    with pytest.raises(IntegrityError):
        OutcomeLedger(pd.DataFrame([{**record, **changes}])).snapshot(keys, "2019-03-01")


def test_ambiguous_versions_are_rejected():
    record = outcome("2019-02-04", "v1")
    keys = pd.DataFrame([record])[["session", "security_id", "maturity_session"]]
    with pytest.raises(IntegrityError, match="duplicate"):
        OutcomeLedger(pd.DataFrame([record, {**record, "version": "v2"}])).snapshot(keys, "2019-03-01")


def synthetic_fit_inputs():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(800, 3))
    y = 0.03*X[:, 0] - 0.01*X[:, 1] + rng.normal(0, 0.002, len(X))
    rows = pd.DataFrame([{**outcome("2019-02-04", "v1", float(y[i])),
        "security_id": f"SYNTH-{i:04d}"} for i in range(len(X))])
    return X, rows


def fit_visible(X, records):
    keys = records.loc[records.version.eq("v1"), ["session", "security_id", "maturity_session"]]
    visible = OutcomeLedger(records).snapshot(keys, "2019-02-05")
    assert visible.resolved.all()
    return fit_lgbm(X, visible.y_excess.to_numpy(), np.ones(len(X)),
        {"learning_rate": 0.06, "num_leaves": 7, "min_data_in_leaf": 15},
        seed=7, require_exact_rounds=True, fit_context="public_synthetic_fixture")


def test_future_revision_cannot_change_the_actual_lightgbm_fit():
    X, rows = synthetic_fit_inputs()
    future = rows.iloc[:1].assign(available_session="2021-12-31", version="future",
        status="POISON", y_excess="INVALID")
    first = fit_visible(X, rows)
    repeated = fit_visible(X, pd.concat([rows, future], ignore_index=True))
    np.testing.assert_array_equal(first.predict(X), repeated.predict(X))
    assert first.model_to_string() == repeated.model_to_string()
    assert np.isfinite(first.predict(X)).all()
    assert fit_diagnostics(first)["iterations"] == (24,)


@pytest.mark.parametrize("change", ["infinite_feature", "infinite_target", "negative_weight", "one_class", "unregistered_seed"])
def test_invalid_fit_requests_fail_before_returning_a_model(change):
    X, y, weights = np.zeros((2, 3)), np.array([0.0, 1.0]), np.ones(2)
    seed, objective = 7, "huber"
    if change == "infinite_feature":
        X[0, 0] = np.inf
    elif change == "infinite_target":
        y[0] = np.inf
    elif change == "negative_weight":
        weights[0] = -1
    elif change == "one_class":
        y[:], objective = 1, "binary"
    else:
        seed = 8
    with pytest.raises(ModelFitError):
        fit_lgbm(X, y, weights, {"learning_rate": 0.06, "num_leaves": 7, "min_data_in_leaf": 15},
            seed=seed, objective=objective)
