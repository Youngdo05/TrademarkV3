from __future__ import annotations

from datetime import date
import math
import time
import warnings

import numpy as np
import pandas as pd

from selected.common import IntegrityError

EXAMPLE_SEEDS = (7,)
EXAMPLE_FIT_LIMIT_SECONDS = 30.0
EXAMPLE_FIT_SETTINGS = dict(n_estimators=24, feature_fraction=1.0,
    bagging_fraction=1.0, bagging_freq=0, deterministic=True,
    force_col_wise=True, device_type="cpu", num_threads=1)

KEYS = ["session", "security_id"]
FIELDS = [*KEYS, "maturity_session", "available_session", "version", "status",
          "y_excess", "evidence_basis", "evidence_ref"]
RESOLVED = ("RESOLVED", "ZERO_RECOVERY_PROVEN")
UNRESOLVED = ("OPEN_UNAVAILABLE", "UNRESOLVED_TERMS", "UNRESOLVED_EXISTENCE",
              "UNRESOLVED_CONTINUITY", "UNRESOLVED_NONCASH_ACTION",
              "UNRESOLVED_DECISION_INPUT", "UNRESOLVED_PRICE_HISTORY", "UNRESOLVED_CASH_WEALTH")
BASES = ("observed", "reconstructed_policy", "synthetic")
AUDIT_FIELDS = ["outcome_ticker", "large_stock_distribution", "large_spy_distribution"]


def _iso_day(value):
    return isinstance(value, str) and len(value) == 10 and date.fromisoformat(value).isoformat() == value


class OutcomeLedger:

    def __init__(self, records):
        if not set(FIELDS).issubset(records.columns):
            raise IntegrityError("dated outcome ledger columns missing")
        self._records = records[[*FIELDS, *[c for c in AUDIT_FIELDS if c in records]]].copy(deep=True)

    def snapshot(self, keys, fit_session):

        try:
            if not _iso_day(fit_session):
                raise ValueError
        except (ValueError, TypeError) as exc:
            raise IntegrityError("outcome fit cutoff must be an ISO session") from exc
        if (not {*KEYS, "maturity_session"}.issubset(keys.columns)
                or keys.duplicated(KEYS).any()):
            raise IntegrityError("outcome snapshot requires unique original sample keys")
        base = keys[[*KEYS, "maturity_session"]].reset_index(drop=True).copy()
        frame = self._records
        available = frame.available_session

        visible = available.map(lambda x: isinstance(x, str) and x <= fit_session)
        rows = frame.loc[visible].merge(base[KEYS], on=KEYS, how="inner", validate="many_to_one")
        if len(rows):
            try:
                valid_dates = rows.available_session.map(_iso_day) & rows.maturity_session.map(_iso_day)
            except (ValueError, TypeError) as exc:
                raise IntegrityError("outcome dates must be ISO sessions") from exc
            if (not valid_dates.all() or (rows.available_session < rows.maturity_session).any()
                    or not rows.status.isin((*RESOLVED, *UNRESOLVED)).all()
                    or not rows.evidence_basis.isin(BASES).all()
                    or any(not rows[c].map(lambda x: isinstance(x, str) and bool(x)).all()
                           for c in ("version", "evidence_ref"))):
                raise IntegrityError("invalid dated outcome status, evidence or availability")
            if (rows.duplicated([*KEYS, "available_session"]).any()
                    or rows.duplicated([*KEYS, "version"]).any()):
                raise IntegrityError("duplicate or ambiguous outcome version")
            expected = rows.merge(base, on=KEYS, suffixes=("", "_expected"), validate="many_to_one")
            if not expected.maturity_session.eq(expected.maturity_session_expected).all():
                raise IntegrityError("outcome maturity differs from the original sample horizon")
            numeric = pd.to_numeric(rows.y_excess, errors="coerce")
            resolved = rows.status.isin(RESOLVED)
            if not np.isfinite(numeric[resolved]).all() or rows.loc[~resolved, "y_excess"].notna().any():
                raise IntegrityError("resolved outcomes require finite values; unresolved versions carry no value")
            rows["y_excess"] = numeric
            rows = rows.sort_values([*KEYS, "available_session"]).drop_duplicates(KEYS, keep="last")
        out = base.merge(rows.drop(columns="maturity_session"), on=KEYS, how="left", validate="one_to_one")
        out["outcome_status"] = out.pop("status").fillna("NO_KNOWN_OUTCOME")
        out["resolved"] = out.outcome_status.isin(RESOLVED)
        out.loc[~out.resolved, "y_excess"] = np.nan
        return out

    @classmethod
    def combine(cls, ledgers):
        ledgers = list(ledgers)
        return cls(pd.concat([v._records for v in ledgers], ignore_index=True)
                   if ledgers else pd.DataFrame(columns=FIELDS))


class ModelFitError(RuntimeError):
    pass


def _finite_fit_inputs(X, y, w, *, allow_nan_X: bool, binary: bool) -> tuple:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or w.ndim != 1 \
            or len(y) != len(w) or X.shape[0] != len(y) or len(y) == 0:
        raise ModelFitError("model fit inputs are empty or not row aligned")
    invalid_X = np.isinf(X) if allow_nan_X else ~np.isfinite(X)
    weight_sum = float(w.sum())
    if (np.any(invalid_X) or np.any(~np.isfinite(y))
            or np.any(~np.isfinite(w)) or np.any(w < 0)
            or not math.isfinite(weight_sum) or weight_sum <= 0.0):
        raise ModelFitError("model fit inputs must be finite with positive weight")
    if binary:
        if np.any((y != 0.0) & (y != 1.0)):
            raise ModelFitError("binary model targets must be exactly zero or one")
        if len(np.unique(y)) != 2:
            raise ModelFitError("binary model fit requires both classes")
    return X, y, w


def fit_diagnostics(model) -> dict:
    value = getattr(model, "_trademark_fit_diagnostics", None)
    if not isinstance(value, dict) or value.get("converged") is not True \
            or value.get("finite_parameters") is not True \
            or value.get("warning_count") != 0:
        raise ModelFitError("model lacks valid convergence diagnostics")
    return value


def _cfg_id(prefix: str, config: dict) -> str:

    return prefix + ":" + ",".join(f"{k}={config[k]!r}" for k in sorted(config))


def fit_lgbm(
        X, y, w, config, seed, objective="huber", *,
        require_exact_rounds: bool = False,
        fit_context: str | None = None,
        _warning_policy_preinstalled: bool = False):

    import lightgbm as lgb
    if objective not in {"huber", "binary"}:
        raise ModelFitError("LightGBM objective is not frozen")
    if int(seed) not in EXAMPLE_SEEDS:
        raise ModelFitError("LightGBM seed is not frozen")
    X, y, w = _finite_fit_inputs(
        X, y, w, allow_nan_X=True, binary=objective == "binary")
    params = dict(EXAMPLE_FIT_SETTINGS)
    params.update(objective=objective,
                  learning_rate=config["learning_rate"],
                  num_leaves=config["num_leaves"],
                  min_data_in_leaf=config["min_data_in_leaf"],
                  seed=int(seed), bagging_seed=int(seed), feature_fraction_seed=int(seed),
                  data_random_seed=int(seed), verbose=-1)
    n_estimators = params.pop("n_estimators")
    dset = lgb.Dataset(X, label=y, weight=w)
    started = time.perf_counter()
    if _warning_policy_preinstalled:
        booster = lgb.train(
            params, dset, num_boost_round=int(n_estimators))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            booster = lgb.train(
                params, dset, num_boost_round=int(n_estimators))
    elapsed = time.perf_counter() - started
    if elapsed > EXAMPLE_FIT_LIMIT_SECONDS:
        raise ModelFitError(
            f"LightGBM exceeded {EXAMPLE_FIT_LIMIT_SECONDS:.0f}s fit budget")
    if require_exact_rounds \
            and booster.current_iteration() != int(n_estimators):
        raise ModelFitError("LightGBM did not complete every frozen round")
    prediction = np.asarray(
        booster.predict(X, num_iteration=int(n_estimators)), dtype=float)
    if prediction.shape != y.shape or np.any(~np.isfinite(prediction)):
        raise ModelFitError("LightGBM produced invalid training predictions")
    if objective == "binary" and np.any(
            (prediction < 0.0) | (prediction > 1.0)):
        raise ModelFitError("LightGBM produced invalid probabilities")
    booster._trademark_fit_diagnostics = {
        "algorithm": "LightGBM",
        "fit_context": (
            fit_context or (
                f"LightGBM|objective={objective}|seed={int(seed)}"
                f"|config={_cfg_id('lgbm', config)}")),
        "converged": True,
        "iterations": (int(booster.current_iteration()),),
        "max_iterations": int(n_estimators),
        "stopping_criterion": {
            "rule": "exact_frozen_boost_rounds",
            "requested_rounds": int(n_estimators),
        },
        "warning_count": 0,
        "finite_parameters": True,
        "finite_output_status": "FINITE",
        "fit_wall_seconds_limit": repr(EXAMPLE_FIT_LIMIT_SECONDS),
        "seed": int(seed),
        "objective": objective,
        "exact_rounds_required": bool(require_exact_rounds),
        "configuration": {
            "learning_rate": repr(float(config["learning_rate"])),
            "num_leaves": int(config["num_leaves"]),
            "min_data_in_leaf": int(config["min_data_in_leaf"]),
        },
    }
    return booster
