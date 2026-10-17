# TradeMark V3

**Python · LightGBM · CVXPY · Historical modeling and account replay**

TradeMark V3 is my private quantitative research project. My work covers historical stock-data modeling, constrained portfolio allocation, and replaying cash and holdings through corporate actions. This repository presents three implementation cases from that work.

**Only selected excerpts of the larger codebase are public.** The complete system, market data, trained models, private experiment settings, and strategy configuration are not included.

## Three engineering decisions

### 1. A corrected label must not rewrite an earlier model fit

An outcome can mature on one date, become available later, and subsequently be corrected or invalidated. The outcome ledger keeps these dates and versions separate. A fit sees the latest version available at its historical cutoff; invalid evidence already visible at that cutoff raises an error.

The public tests add a deliberately invalid future revision and verify that the earlier snapshot and the resulting LightGBM model remain identical. They also exercise finite-input checks, ambiguous revisions, and deterministic fitting.

[Implementation: outcome ledger and LightGBM fitter](selected/modeling.py) · [Tests](tests/test_modeling.py)

### 2. Portfolio limits must stay aligned with the assets being optimized

The constraint excerpt assembles sparse exposure rows and attaches them to the selected CVXPY variables without changing their asset mapping. Cached geometry is immutable and is rebuilt when the constraint rows change. Rows with no exposure are omitted from the solver when they are already satisfied.

One synthetic case has a 25% individual limit and a 50% group limit. Its computed allocation matches a hand-solved answer. Other tests cover a geared instrument whose exposure is three times its capital weight, reordered asset columns, and empty exposure rows.

[Implementation: sparse constraints](selected/constraints.py) · [Tests](tests/test_constraints.py)

### 3. A merger entitlement is not immediately spendable cash

The cash-merger excerpt removes the old shares and records a dated receivable. Cash increases only when that receivable is paid. Amounts use integer microdollars, and invalid terms leave the original account unchanged.

The example starts with $1,000 cash and ten shares worth $60 each. A $60-per-share merger replaces the shares with a $600 claim while cash remains $1,000. Payment later brings cash to $1,600 without changing total account value. Tests also check replay, serialization, duplicate claim identifiers, overdue claims, and missing executable opening prices.

[Implementation: accounting excerpts](selected/accounting.py) · [Tests](tests/test_accounting.py)

## Project scale and public verification

| Evidence | Scope |
| --- | --- |
| 4,625,071 eligible stock observations across 1,511 trading sessions | Saved private feature inventory; dataset coverage, not the row count of every model fit |
| 381 account-replay sessions, January 4, 2016–July 7, 2017 | An earlier private accounting milestone, separate from LightGBM performance evaluation |
| 32 passing synthetic tests | Verification of the excerpts in this repository on Windows with Python 3.13 |

The [private-project summary](evidence/project-summary.json) explains the scale measurements and the checkpoint's limits. The [public verification record](evidence/verification.json) lists the tests and example results. The underlying private reports and data are withheld, so the scale figures cannot be independently reproduced from this selection.

## How the excerpts were prepared

The [selection record](evidence/code-selection.json) maps the excerpts to their original modules and functions. Selected implementation logic was retained, with explanatory code comments and docstrings removed. Small public adapters isolate the examples from private dependencies; the tests and synthetic fixtures were prepared for this selection.

The LightGBM fitter uses clearly named `EXAMPLE_*` settings in place of private seeds and hyperparameters. The allocation tests use an illustrative least-squares objective. The accounting example handles a cash merger and its receivable; it does not include the full account engine or identify exchange halts.

Feature and label construction, the complete optimization objective, risk/cost integration, solver repair logic, full trade execution, and the complete account replay remain private. These excerpts demonstrate specific implementation choices and tested behavior. They do not reproduce a complete trading strategy or establish investment performance. Fresh environments and other operating systems have not been verified.
