from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_EVEN
import datetime as dt
import functools
from types import SimpleNamespace

import pandas as pd
import pandas_market_calendars as mcal

from selected.common import IntegrityError

MICRO = 1_000_000


@functools.lru_cache(maxsize=32)
def _schedule(start_year: int, end_year: int) -> pd.DatetimeIndex:
    cal = mcal.get_calendar("XNYS")
    sched = cal.schedule(start_date=f"{start_year}-01-01", end_date=f"{end_year}-12-31")
    return pd.DatetimeIndex(sched.index)


def _sessions_for(date: dt.date) -> pd.DatetimeIndex:
    return _schedule(date.year - 1, date.year + 1)


def is_trading_day(date: dt.date) -> bool:
    return pd.Timestamp(date) in _sessions_for(date)


def money(value) -> int:
    value = Decimal(str(value))
    if not value.is_finite():
        raise IntegrityError("nonfinite money")
    return int((value * MICRO).to_integral_value(rounding=ROUND_HALF_EVEN))


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise IntegrityError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Receivable:
    security_id: str
    pay_session: str
    amount: int


def _pay_session(terms, key, session, *, required=False):
    value = terms.get(key, None if required else session)
    if not value or value < session or not is_trading_day(date.fromisoformat(value)):
        raise IntegrityError(f"{key} must be a trading session at or after the event")
    return value


@dataclass
class BookExample:
    cash: int
    positions: dict[str, int] = field(default_factory=dict)
    receivables: dict[str, Receivable] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.cash) is not int or self.cash < 0:
            raise IntegrityError("example cash must be nonnegative integer microdollars")
        for sid, quantity in self.positions.items():
            if not isinstance(sid, str) or not sid:
                raise IntegrityError("example holding needs an identity")
            positive_int(quantity, "example whole shares")

    def value(self, prices):
        return self.cash + sum(row.amount for row in self.receivables.values()) + sum(
            quantity * prices[sid] for sid, quantity in self.positions.items())


def recognize_cash_merger(state, *, session, security_id, event_id,
                          cash_per_share_micro, pay_session):
    if not event_id or not security_id or not is_trading_day(date.fromisoformat(session)):
        raise IntegrityError("example event needs identifiers and a trading session")
    out = deepcopy(state)
    sid = security_id
    terms = {"cash_per_share_micro": cash_per_share_micro, "pay_session": pay_session}
    action = SimpleNamespace(event_id=event_id)
    owned = out.positions.get(sid, 0)
    canceled = set()
    def credit(key, sid, amount, pay):
        positive_int(amount, "receivable")
        if key in out.receivables:
            raise IntegrityError("receivable id reused")
        out.receivables[key] = Receivable(sid, pay, amount)
    amount = positive_int(terms.get("cash_per_share_micro"), "cash_per_share_micro")
    if owned:
        out.positions.pop(sid)
        credit(action.event_id+":cash", sid, owned*amount, _pay_session(terms, "pay_session", session, required=True))
    canceled.add(sid)
    return out, canceled


def settle_cash_receivables(state, session):
    out = deepcopy(state)
    estimated = set()
    if any(row.pay_session < session for row in out.receivables.values()):
        raise IntegrityError("overdue receivable; calendar gap")
    for key, row in list(out.receivables.items()):
        if row.pay_session == session and key not in estimated:
            out.cash += row.amount
            del out.receivables[key]
    return out


def missing_opening_quotes(state, opens):
    out = state
    marks = {sid: row[0] for sid, row in opens.items() if row[0] > 0 and row[1] > 0}
    missing_held = sorted(sid for sid in out.positions if sid not in marks)
    return missing_held
