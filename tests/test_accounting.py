from dataclasses import asdict
import json

import pytest

from selected.accounting import (BookExample, Receivable, missing_opening_quotes, money,
    recognize_cash_merger, settle_cash_receivables)
from selected.common import IntegrityError


def fixture():
    return BookExample(money(1000), {"SYNTH-OLD": 10}), dict(
        session="2020-06-02", security_id="SYNTH-OLD", event_id="SYNTH-MERGER",
        cash_per_share_micro=money(60), pay_session="2020-06-04")


def test_merger_claim_is_valued_before_payment_but_is_not_spendable_cash():
    state, event = fixture()
    original = asdict(state)
    before_value = state.value({"SYNTH-OLD": money(60)})
    after, canceled = recognize_cash_merger(state, **event)
    assert asdict(state) == original
    assert after.positions == {} and canceled == {"SYNTH-OLD"}
    assert after.cash == money(1000)
    assert after.receivables["SYNTH-MERGER:cash"].amount == money(600)
    assert after.value({}) == before_value
    early = settle_cash_receivables(after, "2020-06-03")
    assert early == after
    paid = settle_cash_receivables(early, "2020-06-04")
    assert paid.cash == money(1600) and paid.receivables == {}
    assert paid.value({}) == before_value


def test_replay_and_serialized_checkpoint_produce_the_same_payment():
    state, event = fixture()
    first, _ = recognize_cash_merger(state, **event)
    replay, _ = recognize_cash_merger(state, **event)
    assert replay == first
    encoded = json.loads(json.dumps(asdict(first)))
    restored = BookExample(encoded["cash"], encoded["positions"],
        {key: Receivable(**row) for key, row in encoded["receivables"].items()})
    assert settle_cash_receivables(first, "2020-06-04") == settle_cash_receivables(restored, "2020-06-04")


def test_reapplying_the_event_cannot_credit_an_already_converted_position():
    state, event = fixture()
    after, _ = recognize_cash_merger(state, **event)
    repeated, _ = recognize_cash_merger(after, **event)
    assert repeated == after


@pytest.mark.parametrize("changes", [
    {"cash_per_share_micro": 0}, {"cash_per_share_micro": True},
    {"pay_session": "2020-06-01"}, {"pay_session": "2020-06-06"}, {"pay_session": None},
])
def test_bad_terms_fail_without_mutating_the_original_account(changes):
    state, event = fixture()
    original = asdict(state)
    with pytest.raises(IntegrityError):
        recognize_cash_merger(state, **{**event, **changes})
    assert asdict(state) == original


def test_reused_receivable_identifier_is_rejected():
    state, event = fixture()
    state.receivables["SYNTH-MERGER:cash"] = Receivable("SYNTH-OLD", "2020-06-04", money(600))
    original = asdict(state)
    with pytest.raises(IntegrityError, match="receivable id reused"):
        recognize_cash_merger(state, **event)
    assert asdict(state) == original


def test_overdue_claim_cannot_silently_roll_into_a_later_checkpoint():
    state, event = fixture()
    after, _ = recognize_cash_merger(state, **event)
    with pytest.raises(IntegrityError, match="overdue receivable"):
        settle_cash_receivables(after, "2020-06-05")


def test_zero_volume_or_missing_open_is_not_an_executable_held_quote():
    state, _ = fixture()
    for opens in ({}, {"SYNTH-OLD": (money(60), 0)}, {"SYNTH-OLD": (0, 1000)}):
        assert missing_opening_quotes(state, opens) == ["SYNTH-OLD"]
    assert missing_opening_quotes(state, {"SYNTH-OLD": (money(60), 1000)}) == []
