from fractions import Fraction

import pytest

from expert_poker_player.uth import WagerOutcome
from expert_poker_player.uth.wagers import (
    Settlement,
    WagerSettlement,
)


@pytest.mark.parametrize(
    (
        "stake",
        "net_profit",
        "expected_outcome",
        "expected_gross_return",
    ),
    [
        (
            Fraction(1),
            Fraction(1),
            WagerOutcome.WIN,
            Fraction(2),
        ),
        (
            Fraction(1),
            Fraction(3, 2),
            WagerOutcome.WIN,
            Fraction(5, 2),
        ),
        (
            Fraction(1),
            Fraction(0),
            WagerOutcome.PUSH,
            Fraction(1),
        ),
        (
            Fraction(1),
            Fraction(-1),
            WagerOutcome.LOSS,
            Fraction(0),
        ),
        (
            Fraction(0),
            Fraction(0),
            WagerOutcome.NOT_PLACED,
            Fraction(0),
        ),
    ],
)
def test_wager_settlement_derives_outcome_and_gross_return(
    stake: Fraction,
    net_profit: Fraction,
    expected_outcome: WagerOutcome,
    expected_gross_return: Fraction,
) -> None:
    settlement = WagerSettlement(
        stake=stake,
        net_profit=net_profit,
    )

    assert settlement.outcome is expected_outcome
    assert settlement.gross_return == expected_gross_return


def test_wager_settlement_rejects_non_fraction_stake() -> None:
    with pytest.raises(
        TypeError,
        match="stake must be an instance of Fraction",
    ):
        WagerSettlement(
            stake=1,  # type: ignore[arg-type]
            net_profit=Fraction(0),
        )


def test_wager_settlement_rejects_non_fraction_net_profit() -> None:
    with pytest.raises(
        TypeError,
        match="net_profit must be an instance of Fraction",
    ):
        WagerSettlement(
            stake=Fraction(1),
            net_profit=0,  # type: ignore[arg-type]
        )


def test_wager_settlement_rejects_negative_stake() -> None:
    with pytest.raises(
        ValueError,
        match="stake cannot be negative",
    ):
        WagerSettlement(
            stake=Fraction(-1),
            net_profit=Fraction(0),
        )


@pytest.mark.parametrize(
    "net_profit",
    [
        Fraction(1),
        Fraction(-1),
    ],
)
def test_zero_stake_requires_zero_net_profit(
    net_profit: Fraction,
) -> None:
    with pytest.raises(
        ValueError,
        match="zero stake must have zero net profit",
    ):
        WagerSettlement(
            stake=Fraction(0),
            net_profit=net_profit,
        )


@pytest.mark.parametrize(
    "net_profit",
    [
        Fraction(-1, 2),
        Fraction(-2),
    ],
)
def test_losing_wager_must_lose_full_stake(
    net_profit: Fraction,
) -> None:
    with pytest.raises(
        ValueError,
        match="must lose exactly its full stake",
    ):
        WagerSettlement(
            stake=Fraction(1),
            net_profit=net_profit,
        )


def test_settlement_calculates_totals() -> None:
    settlement = Settlement(
        ante=WagerSettlement(
            stake=Fraction(1),
            net_profit=Fraction(1),
        ),
        blind=WagerSettlement(
            stake=Fraction(1),
            net_profit=Fraction(3, 2),
        ),
        play=WagerSettlement(
            stake=Fraction(4),
            net_profit=Fraction(-4),
        ),
    )

    assert settlement.total_staked == Fraction(6)
    assert settlement.total_net_profit == Fraction(-3, 2)
    assert settlement.total_gross_return == Fraction(9, 2)


@pytest.mark.parametrize(
    ("field_name", "expected_message"),
    [
        (
            "ante",
            "ante must be an instance of WagerSettlement",
        ),
        (
            "blind",
            "blind must be an instance of WagerSettlement",
        ),
        (
            "play",
            "play must be an instance of WagerSettlement",
        ),
    ],
)
def test_settlement_rejects_invalid_wager_type(
    field_name: str,
    expected_message: str,
) -> None:
    valid_wager = WagerSettlement(
        stake=Fraction(1),
        net_profit=Fraction(0),
    )

    values: dict[str, object] = {
        "ante": valid_wager,
        "blind": valid_wager,
        "play": valid_wager,
    }
    values[field_name] = "invalid wager"

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        Settlement(
            ante=values["ante"],  # type: ignore[arg-type]
            blind=values["blind"],  # type: ignore[arg-type]
            play=values["play"],  # type: ignore[arg-type]
        )