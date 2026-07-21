from fractions import Fraction

import pytest

from expert_poker_player.hands import HandRank, HandValue
from expert_poker_player.uth import RoundOutcome
from expert_poker_player.uth.settlement import (
    dealer_qualifies,
    determine_round_outcome,
    settle_fold,
    settle_showdown,
)


def high_card(
    *values: int,
) -> HandValue:
    return HandValue(
        rank=HandRank.HIGH_CARD,
        tiebreak=tuple(values),
    )


def one_pair(
    pair: int,
    first_kicker: int,
    second_kicker: int,
    third_kicker: int,
) -> HandValue:
    return HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(
            pair,
            first_kicker,
            second_kicker,
            third_kicker,
        ),
    )


def straight(
    high_card_value: int,
) -> HandValue:
    return HandValue(
        rank=HandRank.STRAIGHT,
        tiebreak=(high_card_value,),
    )


@pytest.mark.parametrize(
    ("dealer_hand", "expected"),
    [
        (
            high_card(14, 13, 11, 9, 7),
            False,
        ),
        (
            one_pair(2, 14, 13, 11),
            True,
        ),
        (
            HandValue(
                rank=HandRank.TWO_PAIR,
                tiebreak=(14, 13, 11),
            ),
            True,
        ),
        (
            HandValue(
                rank=HandRank.THREE_OF_A_KIND,
                tiebreak=(14, 13, 11),
            ),
            True,
        ),
        (
            straight(10),
            True,
        ),
        (
            HandValue(
                rank=HandRank.FLUSH,
                tiebreak=(14, 13, 11, 9, 7),
            ),
            True,
        ),
        (
            HandValue(
                rank=HandRank.FULL_HOUSE,
                tiebreak=(14, 13),
            ),
            True,
        ),
        (
            HandValue(
                rank=HandRank.FOUR_OF_A_KIND,
                tiebreak=(14, 13),
            ),
            True,
        ),
        (
            HandValue(
                rank=HandRank.STRAIGHT_FLUSH,
                tiebreak=(14,),
            ),
            True,
        ),
    ],
)
def test_dealer_qualification(
    dealer_hand: HandValue,
    expected: bool,
) -> None:
    assert dealer_qualifies(dealer_hand) is expected


def test_determine_round_outcome_returns_player_win() -> None:
    player_hand = straight(10)
    dealer_hand = one_pair(14, 13, 11, 9)

    assert (
        determine_round_outcome(player_hand, dealer_hand)
        is RoundOutcome.PLAYER_WIN
    )


def test_determine_round_outcome_returns_dealer_win() -> None:
    player_hand = one_pair(13, 14, 11, 9)
    dealer_hand = one_pair(14, 13, 11, 9)

    assert (
        determine_round_outcome(player_hand, dealer_hand)
        is RoundOutcome.DEALER_WIN
    )


def test_determine_round_outcome_returns_push() -> None:
    player_hand = one_pair(14, 13, 11, 9)
    dealer_hand = one_pair(14, 13, 11, 9)

    assert (
        determine_round_outcome(player_hand, dealer_hand)
        is RoundOutcome.PUSH
    )


def test_settle_fold_loses_ante_and_blind() -> None:
    settlement = settle_fold()

    assert settlement.ante.stake == Fraction(1)
    assert settlement.ante.net_profit == Fraction(-1)

    assert settlement.blind.stake == Fraction(1)
    assert settlement.blind.net_profit == Fraction(-1)

    assert settlement.play.stake == Fraction(0)
    assert settlement.play.net_profit == Fraction(0)

    assert settlement.total_staked == Fraction(2)
    assert settlement.total_net_profit == Fraction(-2)
    assert settlement.total_gross_return == Fraction(0)


def test_player_win_with_qualified_dealer() -> None:
    settlement = settle_showdown(
        player_hand=straight(10),
        dealer_hand=one_pair(14, 13, 11, 9),
        play_multiplier=4,
    )

    assert settlement.ante.net_profit == Fraction(1)
    assert settlement.blind.net_profit == Fraction(1)
    assert settlement.play.net_profit == Fraction(4)

    assert settlement.total_staked == Fraction(6)
    assert settlement.total_net_profit == Fraction(6)
    assert settlement.total_gross_return == Fraction(12)


def test_player_win_with_unqualified_dealer() -> None:
    settlement = settle_showdown(
        player_hand=one_pair(2, 14, 13, 11),
        dealer_hand=high_card(14, 13, 11, 9, 7),
        play_multiplier=2,
    )

    assert settlement.ante.net_profit == Fraction(0)
    assert settlement.blind.net_profit == Fraction(0)
    assert settlement.play.net_profit == Fraction(2)

    assert settlement.total_staked == Fraction(4)
    assert settlement.total_net_profit == Fraction(2)
    assert settlement.total_gross_return == Fraction(6)


def test_dealer_win_with_qualified_dealer() -> None:
    settlement = settle_showdown(
        player_hand=one_pair(13, 14, 11, 9),
        dealer_hand=one_pair(14, 13, 11, 9),
        play_multiplier=3,
    )

    assert settlement.ante.net_profit == Fraction(-1)
    assert settlement.blind.net_profit == Fraction(-1)
    assert settlement.play.net_profit == Fraction(-3)

    assert settlement.total_staked == Fraction(5)
    assert settlement.total_net_profit == Fraction(-5)
    assert settlement.total_gross_return == Fraction(0)


def test_dealer_win_without_qualification() -> None:
    settlement = settle_showdown(
        player_hand=high_card(14, 12, 11, 9, 7),
        dealer_hand=high_card(14, 13, 11, 9, 7),
        play_multiplier=1,
    )

    assert settlement.ante.net_profit == Fraction(0)
    assert settlement.blind.net_profit == Fraction(-1)
    assert settlement.play.net_profit == Fraction(-1)

    assert settlement.total_staked == Fraction(3)
    assert settlement.total_net_profit == Fraction(-2)
    assert settlement.total_gross_return == Fraction(1)


def test_push_returns_all_active_wagers() -> None:
    tied_hand = HandValue(
        rank=HandRank.TWO_PAIR,
        tiebreak=(14, 13, 11),
    )

    settlement = settle_showdown(
        player_hand=tied_hand,
        dealer_hand=tied_hand,
        play_multiplier=4,
    )

    assert settlement.ante.net_profit == Fraction(0)
    assert settlement.blind.net_profit == Fraction(0)
    assert settlement.play.net_profit == Fraction(0)

    assert settlement.total_staked == Fraction(6)
    assert settlement.total_net_profit == Fraction(0)
    assert settlement.total_gross_return == Fraction(6)
def test_dealer_qualifies_rejects_invalid_hand_type() -> None:
    with pytest.raises(
        TypeError,
        match="dealer_hand must be an instance of HandValue",
    ):
        dealer_qualifies("not a hand")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("player_hand", "dealer_hand", "expected_message"),
    [
        (
            "not a hand",
            one_pair(14, 13, 11, 9),
            "player_hand must be an instance of HandValue",
        ),
        (
            one_pair(14, 13, 11, 9),
            "not a hand",
            "dealer_hand must be an instance of HandValue",
        ),
    ],
)
def test_determine_round_outcome_rejects_invalid_hand_types(
    player_hand: object,
    dealer_hand: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        determine_round_outcome(
            player_hand=player_hand,  # type: ignore[arg-type]
            dealer_hand=dealer_hand,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "play_multiplier",
    [
        0,
        5,
        -1,
    ],
)
def test_settle_showdown_rejects_invalid_play_multiplier(
    play_multiplier: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="play_multiplier must be one of: 1, 2, 3, 4",
    ):
        settle_showdown(
            player_hand=one_pair(14, 13, 11, 9),
            dealer_hand=one_pair(13, 14, 11, 9),
            play_multiplier=play_multiplier,
        )


@pytest.mark.parametrize(
    "play_multiplier",
    [
        1.0,
        "4",
        True,
        None,
    ],
)
def test_settle_showdown_rejects_non_integer_play_multiplier(
    play_multiplier: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="play_multiplier must be an integer",
    ):
        settle_showdown(
            player_hand=one_pair(14, 13, 11, 9),
            dealer_hand=one_pair(13, 14, 11, 9),
            play_multiplier=play_multiplier,  # type: ignore[arg-type]
        )


def test_settle_showdown_rejects_invalid_player_hand() -> None:
    with pytest.raises(
        TypeError,
        match="player_hand must be an instance of HandValue",
    ):
        settle_showdown(
            player_hand="not a hand",  # type: ignore[arg-type]
            dealer_hand=one_pair(14, 13, 11, 9),
            play_multiplier=4,
        )


def test_settle_showdown_rejects_invalid_dealer_hand() -> None:
    with pytest.raises(
        TypeError,
        match="dealer_hand must be an instance of HandValue",
    ):
        settle_showdown(
            player_hand=one_pair(14, 13, 11, 9),
            dealer_hand="not a hand",  # type: ignore[arg-type]
            play_multiplier=4,
        )