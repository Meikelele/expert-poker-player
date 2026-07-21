from fractions import Fraction

import pytest

from expert_poker_player.cards import Rank
from expert_poker_player.hands import HandRank, HandValue
from expert_poker_player.uth.paytables import blind_profit


@pytest.mark.parametrize(
    ("hand_value", "expected_profit"),
    [
        (
            HandValue(
                rank=HandRank.HIGH_CARD,
                tiebreak=(14, 13, 11, 9, 7),
            ),
            Fraction(0),
        ),
        (
            HandValue(
                rank=HandRank.ONE_PAIR,
                tiebreak=(14, 13, 11, 9),
            ),
            Fraction(0),
        ),
        (
            HandValue(
                rank=HandRank.TWO_PAIR,
                tiebreak=(14, 13, 11),
            ),
            Fraction(0),
        ),
        (
            HandValue(
                rank=HandRank.THREE_OF_A_KIND,
                tiebreak=(14, 13, 11),
            ),
            Fraction(0),
        ),
        (
            HandValue(
                rank=HandRank.STRAIGHT,
                tiebreak=(Rank.TEN.value,),
            ),
            Fraction(1),
        ),
        (
            HandValue(
                rank=HandRank.FLUSH,
                tiebreak=(14, 13, 11, 9, 7),
            ),
            Fraction(3, 2),
        ),
        (
            HandValue(
                rank=HandRank.FULL_HOUSE,
                tiebreak=(14, 13),
            ),
            Fraction(3),
        ),
        (
            HandValue(
                rank=HandRank.FOUR_OF_A_KIND,
                tiebreak=(14, 13),
            ),
            Fraction(10),
        ),
        (
            HandValue(
                rank=HandRank.STRAIGHT_FLUSH,
                tiebreak=(Rank.KING.value,),
            ),
            Fraction(50),
        ),
        (
            HandValue(
                rank=HandRank.STRAIGHT_FLUSH,
                tiebreak=(Rank.ACE.value,),
            ),
            Fraction(500),
        ),
    ],
)
def test_blind_profit_matches_paytable(
    hand_value: HandValue,
    expected_profit: Fraction,
) -> None:
    assert blind_profit(hand_value) == expected_profit


def test_blind_profit_rejects_invalid_hand_type() -> None:
    with pytest.raises(
        TypeError,
        match="player_hand must be an instance of HandValue",
    ):
        blind_profit("not a hand")  # type: ignore[arg-type]