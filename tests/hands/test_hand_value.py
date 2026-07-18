import pytest

from expert_poker_player.cards import Rank
from expert_poker_player.hands import HandRank, HandValue


def test_stronger_category_wins_even_with_shorter_tiebreak() -> None:
    three_of_a_kind = HandValue(
        rank=HandRank.THREE_OF_A_KIND,
        tiebreak=(
            Rank.TWO.value,
            Rank.FOUR.value,
            Rank.THREE.value,
        ),
    )

    pair_of_aces = HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(
            Rank.ACE.value,
            Rank.KING.value,
            Rank.QUEEN.value,
            Rank.JACK.value,
        ),
    )

    assert three_of_a_kind > pair_of_aces


def test_higher_pair_wins() -> None:
    pair_of_aces = HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(14, 13, 9, 4),
    )

    pair_of_kings = HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(13, 14, 12, 11),
    )

    assert pair_of_aces > pair_of_kings


def test_kicker_breaks_tie_between_equal_pairs() -> None:
    pair_of_aces_with_king = HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(14, 13, 9, 4),
    )

    pair_of_aces_with_queen = HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(14, 12, 11, 10),
    )

    assert pair_of_aces_with_king > pair_of_aces_with_queen


def test_lower_kicker_is_checked_when_previous_values_are_equal() -> None:
    first_hand = HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(14, 13, 9, 5),
    )

    second_hand = HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(14, 13, 9, 4),
    )

    assert first_hand > second_hand


def test_two_identical_hand_values_are_equal() -> None:
    first_hand = HandValue(
        rank=HandRank.TWO_PAIR,
        tiebreak=(14, 13, 7),
    )

    second_hand = HandValue(
        rank=HandRank.TWO_PAIR,
        tiebreak=(14, 13, 7),
    )

    assert first_hand == second_hand
    assert not first_hand != second_hand


def test_higher_full_house_is_determined_by_three_of_a_kind() -> None:
    kings_full_of_twos = HandValue(
        rank=HandRank.FULL_HOUSE,
        tiebreak=(13, 2),
    )

    queens_full_of_aces = HandValue(
        rank=HandRank.FULL_HOUSE,
        tiebreak=(12, 14),
    )

    assert kings_full_of_twos > queens_full_of_aces


def test_full_house_pair_breaks_tie_when_triplets_are_equal() -> None:
    aces_full_of_kings = HandValue(
        rank=HandRank.FULL_HOUSE,
        tiebreak=(14, 13),
    )

    aces_full_of_queens = HandValue(
        rank=HandRank.FULL_HOUSE,
        tiebreak=(14, 12),
    )

    assert aces_full_of_kings > aces_full_of_queens


def test_royal_flush_is_detected() -> None:
    royal_flush = HandValue(
        rank=HandRank.STRAIGHT_FLUSH,
        tiebreak=(Rank.ACE.value,),
    )

    assert royal_flush.is_royal_flush


def test_king_high_straight_flush_is_not_royal_flush() -> None:
    king_high_straight_flush = HandValue(
        rank=HandRank.STRAIGHT_FLUSH,
        tiebreak=(Rank.KING.value,),
    )

    assert not king_high_straight_flush.is_royal_flush


def test_royal_flush_beats_lower_straight_flush() -> None:
    royal_flush = HandValue(
        rank=HandRank.STRAIGHT_FLUSH,
        tiebreak=(Rank.ACE.value,),
    )

    king_high_straight_flush = HandValue(
        rank=HandRank.STRAIGHT_FLUSH,
        tiebreak=(Rank.KING.value,),
    )

    assert royal_flush > king_high_straight_flush


def test_invalid_tiebreak_length_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="ONE_PAIR requires exactly 4 tiebreak values",
    ):
        HandValue(
            rank=HandRank.ONE_PAIR,
            tiebreak=(14, 13),
        )


def test_invalid_card_value_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="tiebreak values must be between 2 and 14",
    ):
        HandValue(
            rank=HandRank.HIGH_CARD,
            tiebreak=(15, 13, 9, 5, 2),
        )


def test_straight_cannot_have_high_card_lower_than_five() -> None:
    with pytest.raises(
        ValueError,
        match="highest card of a straight",
    ):
        HandValue(
            rank=HandRank.STRAIGHT,
            tiebreak=(4,),
        )


def test_tiebreak_must_be_a_tuple() -> None:
    with pytest.raises(
        TypeError,
        match="tiebreak must be a tuple",
    ):
        HandValue(
            rank=HandRank.STRAIGHT,
            tiebreak=[10],  # type: ignore[arg-type]
        )


def test_rank_must_be_hand_rank() -> None:
    with pytest.raises(
        TypeError,
        match="rank must be an instance of HandRank",
    ):
        HandValue(
            rank=1,  # type: ignore[arg-type]
            tiebreak=(14, 13, 9, 4),
        )

def test_tiebreak_values_must_be_integers() -> None:
    with pytest.raises(
        TypeError,
        match="all tiebreak values must be integers",
    ):
        HandValue(
            rank=HandRank.STRAIGHT,
            tiebreak=("10",),  # type: ignore[arg-type]
        )

def test_hand_value_is_not_equal_to_object_of_different_type() -> None:
    hand = HandValue(
        rank=HandRank.STRAIGHT,
        tiebreak=(Rank.TEN.value,),
    )

    assert hand != object()

def test_comparison_with_unsupported_type_returns_not_implemented() -> None:
    hand = HandValue(
        rank=HandRank.STRAIGHT,
        tiebreak=(Rank.TEN.value,),
    )

    assert hand.__lt__(object()) is NotImplemented

def test_ordering_with_unsupported_type_raises_type_error() -> None:
    hand = HandValue(
        rank=HandRank.STRAIGHT,
        tiebreak=(Rank.TEN.value,),
    )

    with pytest.raises(TypeError):
        _ = hand < object()

def test_equal_hand_values_have_equal_hashes() -> None:
    first_hand = HandValue(
        rank=HandRank.TWO_PAIR,
        tiebreak=(14, 13, 7),
    )

    second_hand = HandValue(
        rank=HandRank.TWO_PAIR,
        tiebreak=(14, 13, 7),
    )

    assert first_hand == second_hand
    assert hash(first_hand) == hash(second_hand)
    assert len({first_hand, second_hand}) == 1