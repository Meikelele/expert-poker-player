import pytest

from expert_poker_player.cards import Card, Rank, Suit
from expert_poker_player.hands import (
    HandRank,
    evaluate_best_hand,
)


def make_cards(
    *specifications: tuple[Rank, Suit],
) -> list[Card]:
    """Tworzy karty na podstawie par: wartość i kolor."""

    return [
        Card(rank=rank, suit=suit)
        for rank, suit in specifications
    ]


def test_selects_royal_flush_from_seven_cards() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.SPADES),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.SPADES),
        (Rank.TEN, Suit.SPADES),
        (Rank.TWO, Suit.HEARTS),
        (Rank.THREE, Suit.DIAMONDS),
    )

    result = evaluate_best_hand(cards)

    assert result.value.rank is HandRank.STRAIGHT_FLUSH
    assert result.value.tiebreak == (Rank.ACE.value,)
    assert result.value.is_royal_flush
    assert len(result.cards) == 5


def test_selects_four_of_a_kind_from_seven_cards() -> None:
    cards = make_cards(
        (Rank.NINE, Suit.SPADES),
        (Rank.NINE, Suit.HEARTS),
        (Rank.NINE, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.TWO, Suit.DIAMONDS),
    )

    result = evaluate_best_hand(cards)

    assert result.value.rank is HandRank.FOUR_OF_A_KIND
    assert result.value.tiebreak == (
        Rank.NINE.value,
        Rank.ACE.value,
    )


def test_selects_full_house_from_two_triplets() -> None:
    cards = make_cards(
        (Rank.KING, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.QUEEN, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.TWO, Suit.CLUBS),
    )

    result = evaluate_best_hand(cards)

    assert result.value.rank is HandRank.FULL_HOUSE
    assert result.value.tiebreak == (
        Rank.KING.value,
        Rank.QUEEN.value,
    )


def test_selects_best_five_kickers_for_flush() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.HEARTS),
        (Rank.JACK, Suit.HEARTS),
        (Rank.NINE, Suit.HEARTS),
        (Rank.SIX, Suit.HEARTS),
        (Rank.THREE, Suit.HEARTS),
        (Rank.TWO, Suit.CLUBS),
    )

    result = evaluate_best_hand(cards)

    assert result.value.rank is HandRank.FLUSH
    assert result.value.tiebreak == (
        Rank.ACE.value,
        Rank.KING.value,
        Rank.JACK.value,
        Rank.NINE.value,
        Rank.SIX.value,
    )


def test_selects_wheel_straight_from_seven_cards() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.TWO, Suit.HEARTS),
        (Rank.THREE, Suit.DIAMONDS),
        (Rank.FOUR, Suit.CLUBS),
        (Rank.FIVE, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
    )

    result = evaluate_best_hand(cards)

    assert result.value.rank is HandRank.STRAIGHT
    assert result.value.tiebreak == (Rank.FIVE.value,)


def test_five_cards_are_also_supported() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.FOUR, Suit.SPADES),
    )

    result = evaluate_best_hand(cards)

    assert result.value.rank is HandRank.ONE_PAIR
    assert result.value.tiebreak == (
        Rank.ACE.value,
        Rank.KING.value,
        Rank.NINE.value,
        Rank.FOUR.value,
    )
    assert set(result.cards) == set(cards)


def test_rejects_less_than_five_cards() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.JACK, Suit.CLUBS),
    )

    with pytest.raises(
        ValueError,
        match="requires between 5 and 7 cards",
    ):
        evaluate_best_hand(cards)


def test_rejects_more_than_seven_cards() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.JACK, Suit.CLUBS),
        (Rank.TEN, Suit.SPADES),
        (Rank.NINE, Suit.HEARTS),
        (Rank.EIGHT, Suit.DIAMONDS),
        (Rank.SEVEN, Suit.CLUBS),
    )

    with pytest.raises(
        ValueError,
        match="requires between 5 and 7 cards",
    ):
        evaluate_best_hand(cards)


def test_rejects_duplicate_available_cards() -> None:
    ace_of_spades = Card(
        rank=Rank.ACE,
        suit=Suit.SPADES,
    )

    cards = [
        ace_of_spades,
        ace_of_spades,
        Card(rank=Rank.KING, suit=Suit.HEARTS),
        Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
        Card(rank=Rank.JACK, suit=Suit.CLUBS),
        Card(rank=Rank.TEN, suit=Suit.SPADES),
        Card(rank=Rank.NINE, suit=Suit.HEARTS),
    ]

    with pytest.raises(
        ValueError,
        match="cannot contain duplicates",
    ):
        evaluate_best_hand(cards)