import pytest

from expert_poker_player.cards import Card, Rank, Suit
from expert_poker_player.hands import (
    HandRank,
    HandValue,
    evaluate_five_card_hand,
)


def make_cards(
    *specifications: tuple[Rank, Suit],
) -> list[Card]:
    """Tworzy karty na podstawie par: wartość i kolor."""

    return [
        Card(rank=rank, suit=suit)
        for rank, suit in specifications
    ]


def test_evaluates_high_card() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.JACK, Suit.HEARTS),
        (Rank.NINE, Suit.DIAMONDS),
        (Rank.SIX, Suit.CLUBS),
        (Rank.THREE, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.HIGH_CARD,
        tiebreak=(14, 11, 9, 6, 3),
    )


def test_evaluates_one_pair() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.FOUR, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(14, 13, 9, 4),
    )


def test_evaluates_two_pair() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.KING, Suit.CLUBS),
        (Rank.SEVEN, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.TWO_PAIR,
        tiebreak=(14, 13, 7),
    )


def test_evaluates_three_of_a_kind() -> None:
    cards = make_cards(
        (Rank.QUEEN, Suit.SPADES),
        (Rank.QUEEN, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.SEVEN, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.THREE_OF_A_KIND,
        tiebreak=(12, 14, 7),
    )


def test_evaluates_straight() -> None:
    cards = make_cards(
        (Rank.TEN, Suit.SPADES),
        (Rank.NINE, Suit.HEARTS),
        (Rank.EIGHT, Suit.DIAMONDS),
        (Rank.SEVEN, Suit.CLUBS),
        (Rank.SIX, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.STRAIGHT,
        tiebreak=(10,),
    )


def test_evaluates_wheel_straight() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.TWO, Suit.HEARTS),
        (Rank.THREE, Suit.DIAMONDS),
        (Rank.FOUR, Suit.CLUBS),
        (Rank.FIVE, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.STRAIGHT,
        tiebreak=(5,),
    )


def test_evaluates_flush() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.HEARTS),
        (Rank.JACK, Suit.HEARTS),
        (Rank.NINE, Suit.HEARTS),
        (Rank.SIX, Suit.HEARTS),
        (Rank.THREE, Suit.HEARTS),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.FLUSH,
        tiebreak=(14, 11, 9, 6, 3),
    )


def test_evaluates_full_house() -> None:
    cards = make_cards(
        (Rank.KING, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.EIGHT, Suit.CLUBS),
        (Rank.EIGHT, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.FULL_HOUSE,
        tiebreak=(13, 8),
    )


def test_evaluates_four_of_a_kind() -> None:
    cards = make_cards(
        (Rank.NINE, Suit.SPADES),
        (Rank.NINE, Suit.HEARTS),
        (Rank.NINE, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.ACE, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.FOUR_OF_A_KIND,
        tiebreak=(9, 14),
    )


def test_evaluates_straight_flush() -> None:
    cards = make_cards(
        (Rank.KING, Suit.SPADES),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.SPADES),
        (Rank.TEN, Suit.SPADES),
        (Rank.NINE, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result == HandValue(
        rank=HandRank.STRAIGHT_FLUSH,
        tiebreak=(13,),
    )


def test_evaluates_royal_flush_as_ace_high_straight_flush() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.SPADES),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.SPADES),
        (Rank.TEN, Suit.SPADES),
    )

    result = evaluate_five_card_hand(cards)

    assert result.rank is HandRank.STRAIGHT_FLUSH
    assert result.tiebreak == (Rank.ACE.value,)
    assert result.is_royal_flush


def test_rejects_less_than_five_cards() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.SPADES),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.SPADES),
    )

    with pytest.raises(
        ValueError,
        match="must contain exactly 5 cards",
    ):
        evaluate_five_card_hand(cards)


def test_rejects_more_than_five_cards() -> None:
    cards = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.SPADES),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.SPADES),
        (Rank.TEN, Suit.SPADES),
        (Rank.NINE, Suit.SPADES),
    )

    with pytest.raises(
        ValueError,
        match="must contain exactly 5 cards",
    ):
        evaluate_five_card_hand(cards)


def test_rejects_duplicate_cards() -> None:
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
    ]

    with pytest.raises(
        ValueError,
        match="cannot contain duplicate cards",
    ):
        evaluate_five_card_hand(cards)

def test_rejects_non_card_element() -> None:
    cards = [ # type: ignore
        Card(rank=Rank.ACE, suit=Suit.SPADES),
        Card(rank=Rank.KING, suit=Suit.HEARTS),
        Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
        Card(rank=Rank.JACK, suit=Suit.CLUBS),
        "not a card",
    ]

    with pytest.raises(
        TypeError,
        match="all elements must be instances of Card",
    ):
        evaluate_five_card_hand(cards)  # type: ignore[arg-type]