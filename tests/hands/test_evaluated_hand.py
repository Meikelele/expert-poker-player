import pytest

from expert_poker_player.cards import Card, Rank, Suit
from expert_poker_player.hands import EvaluatedHand, HandRank, HandValue


def make_valid_cards() -> tuple[Card, Card, Card, Card, Card]:
    return (
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.NINE, Suit.SPADES),
    )


def test_evaluated_hand_accepts_valid_data() -> None:
    value = HandValue(
        rank=HandRank.HIGH_CARD,
        tiebreak=(14, 13, 12, 11, 9),
    )

    result = EvaluatedHand(
        value=value,
        cards=make_valid_cards(),
    )

    assert result.value == value
    assert result.cards == make_valid_cards()


def test_evaluated_hand_rejects_wrong_number_of_cards() -> None:
    value = HandValue(
        rank=HandRank.HIGH_CARD,
        tiebreak=(14, 13, 12, 11, 9),
    )

    with pytest.raises(
        ValueError,
        match="must contain exactly 5 cards",
    ):
        EvaluatedHand(
            value=value,
            cards=make_valid_cards()[:4],  # type: ignore[arg-type]
        )


def test_evaluated_hand_rejects_duplicate_cards() -> None:
    value = HandValue(
        rank=HandRank.ONE_PAIR,
        tiebreak=(14, 13, 12, 11),
    )

    ace_of_spades = Card(Rank.ACE, Suit.SPADES)

    with pytest.raises(
        ValueError,
        match="cannot contain duplicate cards",
    ):
        EvaluatedHand(
            value=value,
            cards=(
                ace_of_spades,
                ace_of_spades,
                Card(Rank.KING, Suit.HEARTS),
                Card(Rank.QUEEN, Suit.DIAMONDS),
                Card(Rank.JACK, Suit.CLUBS),
            ),
        )

def test_evaluated_hand_rejects_invalid_value_type() -> None:
    with pytest.raises(
        TypeError,
        match="value must be an instance of HandValue",
    ):
        EvaluatedHand(
            value="not a hand value",  # type: ignore[arg-type]
            cards=make_valid_cards(),
        )

def test_evaluated_hand_rejects_cards_that_are_not_tuple() -> None:
    value = HandValue(
        rank=HandRank.HIGH_CARD,
        tiebreak=(14, 13, 12, 11, 9),
    )

    with pytest.raises(
        TypeError,
        match="cards must be a tuple",
    ):
        EvaluatedHand(
            value=value,
            cards=list(make_valid_cards()),  # type: ignore[arg-type]
        )

def test_evaluated_hand_rejects_non_card_element() -> None:
    value = HandValue(
        rank=HandRank.HIGH_CARD,
        tiebreak=(14, 13, 12, 11, 9),
    )

    invalid_cards = (
        Card(rank=Rank.ACE, suit=Suit.SPADES),
        Card(rank=Rank.KING, suit=Suit.HEARTS),
        Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
        Card(rank=Rank.JACK, suit=Suit.CLUBS),
        "not a card",
    )

    with pytest.raises(
        TypeError,
        match="all elements of cards must be instances of Card",
    ):
        EvaluatedHand(
            value=value,
            cards=invalid_cards,  # type: ignore[arg-type]
        )
