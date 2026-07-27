import pytest

from expert_poker_player.cards import Card, Rank, Suit
from expert_poker_player.uth.card_source import FixedDeck


CARDS = (
    Card(rank=Rank.ACE, suit=Suit.SPADES),
    Card(rank=Rank.KING, suit=Suit.HEARTS),
    Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
)


def test_fixed_deck_draws_cards_in_declared_order() -> None:
    deck = FixedDeck(CARDS)

    assert deck.cards == CARDS
    assert len(deck) == 3

    first_card = deck.draw()
    second_card = deck.draw()
    third_card = deck.draw()

    assert first_card == CARDS[0]
    assert second_card == CARDS[1]
    assert third_card == CARDS[2]

    assert deck.cards == ()
    assert len(deck) == 0


def test_fixed_deck_cards_property_is_immutable_tuple() -> None:
    deck = FixedDeck(CARDS)

    assert isinstance(deck.cards, tuple)


def test_fixed_deck_rejects_non_sequence() -> None:
    with pytest.raises(
        TypeError,
        match="draw_order must be a sequence",
    ):
        FixedDeck(42)  # type: ignore[arg-type]


def test_fixed_deck_rejects_non_card_element() -> None:
    with pytest.raises(
        TypeError,
        match="must contain only Card values",
    ):
        FixedDeck(
            (
                CARDS[0],
                "not a card", # type: ignore
            )
        )  # type: ignore[arg-type]


def test_fixed_deck_rejects_duplicate_cards() -> None:
    with pytest.raises(
        ValueError,
        match="cannot contain duplicate cards",
    ):
        FixedDeck(
            (
                CARDS[0],
                CARDS[0],
            )
        )


def test_fixed_deck_rejects_drawing_from_empty_deck() -> None:
    deck = FixedDeck(())

    with pytest.raises(
        IndexError,
        match="cannot draw a card from an empty fixed deck",
    ):
        deck.draw()