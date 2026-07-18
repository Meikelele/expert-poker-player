import pytest

from expert_poker_player.cards import Deck


def test_deck_has_52_unique_cards() -> None:
    deck = Deck(shuffle=False)

    assert len(deck) == 52
    assert len(set(deck.cards)) == 52


def test_drawing_card_reduces_deck_size() -> None:
    deck = Deck(shuffle=False)

    deck.draw()

    assert len(deck) == 51


def test_same_card_cannot_be_drawn_twice() -> None:
    deck = Deck(shuffle=False)

    drawn_cards = deck.draw_many(52)

    assert len(drawn_cards) == 52
    assert len(set(drawn_cards)) == 52
    assert len(deck) == 0


def test_drawing_from_empty_deck_raises_error() -> None:
    deck = Deck(shuffle=False)
    deck.draw_many(52)

    with pytest.raises(IndexError):
        deck.draw()


def test_cannot_draw_more_cards_than_remain() -> None:
    deck = Deck(shuffle=False)

    with pytest.raises(ValueError):
        deck.draw_many(53)


def test_same_seed_produces_same_card_order() -> None:
    first_deck = Deck(seed=42)
    second_deck = Deck(seed=42)

    assert first_deck.cards == second_deck.cards