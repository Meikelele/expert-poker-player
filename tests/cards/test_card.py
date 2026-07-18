import pytest

from expert_poker_player.cards import Card, Rank, Suit


def test_card_contains_rank_and_suit() -> None:
    card = Card(rank=Rank.ACE, suit=Suit.SPADES)

    assert card.rank is Rank.ACE
    assert card.suit is Suit.SPADES


def test_card_has_readable_string_representation() -> None:
    card = Card(rank=Rank.ACE, suit=Suit.SPADES)

    assert str(card) == "A♠"


def test_card_rejects_invalid_rank() -> None:
    with pytest.raises(TypeError):
        Card(rank=14, suit=Suit.SPADES) # type: ignore


def test_card_rejects_invalid_suit() -> None:
    with pytest.raises(TypeError):
        Card(rank=Rank.ACE, suit="spades") # type: ignore