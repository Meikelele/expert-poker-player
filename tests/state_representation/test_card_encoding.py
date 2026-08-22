import pytest

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.state_representation import (
    CARD_COUNT,
    card_to_index,
    encode_card,
)


def test_card_count_matches_standard_deck() -> None:
    assert CARD_COUNT == 52


@pytest.mark.parametrize(
    ("card", "expected_index"),
    [
        (
            Card(
                rank=Rank.TWO,
                suit=Suit.CLUBS,
            ),
            0,
        ),
        (
            Card(
                rank=Rank.ACE,
                suit=Suit.CLUBS,
            ),
            12,
        ),
        (
            Card(
                rank=Rank.TWO,
                suit=Suit.DIAMONDS,
            ),
            13,
        ),
        (
            Card(
                rank=Rank.ACE,
                suit=Suit.SPADES,
            ),
            51,
        ),
    ],
)
def test_card_has_fixed_index(
    card: Card,
    expected_index: int,
) -> None:
    assert card_to_index(card) == expected_index


def test_all_cards_have_unique_indices() -> None:
    indices = {
        card_to_index(
            Card(
                rank=rank,
                suit=suit,
            )
        )
        for suit in Suit
        for rank in Rank
    }

    assert indices == set(range(CARD_COUNT))


def test_encode_card_returns_one_hot_vector() -> None:
    card = Card(
        rank=Rank.ACE,
        suit=Suit.SPADES,
    )

    encoded = encode_card(card)

    assert len(encoded) == CARD_COUNT
    assert sum(encoded) == 1.0
    assert encoded[51] == 1.0


def test_empty_card_slot_is_zero_vector() -> None:
    encoded = encode_card(None)

    assert len(encoded) == CARD_COUNT
    assert encoded == (0.0,) * CARD_COUNT


def test_card_to_index_rejects_invalid_value() -> None:
    with pytest.raises(
        TypeError,
        match="card must be an instance of Card",
    ):
        card_to_index("A♠")  # type: ignore[arg-type]