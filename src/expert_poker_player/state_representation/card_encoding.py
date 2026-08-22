from types import MappingProxyType
from typing import Final, Mapping

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.state_representation.protocol import (
    StateVector,
)


CARD_COUNT: Final = len(Rank) * len(Suit)

_CARD_ORDER: Final[tuple[Card, ...]] = tuple(
    Card(
        rank=rank,
        suit=suit,
    )
    for suit in Suit
    for rank in Rank
)

_CARD_INDEX: Final[Mapping[Card, int]] = MappingProxyType(
    {
        card: index
        for index, card in enumerate(_CARD_ORDER)
    }
)

_EMPTY_CARD_VECTOR: Final[StateVector] = (
    0.0,
) * CARD_COUNT


def card_to_index(
    card: Card,
) -> int:
    """Zwraca stały indeks karty w zakresie 0-51."""

    if not isinstance(card, Card): # type: ignore
        raise TypeError(
            "card must be an instance of Card"
        )

    return _CARD_INDEX[card]


def encode_card(
    card: Card | None,
) -> StateVector:
    """Koduje kartę jako wektor one-hot lub pusty slot."""

    if card is None:
        return _EMPTY_CARD_VECTOR

    index = card_to_index(card)

    encoded = [0.0] * CARD_COUNT
    encoded[index] = 1.0

    return tuple(encoded)