from collections import deque
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from expert_poker_player.cards import Card


@runtime_checkable
class CardSource(Protocol):
    """Źródło, z którego silnik może dobierać kolejne karty."""

    def draw(self) -> Card:
        """Dobiera jedną kartę."""
        ...


class FixedDeck:
    """
    Deterministyczne źródło kart używane w testach i replayach.

    Pierwszy element draw_order jest pierwszą dobieraną kartą.
    """

    def __init__(
        self,
        draw_order: Sequence[Card],
    ) -> None:
        if not isinstance(draw_order, Sequence): # type: ignore
            raise TypeError("draw_order must be a sequence")

        cards = tuple(draw_order)

        if not all(isinstance(card, Card) for card in cards): # type: ignore
            raise TypeError(
                "draw_order must contain only Card values"
            )

        if len(set(cards)) != len(cards):
            raise ValueError(
                "draw_order cannot contain duplicate cards"
            )

        self._cards = deque(cards)

    def __len__(self) -> int:
        """Zwraca liczbę kart, które pozostały do dobrania."""

        return len(self._cards)

    @property
    def cards(self) -> tuple[Card, ...]:
        """Zwraca pozostałe karty w kolejności dobierania."""

        return tuple(self._cards)

    def draw(self) -> Card:
        """Dobiera pierwszą kartę z ustalonej kolejności."""

        if not self._cards:
            raise IndexError(
                "cannot draw a card from an empty fixed deck"
            )

        return self._cards.popleft()