from dataclasses import dataclass

from expert_poker_player.cards import Card
from expert_poker_player.hands.hand_value import HandValue


@dataclass(frozen=True, slots=True)
class EvaluatedHand:
    """Wynik oceny układu wraz z pięcioma kartami, które go tworzą."""

    value: HandValue
    cards: tuple[Card, Card, Card, Card, Card]

    def __post_init__(self) -> None:
        """Sprawdza poprawność wyniku evaluacji."""

        if not isinstance(self.value, HandValue): # type: ignore
            raise TypeError("value must be an instance of HandValue")

        if not isinstance(self.cards, tuple): # type: ignore
            raise TypeError("cards must be a tuple")

        if len(self.cards) != 5:
            raise ValueError(
                f"an evaluated hand must contain exactly 5 cards, "
                f"but received {len(self.cards)}"
            )

        if not all(isinstance(card, Card) for card in self.cards): # type: ignore
            raise TypeError("all elements of cards must be instances of Card")

        if len(set(self.cards)) != 5:
            raise ValueError("an evaluated hand cannot contain duplicate cards")