from random import Random

from expert_poker_player.cards.card import Card, Rank, Suit


class Deck:
    """Standardowa talia składająca się z 52 unikalnych kart."""

    def __init__(
        self,
        *,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self._random = Random(seed)

        self._cards = [
            Card(rank=rank, suit=suit)
            for suit in Suit
            for rank in Rank
        ]

        if shuffle:
            self.shuffle()

    def __len__(self) -> int:
        """Zwraca aktualną liczbę kart w talii."""
        return len(self._cards)

    @property
    def cards(self) -> tuple[Card, ...]:
        """Zwraca niemodyfikowalny podgląd kart znajdujących się w talii."""
        return tuple(self._cards)

    def shuffle(self) -> None:
        """Tasuje karty znajdujące się w talii."""
        self._random.shuffle(self._cards)

    def draw(self) -> Card:
        """Dobiera i usuwa jedną kartę z wierzchu talii."""
        if not self._cards:
            raise IndexError("Cannot draw a card from an empty deck")

        return self._cards.pop()

    def draw_many(self, count: int) -> list[Card]:
        """Dobiera określoną liczbę kart z talii."""
        if count < 0:
            raise ValueError("[Deck > draw_many]: count cannot be negative")

        if count > len(self._cards):
            raise ValueError(
                f"[Deck > draw_many]:Cannot draw {count} cards: only {len(self._cards)} cards remain"
            )

        return [self.draw() for _ in range(count)]
