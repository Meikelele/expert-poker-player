# przechowuje wynik oceny ręki i helpery do porownania rak pomiedzy soba
from dataclasses import dataclass
from functools import total_ordering
from typing import ClassVar

from expert_poker_player.cards import Rank
from expert_poker_player.hands.hand_rank import HandRank


@total_ordering
@dataclass(frozen=True, slots=True, eq=False)
class HandValue:
    """Pełna wartość układu pokerowego używana do porównywania rąk."""

    rank: HandRank
    tiebreak: tuple[int, ...]

    _EXPECTED_TIEBREAK_LENGTHS: ClassVar[dict[HandRank, int]] = {
        HandRank.HIGH_CARD: 5,
        HandRank.ONE_PAIR: 4,
        HandRank.TWO_PAIR: 3,
        HandRank.THREE_OF_A_KIND: 3,
        HandRank.STRAIGHT: 1,
        HandRank.FLUSH: 5,
        HandRank.FULL_HOUSE: 2,
        HandRank.FOUR_OF_A_KIND: 2,
        HandRank.STRAIGHT_FLUSH: 1,
    }

    def __post_init__(self) -> None:
        """Sprawdza poprawność utworzonej wartości układu."""

        if not isinstance(self.rank, HandRank): # type: ignore
            raise TypeError("[HandValue]: rank must be an instance of HandRank")

        if not isinstance(self.tiebreak, tuple): # type: ignore
            raise TypeError("[HandValue]: tiebreak must be a tuple")

        if not all(type(value) is int for value in self.tiebreak):
            raise TypeError("[HandValue]:all tiebreak values must be integers")

        expected_length = self._EXPECTED_TIEBREAK_LENGTHS[self.rank]

        if len(self.tiebreak) != expected_length:
            raise ValueError(
                f"{self.rank.name} requires exactly "
                f"{expected_length} tiebreak values, "
                f"but received {len(self.tiebreak)}"
            )

        if any(value < Rank.TWO.value or value > Rank.ACE.value
               for value in self.tiebreak):
            raise ValueError("[HandValue]: tiebreak values must be between 2 and 14")

        if self.rank in {
            HandRank.STRAIGHT,
            HandRank.STRAIGHT_FLUSH,
        } and self.tiebreak[0] < Rank.FIVE.value:
            raise ValueError(
                "the highest card of a straight must be between 5 and 14"
            )

    @property
    def comparison_key(self) -> tuple[int, ...]:
        """Zwraca krotkę używaną do porównywania układów."""

        return (self.rank.value, *self.tiebreak)

    @property
    def is_royal_flush(self) -> bool:
        """Informuje, czy układ jest pokerem królewskim."""

        return (
            self.rank is HandRank.STRAIGHT_FLUSH
            and self.tiebreak == (Rank.ACE.value,)
        )

    def __eq__(self, other: object) -> bool:
        """Sprawdza, czy dwa układy mają identyczną wartość."""

        if not isinstance(other, HandValue):
            return False

        return self.comparison_key == other.comparison_key

    def __lt__(self, other: object) -> bool:
        """Sprawdza, czy bieżący układ jest słabszy od drugiego."""

        if not isinstance(other, HandValue):
            return NotImplemented

        return self.comparison_key < other.comparison_key

    def __hash__(self) -> int:
        """Pozwala przechowywać wartości układów w zbiorach i słownikach."""

        return hash(self.comparison_key)