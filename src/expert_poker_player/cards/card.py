from dataclasses import dataclass
from enum import Enum, IntEnum


class Suit(str, Enum):
    """Definiuje kolor karty: trefl, karo, kier, pik."""

    CLUBS = "clubs"
    DIAMONDS = "diamonds"
    HEARTS = "hearts"
    SPADES = "spades"

    @property
    def symbol(self) -> str:
        """Zwraca symbol koloru używany przy wyświetlaniu karty."""
        symbols = {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥",
            Suit.SPADES: "♠",
        }
        return symbols[self]


class Rank(IntEnum):
    """Wartość karty od dwójki do asa."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def symbol(self) -> str:
        """Zwraca skróconą nazwę wartości karty."""
        face_symbols = {
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }
        return face_symbols.get(self, str(self.value))


@dataclass(frozen=True, slots=True)
class Card:
    """Pojedyncza karta w standardowej talii."""

    rank: Rank
    suit: Suit

    def __post_init__(self) -> None:
        """Sprawdza poprawność danych podczas tworzenia karty."""
        if not isinstance(self.rank, Rank): # type: ignore
            raise TypeError("[Card]: 'rank' must be an instance of Rank")

        if not isinstance(self.suit, Suit): # type: ignore
            raise TypeError("[Card]: 'suit' must be an instance of Suit")

    def __str__(self) -> str:
        """Zwraca czytelny zapis karty, np. A♠."""
        return f"{self.rank.symbol}{self.suit.symbol}"
