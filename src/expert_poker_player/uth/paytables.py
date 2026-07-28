from fractions import Fraction
from types import MappingProxyType
from typing import Final, Mapping

from expert_poker_player.hands import HandRank, HandValue


BLIND_PAYOUTS: Final[Mapping[HandRank, Fraction]] = MappingProxyType(
    {
        HandRank.STRAIGHT: Fraction(1),
        HandRank.FLUSH: Fraction(3, 2),
        HandRank.FULL_HOUSE: Fraction(3),
        HandRank.FOUR_OF_A_KIND: Fraction(10),
        HandRank.STRAIGHT_FLUSH: Fraction(50),
    }
)

ROYAL_FLUSH_BLIND_PAYOUT: Final[Fraction] = Fraction(500)


def blind_profit(player_hand: HandValue) -> Fraction:
    """
    Zwraca zysk netto zakładu Blind dla wygrywającej ręki gracza.

    Funkcja zakłada, że gracz pokonał krupiera. Dla układów poniżej
    strita Blind pushuje, dlatego zwracany jest zysk równy zero.
    """

    if not isinstance(player_hand, HandValue): # type: ignore
        raise TypeError("player_hand must be an instance of HandValue")

    if player_hand.is_royal_flush:
        return ROYAL_FLUSH_BLIND_PAYOUT

    return BLIND_PAYOUTS.get(
        player_hand.rank,
        Fraction(0),
    )