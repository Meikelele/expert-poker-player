from fractions import Fraction
from typing import Final

from expert_poker_player.hands import HandRank, HandValue
from expert_poker_player.uth.enums import RoundOutcome
from expert_poker_player.uth.paytables import blind_profit
from expert_poker_player.uth.wagers import (
    Settlement,
    WagerSettlement,
)


ANTE_STAKE: Final[Fraction] = Fraction(1)
BLIND_STAKE: Final[Fraction] = Fraction(1)

VALID_PLAY_MULTIPLIERS: Final[frozenset[int]] = frozenset(
    {
        1,
        2,
        3,
        4,
    }
)


def dealer_qualifies(dealer_hand: HandValue) -> bool:
    """Sprawdza, czy krupier ma parę lub silniejszy układ."""

    _require_hand_value(dealer_hand, "dealer_hand")

    return dealer_hand.rank >= HandRank.ONE_PAIR


def determine_round_outcome(
    player_hand: HandValue,
    dealer_hand: HandValue,
) -> RoundOutcome:
    """Określa wynik showdownu z perspektywy gracza."""

    _require_hand_value(player_hand, "player_hand")
    _require_hand_value(dealer_hand, "dealer_hand")

    return _compare_hands(
        player_hand=player_hand,
        dealer_hand=dealer_hand,
    )


def settle_fold() -> Settlement:
    """
    Rozlicza spasowanie gracza.

    Gracz traci Ante i Blind. Zakład Play nie został postawiony.
    """

    return Settlement(
        ante=WagerSettlement(
            stake=ANTE_STAKE,
            net_profit=-ANTE_STAKE,
        ),
        blind=WagerSettlement(
            stake=BLIND_STAKE,
            net_profit=-BLIND_STAKE,
        ),
        play=WagerSettlement(
            stake=Fraction(0),
            net_profit=Fraction(0),
        ),
    )


def settle_showdown(
    player_hand: HandValue,
    dealer_hand: HandValue,
    play_multiplier: int,
) -> Settlement:
    """
    Rozlicza Ante, Blind i Play po showdownie.

    Wyniki są reprezentowane jako zysk lub strata netto
    wyrażone w jednostkach Ante.
    """

    _require_hand_value(player_hand, "player_hand")
    _require_hand_value(dealer_hand, "dealer_hand")
    _validate_play_multiplier(play_multiplier)

    play_stake = Fraction(play_multiplier)

    outcome = _compare_hands(
        player_hand=player_hand,
        dealer_hand=dealer_hand,
    )

    if outcome is RoundOutcome.PUSH:
        return Settlement(
            ante=_push_wager(ANTE_STAKE),
            blind=_push_wager(BLIND_STAKE),
            play=_push_wager(play_stake),
        )

    qualified = dealer_qualifies(dealer_hand)

    if outcome is RoundOutcome.PLAYER_WIN:
        ante_profit = (
            ANTE_STAKE
            if qualified
            else Fraction(0)
        )

        return Settlement(
            ante=WagerSettlement(
                stake=ANTE_STAKE,
                net_profit=ante_profit,
            ),
            blind=WagerSettlement(
                stake=BLIND_STAKE,
                net_profit=blind_profit(player_hand),
            ),
            play=WagerSettlement(
                stake=play_stake,
                net_profit=play_stake,
            ),
        )

    ante_profit = (
        -ANTE_STAKE
        if qualified
        else Fraction(0)
    )

    return Settlement(
        ante=WagerSettlement(
            stake=ANTE_STAKE,
            net_profit=ante_profit,
        ),
        blind=WagerSettlement(
            stake=BLIND_STAKE,
            net_profit=-BLIND_STAKE,
        ),
        play=WagerSettlement(
            stake=play_stake,
            net_profit=-play_stake,
        ),
    )


def _compare_hands(
    player_hand: HandValue,
    dealer_hand: HandValue,
) -> RoundOutcome:
    """Porównuje dwie wcześniej zwalidowane wartości układów."""

    if player_hand > dealer_hand:
        return RoundOutcome.PLAYER_WIN

    if player_hand < dealer_hand:
        return RoundOutcome.DEALER_WIN

    return RoundOutcome.PUSH


def _push_wager(stake: Fraction) -> WagerSettlement:
    """Tworzy rozliczenie zwróconego zakładu."""

    return WagerSettlement(
        stake=stake,
        net_profit=Fraction(0),
    )


def _require_hand_value(
    value: object,
    field_name: str,
) -> HandValue:
    """Sprawdza typ wartości układu pokerowego."""

    if not isinstance(value, HandValue):
        raise TypeError(
            f"{field_name} must be an instance of HandValue"
        )

    return value


def _validate_play_multiplier(
    play_multiplier: object,
) -> int:
    """Sprawdza mnożnik zakładu Play."""

    if type(play_multiplier) is not int:
        raise TypeError("play_multiplier must be an integer")

    if play_multiplier not in VALID_PLAY_MULTIPLIERS:
        raise ValueError(
            "play_multiplier must be one of: 1, 2, 3, 4"
        )

    return play_multiplier