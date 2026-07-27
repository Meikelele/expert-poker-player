from types import MappingProxyType
from typing import Final, Mapping

from expert_poker_player.uth.enums import Action, GamePhase


LEGAL_ACTIONS_BY_PHASE: Final[
    Mapping[GamePhase, frozenset[Action]]
] = MappingProxyType(
    {
        GamePhase.PREFLOP: frozenset(
            {
                Action.CHECK,
                Action.BET_3X,
                Action.BET_4X,
            }
        ),
        GamePhase.FLOP: frozenset(
            {
                Action.CHECK,
                Action.BET_2X,
            }
        ),
        GamePhase.RIVER: frozenset(
            {
                Action.BET_1X,
                Action.FOLD,
            }
        ),
        GamePhase.TERMINAL: frozenset(),
    }
)


def legal_actions_for_phase(
    phase: GamePhase,
) -> frozenset[Action]:
    """Zwraca akcje legalne w podanej fazie rozdania."""

    if not isinstance(phase, GamePhase): # type: ignore
        raise TypeError("phase must be an instance of GamePhase")

    return LEGAL_ACTIONS_BY_PHASE[phase]