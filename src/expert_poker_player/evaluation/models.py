from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Final, Mapping

from expert_poker_player.uth import (
    Action,
    RoundOutcome,
    Settlement,
)


_PLAY_MULTIPLIER_BY_ACTION_SEQUENCE: Final[
    Mapping[tuple[Action, ...], int | None]
] = MappingProxyType(
    {
        (Action.BET_4X,): 4,
        (Action.BET_3X,): 3,
        (Action.CHECK, Action.BET_2X): 2,
        (
            Action.CHECK,
            Action.CHECK,
            Action.BET_1X,
        ): 1,
        (
            Action.CHECK,
            Action.CHECK,
            Action.FOLD,
        ): None,
    }
)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Zwięzły wynik jednego zakończonego rozdania UTH."""

    actions: tuple[Action, ...]
    outcome: RoundOutcome
    settlement: Settlement

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple): # type: ignore
            raise TypeError("actions must be a tuple")

        if not all(
            isinstance(action, Action) # type: ignore
            for action in self.actions
        ):
            raise TypeError(
                "actions must contain only Action values"
            )

        if (
            self.actions
            not in _PLAY_MULTIPLIER_BY_ACTION_SEQUENCE
        ):
            raise ValueError(
                "actions must represent a completed UTH round"
            )

        if not isinstance(self.outcome, RoundOutcome): # type: ignore
            raise TypeError(
                "outcome must be an instance of RoundOutcome"
            )

        if not isinstance(self.settlement, Settlement): # type: ignore
            raise TypeError(
                "settlement must be an instance of Settlement"
            )

        expected_multiplier = self.play_multiplier

        if expected_multiplier is None:
            if self.outcome is not RoundOutcome.PLAYER_FOLD:
                raise ValueError(
                    "fold action sequence requires PLAYER_FOLD outcome"
                )

            if self.settlement.play.stake != 0:
                raise ValueError(
                    "folded episode cannot contain a Play stake"
                )

            return

        if self.outcome is RoundOutcome.PLAYER_FOLD:
            raise ValueError(
                "bet action sequence cannot have PLAYER_FOLD outcome"
            )

        if self.settlement.play.stake != Fraction(
            expected_multiplier
        ):
            raise ValueError(
                "Play stake must match the final bet action"
            )

    @property
    def decision_count(self) -> int:
        """Zwraca liczbę decyzji wykonanych w rozdaniu."""

        return len(self.actions)

    @property
    def play_multiplier(self) -> int | None:
        """Zwraca mnożnik zakładu Play albo None po foldzie."""

        return _PLAY_MULTIPLIER_BY_ACTION_SEQUENCE[
            self.actions
        ]

    @property
    def folded(self) -> bool:
        """Informuje, czy gracz zakończył rozdanie foldem."""

        return self.outcome is RoundOutcome.PLAYER_FOLD

    @property
    def net_profit(self) -> Fraction:
        """Zwraca końcowy wynik netto w jednostkach Ante."""

        return self.settlement.total_net_profit

    @property
    def total_staked(self) -> Fraction:
        """Zwraca całkowitą stawkę w jednostkach Ante."""

        return self.settlement.total_staked