from dataclasses import dataclass

from expert_poker_player.hands import EvaluatedHand
from expert_poker_player.uth.enums import RoundOutcome
from expert_poker_player.uth.settlement import (
    dealer_qualifies,
    determine_round_outcome,
)


@dataclass(frozen=True, slots=True)
class ShowdownResult:
    """Szczegółowy wynik porównania rąk gracza i krupiera."""

    player_hand: EvaluatedHand
    dealer_hand: EvaluatedHand

    def __post_init__(self) -> None:
        if not isinstance(self.player_hand, EvaluatedHand): # type: ignore
            raise TypeError(
                "player_hand must be an instance of EvaluatedHand"
            )

        if not isinstance(self.dealer_hand, EvaluatedHand): # type: ignore
            raise TypeError(
                "dealer_hand must be an instance of EvaluatedHand"
            )

    @property
    def outcome(self) -> RoundOutcome:
        """Zwraca wynik showdownu z perspektywy gracza."""

        return determine_round_outcome(
            player_hand=self.player_hand.value,
            dealer_hand=self.dealer_hand.value,
        )

    @property
    def dealer_qualified(self) -> bool:
        """Informuje, czy krupier ma parę lub lepszy układ."""

        return dealer_qualifies(self.dealer_hand.value)