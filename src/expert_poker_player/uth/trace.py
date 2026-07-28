from dataclasses import dataclass

from expert_poker_player.uth.enums import (
    Action,
    GamePhase,
    RoundOutcome,
)
from expert_poker_player.uth.models import (
    RoundState,
    UTHObservation,
)
from expert_poker_player.uth.showdown import ShowdownResult
from expert_poker_player.uth.wagers import Settlement


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """
    Pojedyncza decyzja agenta.

    Observation przedstawia dokładnie informacje dostępne
    agentowi przed wybraniem akcji.
    """

    observation: UTHObservation
    action: Action

    def __post_init__(self) -> None:
        if not isinstance(self.observation, UTHObservation): # type: ignore
            raise TypeError(
                "observation must be an instance of UTHObservation"
            )

        if not isinstance(self.action, Action): # type: ignore
            raise TypeError(
                "action must be an instance of Action"
            )

        if self.observation.terminated:
            raise ValueError(
                "cannot record a decision for a terminal observation"
            )

        if self.action not in self.observation.legal_actions:
            raise ValueError(
                "recorded action must be legal for the observation"
            )


@dataclass(frozen=True, slots=True)
class RoundTrace:
    """
    Niezmienny zapis dotychczasowego przebiegu rozdania.

    Decisions zawiera obserwacje widziane przez agenta.
    State jest pełnym wewnętrznym stanem silnika.
    """

    round_id: int
    decisions: tuple[DecisionRecord, ...]
    state: RoundState

    def __post_init__(self) -> None:
        if type(self.round_id) is not int:
            raise TypeError("round_id must be an integer")

        if self.round_id < 1:
            raise ValueError("round_id must be positive")

        if not isinstance(self.decisions, tuple): # type: ignore
            raise TypeError("decisions must be a tuple")

        if not all(
            isinstance(decision, DecisionRecord) # type: ignore
            for decision in self.decisions
        ):
            raise TypeError(
                "decisions must contain only DecisionRecord values"
            )

        if not isinstance(self.state, RoundState): # type: ignore
            raise TypeError(
                "state must be an instance of RoundState"
            )

    @property
    def completed(self) -> bool:
        """Informuje, czy zapis dotyczy zakończonego rozdania."""

        return self.state.phase is GamePhase.TERMINAL

    @property
    def outcome(self) -> RoundOutcome | None:
        """Zwraca końcowy wynik, jeśli rozdanie się zakończyło."""

        return self.state.outcome

    @property
    def settlement(self) -> Settlement | None:
        """Zwraca rozliczenie zakończonego rozdania."""

        return self.state.settlement

    @property
    def showdown(self) -> ShowdownResult | None:
        """Zwraca szczegóły showdownu, jeśli do niego doszło."""

        return self.state.showdown