from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from expert_poker_player.cards import Card
from expert_poker_player.uth.enums import (
    Action,
    GamePhase,
    RoundOutcome,
)
from expert_poker_player.uth.rules import legal_actions_for_phase
from expert_poker_player.uth.settlement import (
    VALID_PLAY_MULTIPLIERS,
)
from expert_poker_player.uth.wagers import Settlement


_EXPECTED_CARD_COUNTS: Final[
    Mapping[GamePhase, tuple[int, int]]
] = MappingProxyType(
    {
        # community cards, burned cards
        GamePhase.PREFLOP: (0, 0),
        GamePhase.FLOP: (3, 1),
        GamePhase.RIVER: (5, 2),
        GamePhase.TERMINAL: (5, 2),
    }
)


@dataclass(frozen=True, slots=True)
class RoundState:
    """
    Pełny wewnętrzny stan rozdania.

    Ten obiekt może zawierać informacje ukryte przed agentem,
    w szczególności karty krupiera i spalone karty.
    """

    phase: GamePhase

    player_cards: tuple[Card, Card]
    dealer_cards: tuple[Card, Card]

    community_cards: tuple[Card, ...]
    burned_cards: tuple[Card, ...]

    play_multiplier: int | None = None
    outcome: RoundOutcome | None = None
    settlement: Settlement | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, GamePhase): # type: ignore
            raise TypeError(
                "phase must be an instance of GamePhase"
            )

        _validate_card_tuple(
            self.player_cards,
            field_name="player_cards",
            expected_length=2,
        )
        _validate_card_tuple(
            self.dealer_cards,
            field_name="dealer_cards",
            expected_length=2,
        )
        _validate_card_tuple(
            self.community_cards,
            field_name="community_cards",
        )
        _validate_card_tuple(
            self.burned_cards,
            field_name="burned_cards",
        )

        expected_community_cards, expected_burned_cards = (
            _EXPECTED_CARD_COUNTS[self.phase]
        )

        if len(self.community_cards) != expected_community_cards:
            raise ValueError(
                f"{self.phase.name} requires exactly "
                f"{expected_community_cards} community cards, "
                f"but received {len(self.community_cards)}"
            )

        if len(self.burned_cards) != expected_burned_cards:
            raise ValueError(
                f"{self.phase.name} requires exactly "
                f"{expected_burned_cards} burned cards, "
                f"but received {len(self.burned_cards)}"
            )

        all_cards = (
            *self.player_cards,
            *self.dealer_cards,
            *self.community_cards,
            *self.burned_cards,
        )

        if len(set(all_cards)) != len(all_cards):
            raise ValueError(
                "round state cannot contain duplicate cards"
            )

        _validate_optional_play_multiplier(
            self.play_multiplier
        )
        _validate_optional_outcome(self.outcome)
        _validate_optional_settlement(self.settlement)

        self._validate_phase_invariants()

    def _validate_phase_invariants(self) -> None:
        if self.phase is GamePhase.TERMINAL:
            if self.outcome is None:
                raise ValueError(
                    "terminal state requires a round outcome"
                )

            if self.settlement is None:
                raise ValueError(
                    "terminal state requires a settlement"
                )

            if (
                self.outcome is RoundOutcome.PLAYER_FOLD
                and self.play_multiplier is not None
            ):
                raise ValueError(
                    "folded round cannot have a Play multiplier"
                )

            if (
                self.outcome is not RoundOutcome.PLAYER_FOLD
                and self.play_multiplier is None
            ):
                raise ValueError(
                    "showdown result requires a Play multiplier"
                )

            return

        if self.play_multiplier is not None:
            raise ValueError(
                "non-terminal state cannot have a Play multiplier"
            )

        if self.outcome is not None:
            raise ValueError(
                "non-terminal state cannot have a round outcome"
            )

        if self.settlement is not None:
            raise ValueError(
                "non-terminal state cannot have a settlement"
            )


@dataclass(frozen=True, slots=True)
class UTHObservation:
    """
    Informacje widoczne dla agenta.

    Celowo nie zawiera kart krupiera, spalonych kart
    ani kolejności pozostałej talii.
    """

    phase: GamePhase
    player_cards: tuple[Card, Card]
    community_cards: tuple[Card, ...]
    legal_actions: frozenset[Action]
    play_multiplier: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, GamePhase): # type: ignore
            raise TypeError(
                "phase must be an instance of GamePhase"
            )

        _validate_card_tuple(
            self.player_cards,
            field_name="player_cards",
            expected_length=2,
        )
        _validate_card_tuple(
            self.community_cards,
            field_name="community_cards",
        )

        expected_community_cards = _EXPECTED_CARD_COUNTS[
            self.phase
        ][0]

        if len(self.community_cards) != expected_community_cards:
            raise ValueError(
                f"{self.phase.name} observation requires exactly "
                f"{expected_community_cards} community cards, "
                f"but received {len(self.community_cards)}"
            )

        visible_cards = (
            *self.player_cards,
            *self.community_cards,
        )

        if len(set(visible_cards)) != len(visible_cards):
            raise ValueError(
                "observation cannot contain duplicate cards"
            )

        if not isinstance(self.legal_actions, frozenset): # type: ignore
            raise TypeError(
                "legal_actions must be a frozenset"
            )

        if not all(
            isinstance(action, Action) # type: ignore
            for action in self.legal_actions
        ):
            raise TypeError(
                "legal_actions must contain only Action values"
            )

        expected_actions = legal_actions_for_phase(self.phase)

        if self.legal_actions != expected_actions:
            raise ValueError(
                "legal_actions do not match the current phase"
            )

        _validate_optional_play_multiplier(
            self.play_multiplier
        )

        if (
            self.phase is not GamePhase.TERMINAL
            and self.play_multiplier is not None
        ):
            raise ValueError(
                "non-terminal observation cannot have "
                "a Play multiplier"
            )

    @property
    def terminated(self) -> bool:
        """Informuje, czy rozdanie zostało zakończone."""

        return self.phase is GamePhase.TERMINAL


@dataclass(frozen=True, slots=True)
class StepResult:
    """Wynik wykonania pojedynczej akcji w silniku UTH."""

    observation: UTHObservation
    outcome: RoundOutcome | None = None
    settlement: Settlement | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation, UTHObservation): # type: ignore
            raise TypeError(
                "observation must be an instance "
                "of UTHObservation"
            )

        _validate_optional_outcome(self.outcome)
        _validate_optional_settlement(self.settlement)

        if self.observation.terminated:
            if self.outcome is None:
                raise ValueError(
                    "terminal step result requires "
                    "a round outcome"
                )

            if self.settlement is None:
                raise ValueError(
                    "terminal step result requires "
                    "a settlement"
                )

            return

        if self.outcome is not None:
            raise ValueError(
                "non-terminal step result cannot have "
                "a round outcome"
            )

        if self.settlement is not None:
            raise ValueError(
                "non-terminal step result cannot have "
                "a settlement"
            )

    @property
    def terminated(self) -> bool:
        """Informuje, czy akcja zakończyła rozdanie."""

        return self.observation.terminated


def observation_from_state(
    state: RoundState,
) -> UTHObservation:
    """Buduje bezpieczną obserwację agenta z pełnego stanu."""

    if not isinstance(state, RoundState): # type: ignore
        raise TypeError(
            "state must be an instance of RoundState"
        )

    return UTHObservation(
        phase=state.phase,
        player_cards=state.player_cards,
        community_cards=state.community_cards,
        legal_actions=legal_actions_for_phase(state.phase),
        play_multiplier=state.play_multiplier,
    )


def step_result_from_state(
    state: RoundState,
) -> StepResult:
    """Buduje wynik kroku na podstawie pełnego stanu."""

    if not isinstance(state, RoundState): # type: ignore
        raise TypeError(
            "state must be an instance of RoundState"
        )

    return StepResult(
        observation=observation_from_state(state),
        outcome=state.outcome,
        settlement=state.settlement,
    )


def _validate_card_tuple(
    cards: object,
    *,
    field_name: str,
    expected_length: int | None = None,
) -> None:
    if not isinstance(cards, tuple):
        raise TypeError(
            f"{field_name} must be a tuple"
        )

    if (
        expected_length is not None
        and len(cards) != expected_length # type: ignore
    ):
        raise ValueError(
            f"{field_name} must contain exactly "
            f"{expected_length} cards"
        )

    if not all(isinstance(card, Card) for card in cards): # type: ignore
        raise TypeError(
            f"{field_name} must contain only Card values"
        )

    if len(set(cards)) != len(cards): # type: ignore
        raise ValueError(
            f"{field_name} cannot contain duplicate cards"
        )


def _validate_optional_play_multiplier(
    play_multiplier: object,
) -> None:
    if play_multiplier is None:
        return

    if type(play_multiplier) is not int:
        raise TypeError(
            "play_multiplier must be an integer or None"
        )

    if play_multiplier not in VALID_PLAY_MULTIPLIERS:
        raise ValueError(
            "play_multiplier must be one of: 1, 2, 3, 4"
        )


def _validate_optional_outcome(
    outcome: object,
) -> None:
    if outcome is not None and not isinstance(
        outcome,
        RoundOutcome,
    ):
        raise TypeError(
            "outcome must be an instance "
            "of RoundOutcome or None"
        )


def _validate_optional_settlement(
    settlement: object,
) -> None:
    if settlement is not None and not isinstance(
        settlement,
        Settlement,
    ):
        raise TypeError(
            "settlement must be an instance "
            "of Settlement or None"
        )