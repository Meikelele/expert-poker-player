from typing import Final

from expert_poker_player.state_representation.card_encoding import (
    CARD_COUNT,
    encode_card,
)
from expert_poker_player.state_representation.protocol import (
    StateVector,
)
from expert_poker_player.uth import (
    Action,
    GamePhase,
    UTHObservation,
)


PLAYER_CARD_SLOTS: Final = 2
COMMUNITY_CARD_SLOTS: Final = 5

_DECISION_PHASES: Final[tuple[GamePhase, ...]] = (
    GamePhase.PREFLOP,
    GamePhase.FLOP,
    GamePhase.RIVER,
)

_ACTION_ORDER: Final[tuple[Action, ...]] = tuple(Action)

PHASE_FEATURE_COUNT: Final = len(_DECISION_PHASES)
ACTION_FEATURE_COUNT: Final = len(_ACTION_ORDER)

RAW_STATE_SIZE: Final = (
    (
        PLAYER_CARD_SLOTS
        + COMMUNITY_CARD_SLOTS
    )
    * CARD_COUNT
    + PHASE_FEATURE_COUNT
    + ACTION_FEATURE_COUNT
)


class RawStateEncoder:
    """Koduje obserwację UTH bez cech domenowych."""

    @property
    def output_size(self) -> int:
        return RAW_STATE_SIZE

    def encode(
        self,
        observation: UTHObservation,
    ) -> StateVector:
        if not isinstance(
            observation,
            UTHObservation,
        ): # type: ignore
            raise TypeError(
                "observation must be an instance "
                "of UTHObservation"
            )

        if observation.terminated:
            raise ValueError(
                "cannot encode a terminal observation"
            )

        community_cards = (
            *observation.community_cards,
            *(
                None
                for _ in range(
                    COMMUNITY_CARD_SLOTS
                    - len(
                        observation.community_cards
                    )
                )
            ),
        )

        card_features = tuple(
            value
            for card in (
                *observation.player_cards,
                *community_cards,
            )
            for value in encode_card(card)
        )

        phase_features = tuple(
            1.0
            if observation.phase is phase
            else 0.0
            for phase in _DECISION_PHASES
        )

        action_features = tuple(
            1.0
            if action in observation.legal_actions
            else 0.0
            for action in _ACTION_ORDER
        )

        encoded = (
            *card_features,
            *phase_features,
            *action_features,
        )

        if len(encoded) != RAW_STATE_SIZE:
            raise RuntimeError(
                "raw state encoder produced "
                "an invalid output size"
            )

        return encoded