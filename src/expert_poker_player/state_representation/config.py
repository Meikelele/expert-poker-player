from enum import Enum

from expert_poker_player.state_representation.feature_encoder import (
    FeatureStateEncoder,
)
from expert_poker_player.state_representation.protocol import (
    StateEncoder,
)
from expert_poker_player.state_representation.raw_encoder import (
    RawStateEncoder,
)


class StateRepresentation(str, Enum):
    """Dostępne warianty reprezentacji stanu."""

    RAW = "raw"
    FEATURES = "features"


def build_state_encoder(
    representation: StateRepresentation,
) -> StateEncoder:
    """Tworzy encoder odpowiadający wybranej reprezentacji."""

    if not isinstance(
        representation,
        StateRepresentation,
    ): # type: ignore
        raise TypeError(
            "representation must be an instance "
            "of StateRepresentation"
        )

    if representation is StateRepresentation.RAW:
        return RawStateEncoder()

    return FeatureStateEncoder()