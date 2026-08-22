from typing import Final

from expert_poker_player.state_representation.poker_features import (
    POKER_FEATURE_COUNT,
    extract_poker_features,
)
from expert_poker_player.state_representation.protocol import (
    StateVector,
)
from expert_poker_player.state_representation.raw_encoder import (
    RAW_STATE_SIZE,
    RawStateEncoder,
)
from expert_poker_player.uth import UTHObservation


FEATURE_STATE_SIZE: Final = (
    RAW_STATE_SIZE
    + POKER_FEATURE_COUNT
)


class FeatureStateEncoder:
    """Rozszerza surową reprezentację o cechy domenowe."""

    def __init__(self) -> None:
        self._raw_encoder = RawStateEncoder()

    @property
    def output_size(self) -> int:
        return FEATURE_STATE_SIZE

    def encode(
        self,
        observation: UTHObservation,
    ) -> StateVector:
        raw_features = self._raw_encoder.encode(
            observation
        )

        poker_features = extract_poker_features(
            observation
        )

        encoded = (
            *raw_features,
            *poker_features,
        )

        if len(encoded) != FEATURE_STATE_SIZE:
            raise RuntimeError(
                "feature state encoder produced "
                "an invalid output size"
            )

        return encoded