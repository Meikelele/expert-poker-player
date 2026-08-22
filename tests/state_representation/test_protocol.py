from expert_poker_player.state_representation import (
    StateEncoder,
    StateVector,
)
from expert_poker_player.uth import UTHObservation


class StubStateEncoder:
    @property
    def output_size(self) -> int:
        return 1

    def encode(
        self,
        observation: UTHObservation,
    ) -> StateVector:
        return (1.0,)


class MissingEncode:
    @property
    def output_size(self) -> int:
        return 1


def test_compatible_encoder_satisfies_protocol() -> None:
    encoder = StubStateEncoder()

    assert isinstance(
        encoder,
        StateEncoder,
    )


def test_incomplete_encoder_does_not_satisfy_protocol() -> None:
    encoder = MissingEncode()

    assert not isinstance(
        encoder,
        StateEncoder,
    )