from typing import (
    Protocol,
    TypeAlias,
    runtime_checkable,
)

from expert_poker_player.uth import UTHObservation


StateVector: TypeAlias = tuple[float, ...]


@runtime_checkable
class StateEncoder(Protocol):
    """Kontrakt kodowania obserwacji UTH do wektora numerycznego."""

    @property
    def output_size(self) -> int:
        """Zwraca stały wymiar wektora generowanego przez encoder."""

        ...

    def encode(
        self,
        observation: UTHObservation,
    ) -> StateVector:
        """Koduje obserwację agenta do stałowymiarowego wektora."""

        ...