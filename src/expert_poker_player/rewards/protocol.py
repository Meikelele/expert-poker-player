from typing import (
    Protocol,
    TypeAlias,
    runtime_checkable,
)

from expert_poker_player.uth import StepResult


RewardValue: TypeAlias = float


@runtime_checkable
class RewardFunction(Protocol):
    """Wspólny kontrakt funkcji nagrody dla środowiska UTH."""

    def calculate_reward(
        self,
        step_result: StepResult,
    ) -> RewardValue:
        """Oblicza nagrodę dla wyniku pojedynczego kroku."""

        ...  # pragma: no cover