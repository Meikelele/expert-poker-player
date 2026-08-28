from dataclasses import dataclass
import math

from expert_poker_player.rl.actions import (
    ACTION_COUNT,
)
from expert_poker_player.state_representation import (
    StateVector,
)


@dataclass(
    frozen=True,
    slots=True,
)
class PolicyStep:
    """Pojedyncza decyzja zapisana podczas epizodu."""

    state: StateVector
    action_index: int
    action_mask: tuple[bool, ...]
    reward: float

    def __post_init__(self) -> None:
        if not isinstance(
            self.state,
            tuple,
        ):  # type: ignore
            raise TypeError(
                "state must be a tuple"
            )

        if not self.state:
            raise ValueError(
                "state cannot be empty"
            )

        if type(self.action_index) is not int:
            raise TypeError(
                "action_index must be an integer"
            )

        if not 0 <= self.action_index < ACTION_COUNT:
            raise ValueError(
                "action_index must reference "
                "an available action"
            )

        if not isinstance(
            self.action_mask,
            tuple,
        ):  # type: ignore
            raise TypeError(
                "action_mask must be a tuple"
            )

        if len(
            self.action_mask
        ) != ACTION_COUNT:
            raise ValueError(
                "action_mask must match "
                "the action count"
            )

        if not all(
            type(value) is bool
            for value in self.action_mask
        ):
            raise TypeError(
                "action_mask must contain "
                "boolean values"
            )

        if not any(
            self.action_mask
        ):
            raise ValueError(
                "action_mask must contain "
                "at least one legal action"
            )

        if not self.action_mask[
            self.action_index
        ]:
            raise ValueError(
                "selected action must be legal"
            )

        if not isinstance(
            self.reward,
            (int, float),
        ):  # type: ignore
            raise TypeError(
                "reward must be a number"
            )

        reward = float(
            self.reward
        )

        if not math.isfinite(
            reward
        ):
            raise ValueError(
                "reward must be finite"
            )

        object.__setattr__(
            self,
            "reward",
            reward,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class Trajectory:
    """Kompletny epizod wykorzystywany przez REINFORCE."""

    steps: tuple[PolicyStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.steps,
            tuple,
        ):  # type: ignore
            raise TypeError(
                "steps must be a tuple"
            )

        if not self.steps:
            raise ValueError(
                "trajectory cannot be empty"
            )

        if not all(
            isinstance(
                step,
                PolicyStep,
            ) # type: ignore
            for step in self.steps
        ):
            raise TypeError(
                "steps must contain "
                "only PolicyStep values"
            )

    def __len__(
        self,
    ) -> int:
        return len(
            self.steps
        )

    @property
    def total_reward(
        self,
    ) -> float:
        return sum(
            step.reward
            for step in self.steps
        )