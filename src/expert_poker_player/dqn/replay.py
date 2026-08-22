from collections import deque
from dataclasses import dataclass
import math
import random
from typing import TypeAlias

from expert_poker_player.dqn.actions import (
    ACTION_COUNT,
)
from expert_poker_player.state_representation import (
    StateVector,
)


ActionMask: TypeAlias = tuple[bool, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class Transition:
    """Pojedyncze przejście wykorzystywane przez DQN."""

    state: StateVector
    action_index: int
    reward: float
    next_state: StateVector | None
    terminated: bool
    next_action_mask: ActionMask | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.state,
            tuple,
        ): # type: ignore
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
            self.reward,
            (int, float),
        ): # type: ignore
            raise TypeError(
                "reward must be a number"
            )

        reward = float(
            self.reward
        )

        if not math.isfinite(reward):
            raise ValueError(
                "reward must be finite"
            )

        object.__setattr__(
            self,
            "reward",
            reward,
        )

        if type(self.terminated) is not bool:
            raise TypeError(
                "terminated must be a boolean"
            )

        if self.terminated:
            if self.next_state is not None:
                raise ValueError(
                    "terminal transition cannot "
                    "have next_state"
                )

            if self.next_action_mask is not None:
                raise ValueError(
                    "terminal transition cannot "
                    "have next_action_mask"
                )

            return

        if self.next_state is None:
            raise ValueError(
                "non-terminal transition requires "
                "next_state"
            )

        if not isinstance(
            self.next_state,
            tuple,
        ): # type: ignore
            raise TypeError(
                "next_state must be a tuple"
            )

        if self.next_action_mask is None:
            raise ValueError(
                "non-terminal transition requires "
                "next_action_mask"
            )

        if not isinstance(
            self.next_action_mask,
            tuple,
        ): # type: ignore
            raise TypeError(
                "next_action_mask must be a tuple"
            )

        if len(
            self.next_action_mask
        ) != ACTION_COUNT:
            raise ValueError(
                "next_action_mask must match "
                "the action count"
            )

        if not all(
            type(value) is bool
            for value in self.next_action_mask
        ):
            raise TypeError(
                "next_action_mask must contain "
                "boolean values"
            )

        if not any(
            self.next_action_mask
        ):
            raise ValueError(
                "non-terminal transition must "
                "have a legal next action"
            )


class ReplayBuffer:
    """Ograniczony bufor doświadczeń DQN."""

    def __init__(
        self,
        *,
        capacity: int,
        seed: int | None = None,
    ) -> None:
        if type(capacity) is not int:
            raise TypeError(
                "capacity must be an integer"
            )

        if capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        self._capacity = capacity

        self._buffer: deque[
            Transition
        ] = deque(
            maxlen=capacity
        )

        self._rng = random.Random(
            seed
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(
            self._buffer
        )

    def add(
        self,
        transition: Transition,
    ) -> None:
        if not isinstance(
            transition,
            Transition,
        ): # type: ignore
            raise TypeError(
                "transition must be an instance "
                "of Transition"
            )

        self._buffer.append(
            transition
        )

    def sample(
        self,
        batch_size: int,
    ) -> tuple[Transition, ...]:
        if type(batch_size) is not int:
            raise TypeError(
                "batch_size must be an integer"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if batch_size > len(
            self._buffer
        ):
            raise ValueError(
                "batch_size cannot exceed "
                "the number of stored transitions"
            )

        return tuple(
            self._rng.sample(
                tuple(
                    self._buffer
                ),
                batch_size,
            )
        )