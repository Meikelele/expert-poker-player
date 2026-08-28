from collections.abc import Sequence

import torch
from torch import nn

from expert_poker_player.rl.actions import (
    ACTION_COUNT,
)


class PolicyNetwork(nn.Module):
    """Sieć generująca logity polityki dla akcji UTH."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (
            256,
            256,
        ),
    ) -> None:
        super().__init__()

        if type(input_size) is not int:
            raise TypeError(
                "input_size must be an integer"
            )

        if input_size <= 0:
            raise ValueError(
                "input_size must be positive"
            )

        if not hidden_sizes:
            raise ValueError(
                "hidden_sizes cannot be empty"
            )

        if any(
            type(size) is not int
            for size in hidden_sizes
        ):
            raise TypeError(
                "hidden_sizes must contain integers"
            )

        if any(
            size <= 0
            for size in hidden_sizes
        ):
            raise ValueError(
                "hidden_sizes must contain positive values"
            )

        self._input_size = input_size
        self._hidden_sizes = tuple(
            hidden_sizes
        )

        layers: list[nn.Module] = []

        previous_size = input_size

        for hidden_size in self._hidden_sizes:
            layers.extend(
                [
                    nn.Linear(
                        previous_size,
                        hidden_size,
                    ),
                    nn.ReLU(),
                ]
            )

            previous_size = hidden_size

        layers.append(
            nn.Linear(
                previous_size,
                ACTION_COUNT,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def hidden_sizes(
        self,
    ) -> tuple[int, ...]:
        return self._hidden_sizes

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            state
        )