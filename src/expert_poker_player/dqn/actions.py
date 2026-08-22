from types import MappingProxyType
from typing import (
    Final,
    Mapping,
)

import torch

from expert_poker_player.uth import (
    Action,
    UTHObservation,
)


ACTION_ORDER: Final[tuple[Action, ...]] = tuple(
    Action
)

ACTION_COUNT: Final[int] = len(
    ACTION_ORDER
)

_ACTION_TO_INDEX: Final[
    Mapping[Action, int]
] = MappingProxyType(
    {
        action: index
        for index, action in enumerate(
            ACTION_ORDER
        )
    }
)


def action_to_index(
    action: Action,
) -> int:
    """Zwraca indeks odpowiadający akcji UTH."""

    if not isinstance(
        action,
        Action,
    ): # type: ignore
        raise TypeError(
            "action must be an instance of Action"
        )

    return _ACTION_TO_INDEX[action]


def action_from_index(
    index: int,
) -> Action:
    """Zwraca akcję UTH odpowiadającą indeksowi."""

    if type(index) is not int:
        raise TypeError(
            "index must be an integer"
        )

    if not 0 <= index < ACTION_COUNT:
        raise ValueError(
            "index must reference an available action"
        )

    return ACTION_ORDER[index]


def legal_action_mask(
    observation: UTHObservation,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Buduje maskę legalnych akcji dla stanu decyzyjnego."""

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
            "cannot build an action mask "
            "for a terminal observation"
        )

    return torch.tensor(
        [
            action in observation.legal_actions
            for action in ACTION_ORDER
        ],
        dtype=torch.bool,
        device=device,
    )


def mask_q_values(
    q_values: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Maskuje wartości Q odpowiadające nielegalnym akcjom."""

    if not isinstance(
        q_values,
        torch.Tensor,
    ): # type: ignore
        raise TypeError(
            "q_values must be a torch.Tensor"
        )

    if not isinstance(
        action_mask,
        torch.Tensor,
    ): # type: ignore
        raise TypeError(
            "action_mask must be a torch.Tensor"
        )

    if action_mask.dtype is not torch.bool:
        raise TypeError(
            "action_mask must have boolean dtype"
        )

    if q_values.shape != action_mask.shape:
        raise ValueError(
            "q_values and action_mask "
            "must have the same shape"
        )

    if q_values.shape[-1] != ACTION_COUNT:
        raise ValueError(
            "last dimension must match "
            "the action count"
        )

    return q_values.masked_fill(
        ~action_mask,
        float("-inf"),
    )