import torch

from expert_poker_player.rl.actions import (
    ACTION_COUNT,
    ACTION_ORDER,
    action_from_index,
    action_to_index,
    legal_action_mask,
    mask_action_values,
)


def mask_q_values(
    q_values: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Maskuje wartości Q odpowiadające nielegalnym akcjom."""

    return mask_action_values(
        q_values,
        action_mask,
    )


__all__ = [
    "ACTION_COUNT",
    "ACTION_ORDER",
    "action_from_index",
    "action_to_index",
    "legal_action_mask",
    "mask_q_values",
]