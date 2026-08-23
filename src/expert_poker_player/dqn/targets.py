import math

import torch

from expert_poker_player.dqn.actions import (
    ACTION_COUNT,
    mask_q_values,
)
from expert_poker_player.dqn.network import (
    QNetwork,
)
from expert_poker_player.dqn.replay import (
    Transition,
)


def compute_bellman_targets(
    transitions: tuple[Transition, ...],
    *,
    target_network: QNetwork,
    gamma: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Oblicza wartości docelowe Bellmana dla batcha DQN."""

    if not transitions:
        raise ValueError(
            "transitions cannot be empty"
        )

    if not all(
        isinstance(
            transition,
            Transition,
        ) # type: ignore
        for transition in transitions
    ):
        raise TypeError(
            "transitions must contain "
            "only Transition values"
        )

    if not isinstance(
        target_network,
        QNetwork,
    ): # type: ignore
        raise TypeError(
            "target_network must be an instance "
            "of QNetwork"
        )

    if not isinstance(
        gamma,
        (int, float),
    ): # type: ignore
        raise TypeError(
            "gamma must be a number"
        )

    gamma = float(
        gamma
    )

    if (
        not math.isfinite(gamma)
        or not 0.0 <= gamma <= 1.0
    ):
        raise ValueError(
            "gamma must be between 0 and 1"
        )

    if device is None:
        device = next(
            target_network.parameters()
        ).device

    rewards = torch.tensor(
        [
            transition.reward
            for transition in transitions
        ],
        dtype=torch.float32,
        device=device,
    )

    targets = rewards.clone()

    non_terminal_indices = [
        index
        for index, transition in enumerate(
            transitions
        )
        if not transition.terminated
    ]

    if not non_terminal_indices:
        return targets

    next_states = torch.tensor(
        [
            transitions[index].next_state
            for index in non_terminal_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    next_action_masks = torch.tensor(
        [
            transitions[index].next_action_mask
            for index in non_terminal_indices
        ],
        dtype=torch.bool,
        device=device,
    )

    if next_states.shape[1] != target_network.input_size:
        raise ValueError(
            "next_state size must match "
            "target network input size"
        )

    if next_action_masks.shape[1] != ACTION_COUNT:
        raise ValueError(
            "next action mask must match "
            "the action count"
        )

    with torch.no_grad():
        next_q_values = target_network(
            next_states
        )

        masked_q_values = mask_q_values(
            next_q_values,
            next_action_masks,
        )

        max_next_q_values = torch.max(
            masked_q_values,
            dim=1,
        ).values

    target_indices = torch.tensor(
        non_terminal_indices,
        dtype=torch.long,
        device=device,
    )

    targets[target_indices] = (
        targets[target_indices]
        + gamma
        * max_next_q_values
    )

    return targets