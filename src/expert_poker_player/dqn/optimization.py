import math

import torch
from torch import nn

from expert_poker_player.dqn.network import (
    QNetwork,
)
from expert_poker_player.dqn.replay import (
    Transition,
)
from expert_poker_player.dqn.targets import (
    compute_bellman_targets,
)


class DQNOptimizer:
    """Wykonuje aktualizacje parametrów sieci policy DQN."""

    def __init__(
        self,
        *,
        policy_network: QNetwork,
        target_network: QNetwork,
        learning_rate: float,
        gamma: float,
    ) -> None:
        if not isinstance(
            policy_network,
            QNetwork,
        ): # type: ignore
            raise TypeError(
                "policy_network must be an instance "
                "of QNetwork"
            )

        if not isinstance(
            target_network,
            QNetwork,
        ): # type: ignore
            raise TypeError(
                "target_network must be an instance "
                "of QNetwork"
            )

        if (
            policy_network.input_size
            != target_network.input_size
        ):
            raise ValueError(
                "policy and target networks must "
                "have the same input size"
            )

        if not isinstance(
            learning_rate,
            (int, float),
        ): # type: ignore
            raise TypeError(
                "learning_rate must be a number"
            )

        learning_rate = float(
            learning_rate
        )

        if (
            not math.isfinite(learning_rate)
            or learning_rate <= 0.0
        ):
            raise ValueError(
                "learning_rate must be positive "
                "and finite"
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

        self._policy_network = policy_network
        self._target_network = target_network
        self._gamma = gamma

        self._optimizer = torch.optim.Adam(
            policy_network.parameters(),
            lr=learning_rate,
        )

        self._loss_function = nn.SmoothL1Loss()

    def optimize(
        self,
        transitions: tuple[Transition, ...],
    ) -> float:
        """Wykonuje jeden krok optymalizacji policy network."""

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

        device = next(
            self._policy_network.parameters()
        ).device

        target_device = next(
            self._target_network.parameters()
        ).device

        if device != target_device:
            raise ValueError(
                "policy and target networks must "
                "be on the same device"
            )

        states = torch.tensor(
            [
                transition.state
                for transition in transitions
            ],
            dtype=torch.float32,
            device=device,
        )

        if states.shape[1] != self._policy_network.input_size:
            raise ValueError(
                "state size must match "
                "policy network input size"
            )

        action_indices = torch.tensor(
            [
                transition.action_index
                for transition in transitions
            ],
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)

        q_values = self._policy_network(
            states
        )

        selected_q_values = q_values.gather(
            1,
            action_indices,
        ).squeeze(1)

        targets = compute_bellman_targets(
            transitions,
            target_network=self._target_network,
            gamma=self._gamma,
            device=device,
        )

        loss = self._loss_function(
            selected_q_values,
            targets,
        )

        self._optimizer.zero_grad()

        loss.backward()

        self._optimizer.step() # type: ignore

        return float(
            loss.item()
        )

    def sync_target_network(
        self,
    ) -> None:
        """Kopiuje parametry policy network do target network."""

        self._target_network.load_state_dict(
            self._policy_network.state_dict()
        )