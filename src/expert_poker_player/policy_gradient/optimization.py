import math

import torch
from torch.distributions import Categorical

from expert_poker_player.policy_gradient.network import (
    PolicyNetwork,
)
from expert_poker_player.policy_gradient.returns import (
    compute_discounted_returns,
)
from expert_poker_player.policy_gradient.trajectory import (
    Trajectory,
)
from expert_poker_player.rl.actions import (
    mask_action_values,
)


class ReinforceOptimizer:
    """Wykonuje aktualizację policy network metodą REINFORCE."""

    def __init__(
        self,
        *,
        policy_network: PolicyNetwork,
        learning_rate: float,
        gamma: float,
    ) -> None:
        if not isinstance(
            policy_network,
            PolicyNetwork,
        ):  # type: ignore
            raise TypeError(
                "policy_network must be an instance "
                "of PolicyNetwork"
            )

        if not isinstance(
            learning_rate,
            (int, float),
        ):  # type: ignore
            raise TypeError(
                "learning_rate must be a number"
            )

        learning_rate = float(
            learning_rate
        )

        if (
            not math.isfinite(
                learning_rate
            )
            or learning_rate <= 0.0
        ):
            raise ValueError(
                "learning_rate must be positive "
                "and finite"
            )

        if not isinstance(
            gamma,
            (int, float),
        ):  # type: ignore
            raise TypeError(
                "gamma must be a number"
            )

        gamma = float(
            gamma
        )

        if (
            not math.isfinite(
                gamma
            )
            or not 0.0 <= gamma <= 1.0
        ):
            raise ValueError(
                "gamma must be between 0 and 1"
            )

        self._policy_network = (
            policy_network
        )

        self._gamma = gamma

        self._optimizer = (
            torch.optim.Adam(
                policy_network.parameters(),
                lr=learning_rate,
            )
        )

    def optimize(
        self,
        trajectory: Trajectory,
    ) -> float:
        """Wykonuje jeden update REINFORCE dla pełnego epizodu."""

        if not isinstance(
            trajectory,
            Trajectory,
        ):  # type: ignore
            raise TypeError(
                "trajectory must be an instance "
                "of Trajectory"
            )

        if any(
            len(step.state)
            != self._policy_network.input_size
            for step in trajectory.steps
        ):
            raise ValueError(
                "state size must match "
                "policy network input size"
            )

        device = next(
            self._policy_network.parameters()
        ).device

        states = torch.tensor(
            [
                step.state
                for step in trajectory.steps
            ],
            dtype=torch.float32,
            device=device,
        )

        action_indices = torch.tensor(
            [
                step.action_index
                for step in trajectory.steps
            ],
            dtype=torch.long,
            device=device,
        )

        action_masks = torch.tensor(
            [
                step.action_mask
                for step in trajectory.steps
            ],
            dtype=torch.bool,
            device=device,
        )

        discounted_returns = (
            compute_discounted_returns(
                tuple(
                    step.reward
                    for step in trajectory.steps
                ),
                gamma=self._gamma,
            )
        )

        returns_tensor = torch.tensor(
            discounted_returns,
            dtype=torch.float32,
            device=device,
        )

        logits = self._policy_network(
            states
        )

        masked_logits = mask_action_values(
            logits,
            action_masks,
        )

        distribution = Categorical(
            logits=masked_logits
        )

        log_probabilities = ( # type: ignore
            distribution.log_prob(
                action_indices
            )
        )

        loss = -( # type: ignore
            log_probabilities
            * returns_tensor
        ).sum()

        self._optimizer.zero_grad()

        loss.backward() # type: ignore

        self._optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        return float(
            loss.item() # type: ignore
        )