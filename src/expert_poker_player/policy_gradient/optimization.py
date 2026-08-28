from dataclasses import dataclass
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


@dataclass(
    frozen=True,
    slots=True,
)
class BatchOptimizationResult:
    """Wynik jednego update'u REINFORCE dla batcha epizodów."""

    loss: float
    gradient_norm: float

    def __post_init__(self) -> None:
        if not isinstance(
            self.loss,
            float,
        ):  # type: ignore
            raise TypeError(
                "loss must be a float"
            )

        if not isinstance(
            self.gradient_norm,
            float,
        ):  # type: ignore
            raise TypeError(
                "gradient_norm must be a float"
            )

        if self.gradient_norm < 0.0:
            raise ValueError(
                "gradient_norm cannot be negative"
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
    ) -> BatchOptimizationResult:
        """Wykonuje jeden update REINFORCE dla pełnego epizodu."""

        if not isinstance(
            trajectory,
            Trajectory,
        ):  # type: ignore
            raise TypeError(
                "trajectory must be an instance "
                "of Trajectory"
            )

        return self.optimize_batch(
            (
                trajectory,
            )
        )

    def optimize_batch(
        self,
        trajectories: tuple[
            Trajectory,
            ...,
        ],
    ) -> BatchOptimizationResult:
        """Wykonuje jeden update REINFORCE dla wielu epizodów naraz."""

        if not isinstance(
            trajectories,
            tuple,
        ):  # type: ignore
            raise TypeError(
                "trajectories must be a tuple"
            )

        if not trajectories:
            raise ValueError(
                "trajectories cannot be empty"
            )

        if not all(
            isinstance(  # type: ignore
                trajectory,
                Trajectory,
            )
            for trajectory in trajectories
        ):
            raise TypeError(
                "trajectories must contain "
                "only Trajectory values"
            )

        if any(
            len(step.state)
            != self._policy_network.input_size
            for trajectory in trajectories
            for step in trajectory.steps
        ):
            raise ValueError(
                "state size must match "
                "policy network input size"
            )

        states: list[tuple[float, ...]] = []
        action_indices: list[int] = []
        action_masks: list[tuple[bool, ...]] = []
        returns: list[float] = []

        for trajectory in trajectories:
            discounted_returns = (
                compute_discounted_returns(
                    tuple(
                        step.reward
                        for step in trajectory.steps
                    ),
                    gamma=self._gamma,
                )
            )

            for step, discounted_return in zip(
                trajectory.steps,
                discounted_returns,
                strict=True,
            ):
                states.append(
                    step.state
                )

                action_indices.append(
                    step.action_index
                )

                action_masks.append(
                    step.action_mask
                )

                returns.append(
                    discounted_return
                )

        device = next(
            self._policy_network.parameters()
        ).device

        states_tensor = torch.tensor(
            states,
            dtype=torch.float32,
            device=device,
        )

        action_indices_tensor = torch.tensor(
            action_indices,
            dtype=torch.long,
            device=device,
        )

        action_masks_tensor = torch.tensor(
            action_masks,
            dtype=torch.bool,
            device=device,
        )

        returns_tensor = torch.tensor(
            returns,
            dtype=torch.float32,
            device=device,
        )

        logits = self._policy_network(
            states_tensor
        )

        masked_logits = mask_action_values(
            logits,
            action_masks_tensor,
        )

        distribution = Categorical(
            logits=masked_logits
        )

        log_probabilities = ( # type: ignore
            distribution.log_prob(
                action_indices_tensor
            )
        )

        loss = -( # type: ignore
            log_probabilities
            * returns_tensor
        ).sum() / len(
            trajectories
        )

        self._optimizer.zero_grad()

        loss.backward() # type: ignore

        gradient_norm = _compute_gradient_norm(
            self._policy_network
        )

        self._optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        return BatchOptimizationResult(
            loss=float(
                loss.item() # type: ignore
            ),
            gradient_norm=gradient_norm,
        )


def _compute_gradient_norm(
    policy_network: PolicyNetwork,
) -> float:
    """
    Liczy globalną normę L2 gradientów przed krokiem optymalizatora.

    Wywoływana po `loss.backward()`, a przed `optimizer.step()`, żeby
    opisywać gradienty wyprodukowane przez bieżący batch, zanim Adam
    zmieni parametry sieci.
    """

    squared_norm_sum = 0.0

    for parameter in policy_network.parameters():
        if parameter.grad is None:
            continue

        squared_norm_sum += float(
            parameter.grad.detach().norm(2).item() ** 2
        )

    return math.sqrt(
        squared_norm_sum
    )