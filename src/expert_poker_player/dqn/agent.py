import random

import torch

from expert_poker_player.dqn.actions import (
    ACTION_ORDER,
    action_from_index,
    legal_action_mask,
    mask_q_values,
)
from expert_poker_player.dqn.network import (
    QNetwork,
)
from expert_poker_player.state_representation import (
    StateEncoder,
)
from expert_poker_player.uth import (
    Action,
    UTHObservation,
)


class DQNAgent:
    """Agent wybierający akcje strategią epsilon-greedy."""

    def __init__(
        self,
        *,
        q_network: QNetwork,
        state_encoder: StateEncoder,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not isinstance(
            q_network,
            QNetwork,
        ): # type: ignore
            raise TypeError(
                "q_network must be an instance "
                "of QNetwork"
            )

        if not isinstance(
            state_encoder,
            StateEncoder,
        ): # type: ignore
            raise TypeError(
                "state_encoder must satisfy "
                "StateEncoder"
            )

        if (
            q_network.input_size
            != state_encoder.output_size
        ):
            raise ValueError(
                "q_network input size must match "
                "state encoder output size"
            )

        self._q_network = q_network
        self._state_encoder = state_encoder
        self._rng = random.Random(seed)

        self._epsilon = 0.0
        self.epsilon = epsilon

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(
        self,
        value: float,
    ) -> None:
        if not isinstance(
            value,
            (int, float),
        ): # type: ignore
            raise TypeError(
                "epsilon must be a number"
            )

        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1"
            )

        self._epsilon = value

    def select_action(
        self,
        observation: UTHObservation,
    ) -> Action:
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
                "cannot select an action "
                "for a terminal observation"
            )

        if self._rng.random() < self._epsilon:
            return self._select_random_action(
                observation
            )

        return self._select_greedy_action(
            observation
        )

    def _select_random_action(
        self,
        observation: UTHObservation,
    ) -> Action:
        legal_actions = [
            action
            for action in ACTION_ORDER
            if action in observation.legal_actions
        ]

        return self._rng.choice(
            legal_actions
        )

    def _select_greedy_action(
        self,
        observation: UTHObservation,
    ) -> Action:
        state = self._state_encoder.encode(
            observation
        )

        device = next(
            self._q_network.parameters()
        ).device

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        action_mask = legal_action_mask(
            observation,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self._q_network(
                state_tensor
            )

            masked_q_values = mask_q_values(
                q_values,
                action_mask,
            )

            action_index = int(
                torch.argmax(
                    masked_q_values,
                    dim=1,
                ).item()
            )

        return action_from_index(
            action_index
        )