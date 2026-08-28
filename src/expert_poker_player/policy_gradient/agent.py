import random

import torch

from expert_poker_player.policy_gradient.network import (
    PolicyNetwork,
)
from expert_poker_player.rl.actions import (
    ACTION_ORDER,
    action_from_index,
    legal_action_mask,
    mask_action_values,
)
from expert_poker_player.state_representation import (
    StateEncoder,
)
from expert_poker_player.uth import (
    Action,
    UTHObservation,
)


class PolicyGradientAgent:
    """Agent wybierający akcje na podstawie polityki neuronowej."""

    def __init__(
        self,
        *,
        policy_network: PolicyNetwork,
        state_encoder: StateEncoder,
        deterministic: bool = True,
        seed: int | None = None,
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
            state_encoder,
            StateEncoder,
        ):  # type: ignore
            raise TypeError(
                "state_encoder must satisfy "
                "StateEncoder"
            )

        if (
            policy_network.input_size
            != state_encoder.output_size
        ):
            raise ValueError(
                "policy_network input size must match "
                "state encoder output size"
            )

        if type(deterministic) is not bool:
            raise TypeError(
                "deterministic must be a boolean"
            )

        self._policy_network = policy_network
        self._state_encoder = state_encoder
        self._deterministic = deterministic
        self._rng = random.Random(
            seed
        )

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @deterministic.setter
    def deterministic(
        self,
        value: bool,
    ) -> None:
        if type(value) is not bool:
            raise TypeError(
                "deterministic must be a boolean"
            )

        self._deterministic = value

    def select_action(
        self,
        observation: UTHObservation,
    ) -> Action:
        if not isinstance(
            observation,
            UTHObservation,
        ):  # type: ignore
            raise TypeError(
                "observation must be an instance "
                "of UTHObservation"
            )

        if observation.terminated:
            raise ValueError(
                "cannot select an action "
                "for a terminal observation"
            )

        probabilities = self._masked_action_probabilities(
            observation
        )

        if self._deterministic:
            action_index = int(
                torch.argmax(
                    probabilities
                ).item()
            )

            return action_from_index(
                action_index
            )

        return self._sample_action(
            probabilities,
            observation,
        )

    def action_probabilities(
        self,
        observation: UTHObservation,
    ) -> tuple[float, ...]:
        """
        Zwraca maskowany rozkład prawdopodobieństwa akcji.

        Metoda diagnostyczna używana do inspekcji polityki, niezależna
        od trybu `deterministic`. Nielegalne akcje mają
        prawdopodobieństwo dokładnie 0.0.
        """

        if not isinstance(
            observation,
            UTHObservation,
        ):  # type: ignore
            raise TypeError(
                "observation must be an instance "
                "of UTHObservation"
            )

        if observation.terminated:
            raise ValueError(
                "cannot compute action probabilities "
                "for a terminal observation"
            )

        probabilities = self._masked_action_probabilities(
            observation
        )

        return tuple(
            float(value)
            for value in probabilities.detach().cpu().tolist()  # type: ignore[reportUnknownMemberType]
        )

    def _masked_action_probabilities(
        self,
        observation: UTHObservation,
    ) -> torch.Tensor:
        """Liczy maskowany softmax logitów sieci dla obserwacji."""

        state = self._state_encoder.encode(
            observation
        )

        device = next(
            self._policy_network.parameters()
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
            logits = self._policy_network(
                state_tensor
            )

            masked_logits = mask_action_values(
                logits,
                action_mask,
            )

            probabilities = torch.softmax(
                masked_logits,
                dim=1,
            ).squeeze(0)

        return probabilities

    def _sample_action(
        self,
        probabilities: torch.Tensor,
        observation: UTHObservation,
    ) -> Action:
        probabilities_cpu = (
            probabilities
            .detach()
            .cpu()
        )

        legal_actions = [
            action
            for action in ACTION_ORDER
            if action in observation.legal_actions
        ]

        weights = [
            float(
                probabilities_cpu[
                    index
                ].item()
            )
            for index, action in enumerate(
                ACTION_ORDER
            )
            if action in observation.legal_actions
        ]

        return self._rng.choices(
            legal_actions,
            weights=weights,
            k=1,
        )[0]