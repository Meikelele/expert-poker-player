from random import Random

from expert_poker_player.uth import (
    Action,
    UTHObservation,
)


class RandomAgent:
    """Agent wybierający losowo jedną z legalnych akcji."""

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:
        if seed is not None and type(seed) is not int:
            raise TypeError(
                "seed must be an integer or None"
            )

        self._random = Random(seed)

    def select_action(
        self,
        observation: UTHObservation,
    ) -> Action:
        """Wybiera losowo jedną z legalnych akcji."""

        if not isinstance(observation, UTHObservation): # type: ignore
            raise TypeError(
                "observation must be an instance "
                "of UTHObservation"
            )

        if observation.terminated:
            raise ValueError(
                "cannot select an action for "
                "a terminal observation"
            )

        legal_actions = tuple(
            action
            for action in Action
            if action in observation.legal_actions
        )

        return self._random.choice(legal_actions)