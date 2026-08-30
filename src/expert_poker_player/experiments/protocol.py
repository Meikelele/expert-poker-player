from dataclasses import dataclass
from enum import Enum

from expert_poker_player.rewards import RewardType
from expert_poker_player.state_representation import (
    StateRepresentation,
)


class RLAlgorithm(str, Enum):
    DQN = "dqn"
    REINFORCE = "reinforce"


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    algorithm: RLAlgorithm
    state_representation: StateRepresentation
    reward_type: RewardType

    def __post_init__(self) -> None:
        if not isinstance(
            self.algorithm,
            RLAlgorithm,
        ):  # type: ignore
            raise TypeError(
                "algorithm must be an instance of RLAlgorithm"
            )

        if not isinstance(
            self.state_representation,
            StateRepresentation,
        ):  # type: ignore
            raise TypeError(
                "state_representation must be an instance "
                "of StateRepresentation"
            )

        if not isinstance(
            self.reward_type,
            RewardType,
        ):  # type: ignore
            raise TypeError(
                "reward_type must be an instance of RewardType"
            )

    @property
    def name(self) -> str:
        return "_".join(
            (
                self.algorithm.value,
                self.state_representation.value,
                self.reward_type.value,
            )
        )


FINAL_VARIANTS = tuple(
    ExperimentVariant(
        algorithm=algorithm,
        state_representation=state_representation,
        reward_type=reward_type,
    )
    for algorithm in RLAlgorithm
    for state_representation in StateRepresentation
    for reward_type in RewardType
)


FINAL_TRAINING_SEED_SOURCE = 20260828

FINAL_TRAINING_SEEDS = (
    6762274154327289452,
    3828104833377202082,
    529668385939893067,
    9098740095607130738,
    7131037655593136768,
)