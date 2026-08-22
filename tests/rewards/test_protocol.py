from expert_poker_player.rewards import (
    RewardFunction,
    RewardValue,
)
from expert_poker_player.uth import StepResult


class StubRewardFunction:
    def calculate_reward(
        self,
        step_result: StepResult,
    ) -> RewardValue:
        return 0.0


class MissingCalculateReward:
    pass


def test_compatible_reward_satisfies_protocol() -> None:
    reward_function = StubRewardFunction()

    assert isinstance(
        reward_function,
        RewardFunction,
    )


def test_incomplete_reward_does_not_satisfy_protocol() -> None:
    reward_function = MissingCalculateReward()

    assert not isinstance(
        reward_function,
        RewardFunction,
    )