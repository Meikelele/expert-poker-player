from fractions import Fraction
from typing import Final

from expert_poker_player.rewards.net_profit import (
    NetProfitReward,
)
from expert_poker_player.rewards.protocol import (
    RewardValue,
)
from expert_poker_player.uth import (
    ANTE_STAKE,
    BLIND_STAKE,
    VALID_PLAY_MULTIPLIERS,
    StepResult,
)


MAX_TOTAL_STAKE: Final[Fraction] = (
    ANTE_STAKE
    + BLIND_STAKE
    + Fraction(
        max(VALID_PLAY_MULTIPLIERS)
    )
)


class StakeScaledNetProfitReward:
    """Skaluje terminalny zysk netto względem maksymalnej stawki."""

    def __init__(self) -> None:
        self._net_profit_reward = NetProfitReward()

    def calculate_reward(
        self,
        step_result: StepResult,
    ) -> RewardValue:
        reward = self._net_profit_reward.calculate_reward(
            step_result
        )

        return reward / float(
            MAX_TOTAL_STAKE
        )