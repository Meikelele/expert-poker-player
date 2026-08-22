from typing import cast

from expert_poker_player.rewards.protocol import (
    RewardValue,
)
from expert_poker_player.uth import (
    Settlement,
    StepResult,
)


class NetProfitReward:
    """Zwraca terminalny zysk netto w jednostkach Ante."""

    def calculate_reward(
        self,
        step_result: StepResult,
    ) -> RewardValue:
        if not isinstance(
            step_result,
            StepResult,
        ): # type: ignore
            raise TypeError(
                "step_result must be an instance "
                "of StepResult"
            )

        if not step_result.terminated:
            return 0.0

        settlement = cast(
            Settlement,
            step_result.settlement,
        )

        return float(
            settlement.total_net_profit
        )