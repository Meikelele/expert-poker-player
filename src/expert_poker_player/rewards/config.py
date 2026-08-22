from enum import Enum

from expert_poker_player.rewards.net_profit import (
    NetProfitReward,
)
from expert_poker_player.rewards.protocol import (
    RewardFunction,
)
from expert_poker_player.rewards.stake_scaled import (
    StakeScaledNetProfitReward,
)


class RewardType(str, Enum):
    """Dostępne warianty funkcji nagrody."""

    NET_PROFIT = "net_profit"
    STAKE_SCALED_NET_PROFIT = (
        "stake_scaled_net_profit"
    )


def build_reward_function(
    reward_type: RewardType,
) -> RewardFunction:
    """Tworzy funkcję nagrody odpowiadającą konfiguracji."""

    if not isinstance(
        reward_type,
        RewardType,
    ): # type: ignore
        raise TypeError(
            "reward_type must be an instance "
            "of RewardType"
        )

    if reward_type is RewardType.NET_PROFIT:
        return NetProfitReward()

    return StakeScaledNetProfitReward()