import pytest

from expert_poker_player.rewards import (
    NetProfitReward,
    RewardFunction,
    RewardType,
    StakeScaledNetProfitReward,
    build_reward_function,
)


def test_net_profit_reward_type_has_stable_value() -> None:
    assert (
        RewardType.NET_PROFIT.value
        == "net_profit"
    )


def test_scaled_reward_type_has_stable_value() -> None:
    assert (
        RewardType.STAKE_SCALED_NET_PROFIT.value
        == "stake_scaled_net_profit"
    )


def test_builds_net_profit_reward() -> None:
    reward_function = build_reward_function(
        RewardType.NET_PROFIT
    )

    assert isinstance(
        reward_function,
        NetProfitReward,
    )

    assert isinstance(
        reward_function,
        RewardFunction,
    )


def test_builds_stake_scaled_reward() -> None:
    reward_function = build_reward_function(
        RewardType.STAKE_SCALED_NET_PROFIT
    )

    assert isinstance(
        reward_function,
        StakeScaledNetProfitReward,
    )

    assert isinstance(
        reward_function,
        RewardFunction,
    )

@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (
            "net_profit",
            RewardType.NET_PROFIT,
        ),
        (
            "stake_scaled_net_profit",
            RewardType.STAKE_SCALED_NET_PROFIT,
        ),
    ],
)
def test_reward_type_can_be_restored_from_string(
    raw_value: str,
    expected: RewardType,
) -> None:
    assert RewardType(raw_value) is expected

def test_builder_rejects_invalid_reward_type() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "reward_type must be an instance "
            "of RewardType"
        ),
    ):
        build_reward_function(
            "net_profit"  # type: ignore[arg-type]
        )

    