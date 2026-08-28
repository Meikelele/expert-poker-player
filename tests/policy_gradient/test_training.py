import pytest
import torch

from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
    PolicyGradientTrainingResult,
    train_policy_gradient,
)
from expert_poker_player.rewards import (
    NetProfitReward,
    StakeScaledNetProfitReward,
)
from expert_poker_player.rewards.protocol import (
    RewardFunction,
)
from expert_poker_player.state_representation import (
    FeatureStateEncoder,
    RawStateEncoder,
    StateEncoder,
)


def build_test_config(
    *,
    seed: int = 42,
) -> PolicyGradientConfig:
    return PolicyGradientConfig(
        learning_rate=1e-3,
        gamma=1.0,
        batch_size=4,
        hidden_sizes=(16,),
        training_episodes=8,
        seed=seed,
    )


def test_training_runs_multiple_episodes() -> None:
    config = build_test_config()

    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    assert isinstance(
        result,
        PolicyGradientTrainingResult,
    )

    assert len(
        result.episode_stats
    ) == config.training_episodes

    assert result.total_steps > 0

    assert result.optimizer_updates == 2


def test_episode_step_counts_match_uth_decision_horizon() -> None:
    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=build_test_config(),
    )

    assert all(
        1 <= stats.steps <= 3
        for stats in result.episode_stats
    )


def test_training_records_finite_losses() -> None:
    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=build_test_config(),
    )

    assert all(
        torch.isfinite(
            torch.tensor(
                stats.loss
            )
        )
        for stats in result.update_stats
    )


@pytest.mark.parametrize(
    "encoder",
    [
        RawStateEncoder(),
        FeatureStateEncoder(),
    ],
)
def test_training_supports_state_representations(
    encoder: StateEncoder,
) -> None:
    result = train_policy_gradient(
        state_encoder=encoder,
        reward_function=NetProfitReward(),
        config=build_test_config(),
    )

    assert (
        result.policy_network.input_size
        == encoder.output_size
    )


@pytest.mark.parametrize(
    "reward_function",
    [
        NetProfitReward(),
        StakeScaledNetProfitReward(),
    ],
)
def test_training_supports_reward_functions(
    reward_function: RewardFunction,
) -> None:
    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=reward_function,
        config=build_test_config(),
    )

    assert result.total_steps > 0

def test_training_batches_episodes_into_updates() -> None:
    config = build_test_config()

    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    assert result.optimizer_updates == 2


def test_training_optimizes_partial_final_batch() -> None:
    config = PolicyGradientConfig(
        batch_size=4,
        hidden_sizes=(16,),
        training_episodes=10,
        seed=42,
    )

    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    assert result.optimizer_updates == 3

    assert [
        stats.batch_size
        for stats in result.update_stats
    ] == [
        4,
        4,
        2,
    ]

    assert [
        (
            stats.first_episode,
            stats.last_episode,
        )
        for stats in result.update_stats
    ] == [
        (0, 3),
        (4, 7),
        (8, 9),
    ]

