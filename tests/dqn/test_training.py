import torch

from expert_poker_player.dqn import (
    DQNConfig,
    DQNTrainingResult,
    train_dqn,
)
from expert_poker_player.dqn.network import QNetwork
from expert_poker_player.rewards import (
    NetProfitReward,
    StakeScaledNetProfitReward,
)
from expert_poker_player.rewards.protocol import RewardFunction
from expert_poker_player.state_representation import (
    FeatureStateEncoder,
    RawStateEncoder,
)
from expert_poker_player.state_representation.protocol import StateEncoder


def build_test_config(
    *,
    seed: int = 42,
) -> DQNConfig:
    return DQNConfig(
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=2,
        replay_capacity=32,
        warmup_steps=2,
        target_sync_interval=2,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=10,
        hidden_sizes=(16,),
        training_episodes=8,
        seed=seed,
    )
def test_training_runs_multiple_episodes() -> None:
    config = build_test_config()

    result = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    assert isinstance(
        result,
        DQNTrainingResult,
    )

    assert len(
        result.episode_stats
    ) == config.training_episodes

    assert result.total_steps > 0
    assert result.optimizer_updates > 0

def test_episode_step_counts_match_uth_decision_horizon() -> None:
    result = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=build_test_config(),
    )

    assert all(
        1 <= stats.steps <= 3
        for stats in result.episode_stats
    )

import pytest


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
    result = train_dqn(
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
    result = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=reward_function,
        config=build_test_config(),
    )

    assert result.total_steps > 0

def test_training_applies_epsilon_schedule() -> None:
    config = build_test_config()

    result = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    epsilons = [
        stats.epsilon
        for stats in result.episode_stats
    ]

    assert all(
        left >= right
        for left, right in zip(
            epsilons,
            epsilons[1:],
        )
    )

def test_same_seed_reproduces_short_training_run() -> None:
    config = build_test_config(
        seed=123
    )

    first = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    second = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    assert (
        first.episode_stats
        == second.episode_stats
    )

    first_parameters = list(
        first.policy_network.parameters()
    )

    second_parameters = list(
        second.policy_network.parameters()
    )

    assert all(
        torch.equal(
            first_parameter,
            second_parameter,
        )
        for first_parameter, second_parameter in zip(
            first_parameters,
            second_parameters,
        )
    )

def test_training_records_finite_losses() -> None:
    result = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=build_test_config(),
    )

    losses = [
        stats.mean_loss
        for stats in result.episode_stats
        if stats.mean_loss is not None
    ]

    assert losses

    assert all(
        torch.isfinite(
            torch.tensor(loss)
        )
        for loss in losses
    )

def test_warmup_steps_blocks_optimizer_updates() -> None:
    config = DQNConfig(
        batch_size=2,
        replay_capacity=32,
        warmup_steps=10_000,
        target_sync_interval=2,
        hidden_sizes=(16,),
        training_episodes=8,
        seed=1,
    )

    result = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    assert result.total_steps > 0
    assert result.optimizer_updates == 0

    assert all(
        stats.mean_loss is None
        for stats in result.episode_stats
    )

def test_target_network_syncs_on_every_update_when_interval_is_one() -> None:
    config = DQNConfig(
        batch_size=2,
        replay_capacity=32,
        warmup_steps=2,
        target_sync_interval=1,
        hidden_sizes=(16,),
        training_episodes=8,
        seed=1,
    )

    result = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    assert result.optimizer_updates > 0

    assert all(
        torch.equal(
            policy_parameter,
            target_parameter,
        )
        for policy_parameter, target_parameter in zip(
            result.policy_network.parameters(),
            result.target_network.parameters(),
        )
    )

def test_target_network_does_not_sync_before_interval_elapses() -> None:
    config = DQNConfig(
        batch_size=2,
        replay_capacity=32,
        warmup_steps=2,
        target_sync_interval=1_000_000,
        hidden_sizes=(16,),
        training_episodes=8,
        seed=1,
    )

    result = train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
    )

    assert result.optimizer_updates > 0

    assert any(
        not torch.equal(
            policy_parameter,
            target_parameter,
        )
        for policy_parameter, target_parameter in zip(
            result.policy_network.parameters(),
            result.target_network.parameters(),
        )
    )

def test_training_notifies_observer_after_each_episode() -> None:
    config = build_test_config()

    completed_episodes: list[int] = []

    def observer(
        episode: int,
        network: QNetwork,
    ) -> None:
        assert isinstance(
            network,
            QNetwork,
        )

        completed_episodes.append(
            episode
        )

    train_dqn(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
        on_episode_completed=observer,
    )

    assert completed_episodes == list(
        range(
            1,
            config.training_episodes + 1,
        )
    )
