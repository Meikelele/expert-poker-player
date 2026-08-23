import pytest
import torch

from expert_poker_player.dqn import (
    DQNConfig,
    train_dqn,
)
from expert_poker_player.rewards import (
    RewardType,
    build_reward_function,
)
from expert_poker_player.state_representation import (
    StateRepresentation,
    build_state_encoder,
)


@pytest.mark.parametrize(
    (
        "state_representation",
        "reward_type",
    ),
    [
        (
            StateRepresentation.RAW,
            RewardType.NET_PROFIT,
        ),
        (
            StateRepresentation.RAW,
            RewardType.STAKE_SCALED_NET_PROFIT,
        ),
        (
            StateRepresentation.FEATURES,
            RewardType.NET_PROFIT,
        ),
        (
            StateRepresentation.FEATURES,
            RewardType.STAKE_SCALED_NET_PROFIT,
        ),
    ],
)
def test_training_pipeline_is_deterministic(
    state_representation: StateRepresentation,
    reward_type: RewardType,
) -> None:
    config = DQNConfig(
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
        seed=123,
    )

    first = train_dqn(
        state_encoder=build_state_encoder(
            state_representation
        ),
        reward_function=build_reward_function(
            reward_type
        ),
        config=config,
    )

    second = train_dqn(
        state_encoder=build_state_encoder(
            state_representation
        ),
        reward_function=build_reward_function(
            reward_type
        ),
        config=config,
    )

    assert (
        first.episode_stats
        == second.episode_stats
    )

    assert (
        first.total_steps
        == second.total_steps
    )

    assert (
        first.optimizer_updates
        == second.optimizer_updates
    )

    assert all(
        torch.equal(
            first_parameter,
            second_parameter,
        )
        for first_parameter, second_parameter in zip(
            first.policy_network.parameters(),
            second.policy_network.parameters(),
        )
    )

def test_different_training_seeds_produce_different_models() -> None:
    first = train_dqn(
        state_encoder=build_state_encoder(
            StateRepresentation.RAW
        ),
        reward_function=build_reward_function(
            RewardType.NET_PROFIT
        ),
        config=DQNConfig(
            batch_size=2,
            replay_capacity=32,
            warmup_steps=2,
            target_sync_interval=2,
            hidden_sizes=(16,),
            training_episodes=8,
            seed=123,
        ),
    )

    second = train_dqn(
        state_encoder=build_state_encoder(
            StateRepresentation.RAW
        ),
        reward_function=build_reward_function(
            RewardType.NET_PROFIT
        ),
        config=DQNConfig(
            batch_size=2,
            replay_capacity=32,
            warmup_steps=2,
            target_sync_interval=2,
            hidden_sizes=(16,),
            training_episodes=8,
            seed=456,
        ),
    )

    assert any(
        not torch.equal(
            first_parameter,
            second_parameter,
        )
        for first_parameter, second_parameter in zip(
            first.policy_network.parameters(),
            second.policy_network.parameters(),
        )
    )