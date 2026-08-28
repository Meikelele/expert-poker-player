import pytest
import torch

from expert_poker_player.evaluation import (
    SimulationConfig,
    SimulationResult,
    run_simulation,
)
from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
    train_policy_gradient,
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
    config = PolicyGradientConfig(
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=4,
        hidden_sizes=(16,),
        training_episodes=8,
        seed=123,
    )

    first = train_policy_gradient(
        state_encoder=build_state_encoder(
            state_representation
        ),
        reward_function=build_reward_function(
            reward_type
        ),
        config=config,
    )

    second = train_policy_gradient(
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
    first = train_policy_gradient(
        state_encoder=build_state_encoder(
            StateRepresentation.RAW
        ),
        reward_function=build_reward_function(
            RewardType.NET_PROFIT
        ),
        config=PolicyGradientConfig(
            hidden_sizes=(16,),
            training_episodes=8,
            seed=123,
        ),
    )

    second = train_policy_gradient(
        state_encoder=build_state_encoder(
            StateRepresentation.RAW
        ),
        reward_function=build_reward_function(
            RewardType.NET_PROFIT
        ),
        config=PolicyGradientConfig(
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

def test_trained_agent_works_with_run_simulation() -> None:
    training_result = train_policy_gradient(
        state_encoder=build_state_encoder(
            StateRepresentation.RAW
        ),
        reward_function=build_reward_function(
            RewardType.NET_PROFIT
        ),
        config=PolicyGradientConfig(
            hidden_sizes=(16,),
            training_episodes=4,
            seed=7,
        ),
    )

    agent = training_result.agent

    agent.deterministic = True
    training_result.policy_network.eval()

    simulation = run_simulation(
        agent=agent,
        config=SimulationConfig(
            deck_seeds=(
                101,
                202,
                303,
            ),
        ),
    )

    assert isinstance(
        simulation,
        SimulationResult,
    )

    assert simulation.round_count == 3

def test_training_result_returns_stochastic_agent() -> None:
    result = train_policy_gradient(
        state_encoder=build_state_encoder(
            StateRepresentation.RAW
        ),
        reward_function=build_reward_function(
            RewardType.NET_PROFIT
        ),
        config=PolicyGradientConfig(
            hidden_sizes=(16,),
            training_episodes=2,
            seed=7,
        ),
    )

    assert not result.agent.deterministic

