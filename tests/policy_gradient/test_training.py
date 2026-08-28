import pytest
import torch

from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
    PolicyGradientTrainingResult,
    train_policy_gradient,
)
from expert_poker_player.policy_gradient.network import PolicyNetwork
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
        stats.update
        for stats in result.update_stats
    ] == [
        1,
        2,
        3,
    ]

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

def test_training_notifies_observer_with_completed_episode_count() -> None:
    config = PolicyGradientConfig(
        batch_size=2,
        hidden_sizes=(16,),
        training_episodes=4,
        seed=42,
    )

    completed_episodes: list[int] = []

    def observer(
        episode: int,
        network: PolicyNetwork,
    ) -> None:
        assert isinstance(
            network,
            PolicyNetwork,
        )

        completed_episodes.append(
            episode
        )

    train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
        on_episode_completed=observer,
    )

    assert completed_episodes == [
        1,
        2,
        3,
        4,
    ]

def test_pg_observer_runs_after_batch_update() -> None:
    config = PolicyGradientConfig(
        learning_rate=1e-3,
        gamma=1.0,
        batch_size=2,
        hidden_sizes=(16,),
        training_episodes=2,
        seed=42,
    )

    observed_parameters: list[
        tuple[torch.Tensor, ...]
    ] = []

    def observer(
        episode: int,
        network: PolicyNetwork,
    ) -> None:
        if episode != 2:
            return

        observed_parameters.append(
            tuple(
                parameter.detach().clone()
                for parameter in network.parameters()
            )
        )

    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
        on_episode_completed=observer,
    )

    assert len(
        observed_parameters
    ) == 1

    final_parameters = tuple(
        parameter.detach().clone()
        for parameter in result.policy_network.parameters()
    )

    assert all(
        torch.equal(
            observed,
            final,
        )
        for observed, final in zip(
            observed_parameters[0],
            final_parameters,
        )
    )

def test_pg_final_observer_sees_partial_batch_update() -> None:
    config = PolicyGradientConfig(
        learning_rate=1e-3,
        gamma=1.0,
        batch_size=4,
        hidden_sizes=(16,),
        training_episodes=2,
        seed=42,
    )

    observed_parameters: list[
        tuple[torch.Tensor, ...]
    ] = []

    def observer(
        episode: int,
        network: PolicyNetwork,
    ) -> None:
        if episode != 2:
            return

        observed_parameters.append(
            tuple(
                parameter.detach().clone()
                for parameter in network.parameters()
            )
        )

    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
        on_episode_completed=observer,
    )

    assert len(
        observed_parameters
    ) == 1

    final_parameters = tuple(
        parameter.detach().clone()
        for parameter in result.policy_network.parameters()
    )

    assert all(
        torch.equal(
            observed,
            final,
        )
        for observed, final in zip(
            observed_parameters[0],
            final_parameters,
        )
    )

