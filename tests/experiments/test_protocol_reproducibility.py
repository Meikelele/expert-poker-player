import torch

from expert_poker_player.dqn import (
    DQNConfig,
    train_dqn,
)
from expert_poker_player.experiments import (
    DQNPeriodicEvaluator,
    PolicyGradientPeriodicEvaluator,
    build_validation_schedule,
)
from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
    train_policy_gradient,
)
from expert_poker_player.rewards import (
    NetProfitReward,
)
from expert_poker_player.state_representation import (
    RawStateEncoder,
)


def test_dqn_periodic_evaluation_does_not_change_training() -> None:
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

    baseline_encoder = RawStateEncoder()

    baseline = train_dqn(
        state_encoder=baseline_encoder,
        reward_function=NetProfitReward(),
        config=config,
    )

    observed_encoder = RawStateEncoder()

    evaluator = DQNPeriodicEvaluator(
        state_encoder=observed_encoder,
        schedule=build_validation_schedule(3),
        checkpoints=(
            2,
            4,
            8,
        ),
    )

    observed = train_dqn(
        state_encoder=observed_encoder,
        reward_function=NetProfitReward(),
        config=config,
        on_episode_completed=evaluator,
    )

    assert baseline.episode_stats == observed.episode_stats
    assert baseline.total_steps == observed.total_steps

    assert (
        baseline.optimizer_updates
        == observed.optimizer_updates
    )

    assert all(
        torch.equal(
            baseline_parameter,
            observed_parameter,
        )
        for baseline_parameter, observed_parameter in zip(
            baseline.policy_network.parameters(),
            observed.policy_network.parameters(),
        )
    )

    assert all(
        torch.equal(
            baseline_parameter,
            observed_parameter,
        )
        for baseline_parameter, observed_parameter in zip(
            baseline.target_network.parameters(),
            observed.target_network.parameters(),
        )
    )

def test_pg_periodic_evaluation_does_not_change_training() -> None:
    config = PolicyGradientConfig(
        learning_rate=1e-3,
        gamma=1.0,
        batch_size=2,
        hidden_sizes=(16,),
        training_episodes=8,
        seed=123,
    )

    baseline_encoder = RawStateEncoder()

    baseline = train_policy_gradient(
        state_encoder=baseline_encoder,
        reward_function=NetProfitReward(),
        config=config,
        probe_states=None,
    )

    observed_encoder = RawStateEncoder()

    evaluator = PolicyGradientPeriodicEvaluator(
        state_encoder=observed_encoder,
        schedule=build_validation_schedule(3),
        checkpoints=(
            2,
            4,
            8,
        ),
    )

    observed = train_policy_gradient(
        state_encoder=observed_encoder,
        reward_function=NetProfitReward(),
        config=config,
        probe_states=None,
        on_episode_completed=evaluator,
    )

    assert baseline.episode_stats == observed.episode_stats

    assert baseline.update_stats == observed.update_stats

    assert baseline.total_steps == observed.total_steps

    assert (
        baseline.optimizer_updates
        == observed.optimizer_updates
    )

    assert all(
        torch.equal(
            baseline_parameter,
            observed_parameter,
        )
        for baseline_parameter, observed_parameter in zip(
            baseline.policy_network.parameters(),
            observed.policy_network.parameters(),
        )
    )

from expert_poker_player.experiments import (
    ExperimentRunSpec,
    ExperimentVariant,
    RLAlgorithm,
    run_experiment,
)
from expert_poker_player.rewards import (
    RewardType,
)
from expert_poker_player.state_representation import (
    StateRepresentation,
)


def test_same_experiment_spec_is_reproducible() -> None:
    spec = ExperimentRunSpec(
        variant=ExperimentVariant(
            algorithm=RLAlgorithm.DQN,
            state_representation=StateRepresentation.RAW,
            reward_type=RewardType.NET_PROFIT,
        ),
        training_seed=123,
        training_episodes=8,
        validation_schedule=(
            build_validation_schedule(3)
        ),
        validation_checkpoints=(
            4,
            8,
        ),
    )

    first = run_experiment(
        spec
    )

    second = run_experiment(
        spec
    )

    assert (
        first.summary.to_dict()
        == second.summary.to_dict()
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

def test_final_evaluation_does_not_change_trained_model() -> None:
    variant = ExperimentVariant(
        algorithm=RLAlgorithm.DQN,
        state_representation=StateRepresentation.RAW,
        reward_type=RewardType.NET_PROFIT,
    )

    common_kwargs = { # type: ignore
        "variant": variant,
        "training_seed": 123,
        "training_episodes": 8,
        "validation_schedule": (
            build_validation_schedule(3)
        ),
        "validation_checkpoints": (
            4,
            8,
        ),
    }

    without_final = run_experiment(
        ExperimentRunSpec(
            **common_kwargs,
        )
    )

    from expert_poker_player.experiments import (
        build_final_evaluation_schedule,
    )

    with_final = run_experiment(
        ExperimentRunSpec(
            **common_kwargs,
            final_evaluation_schedule=(
                build_final_evaluation_schedule(5)
            ),
        )
    )

    assert (
        without_final.summary.validation_curve
        == with_final.summary.validation_curve
    )

    assert all(
        torch.equal(
            without_parameter,
            with_parameter,
        )
        for without_parameter, with_parameter in zip(
            without_final.policy_network.parameters(),
            with_final.policy_network.parameters(),
        )
    )

