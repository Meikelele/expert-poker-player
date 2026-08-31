from expert_poker_player.experiments import (
    ExperimentRunSpec,
    ExperimentVariant,
    RLAlgorithm,
    build_final_evaluation_schedule,
    build_validation_schedule,
    run_experiment,
)
from expert_poker_player.rewards import RewardType
from expert_poker_player.state_representation import (
    StateRepresentation,
)


def test_dqn_experiment_builds_validation_curve() -> None:
    spec = ExperimentRunSpec(
        variant=ExperimentVariant(
            algorithm=RLAlgorithm.DQN,
            state_representation=(
                StateRepresentation.RAW
            ),
            reward_type=RewardType.NET_PROFIT,
        ),
        training_seed=123,
        training_episodes=4,
        validation_schedule=(
            build_validation_schedule(3)
        ),
        validation_checkpoints=(
            2,
            4,
        ),
    )

    result = run_experiment(
        spec
    )

    assert result.summary.variant == spec.variant

    assert [
        point.completed_episodes
        for point
        in result.summary.validation_curve
    ] == [
        2,
        4,
    ]

    assert all(
        point.evaluation.round_count == 3
        for point
        in result.summary.validation_curve
    )

    assert (
        result.summary.final_evaluation
        is None
    )

    # default DQNConfig.warmup_steps (1000) exceeds this test's tiny
    # training_episodes budget, so no optimizer updates happen here --
    # only the shape of any recorded diagnostics is checked.
    assert all(
        isinstance(episode, int)
        and isinstance(gradient_norm, float)
        and gradient_norm >= 0.0
        for episode, gradient_norm in result.diagnostics
    )


def test_policy_gradient_experiment_builds_validation_curve() -> None:
    spec = ExperimentRunSpec(
        variant=ExperimentVariant(
            algorithm=RLAlgorithm.REINFORCE,
            state_representation=(
                StateRepresentation.RAW
            ),
            reward_type=RewardType.NET_PROFIT,
        ),
        training_seed=123,
        training_episodes=4,
        validation_schedule=(
            build_validation_schedule(3)
        ),
        validation_checkpoints=(
            2,
            4,
        ),
    )

    result = run_experiment(
        spec
    )

    assert [
        point.completed_episodes
        for point
        in result.summary.validation_curve
    ] == [
        2,
        4,
    ]

    assert result.summary.optimizer_updates == 1

    assert result.diagnostics

    assert all(
        isinstance(episode, int)
        and isinstance(gradient_norm, float)
        and gradient_norm >= 0.0
        for episode, gradient_norm in result.diagnostics
    )


def test_experiment_can_run_final_evaluation() -> None:
    spec = ExperimentRunSpec(
        variant=ExperimentVariant(
            algorithm=RLAlgorithm.DQN,
            state_representation=(
                StateRepresentation.RAW
            ),
            reward_type=RewardType.NET_PROFIT,
        ),
        training_seed=123,
        training_episodes=2,
        validation_schedule=(
            build_validation_schedule(2)
        ),
        validation_checkpoints=(2,),
        final_evaluation_schedule=(
            build_final_evaluation_schedule(5)
        ),
    )

    result = run_experiment(
        spec
    )

    assert (
        result.summary.final_evaluation
        is not None
    )

    assert (
        result.summary.final_evaluation.round_count
        == 5
    )
import pytest


def test_run_spec_rejects_checkpoint_after_training() -> None:
    variant = ExperimentVariant(
        algorithm=RLAlgorithm.DQN,
        state_representation=StateRepresentation.RAW,
        reward_type=RewardType.NET_PROFIT,
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        ExperimentRunSpec(
            variant=variant,
            training_seed=123,
            training_episodes=10,
            validation_schedule=(
                build_validation_schedule(2)
            ),
            validation_checkpoints=(
                5,
                20,
            ),
        )