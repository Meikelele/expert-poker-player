from expert_poker_player.agents import RandomAgent
from expert_poker_player.evaluation import (
    calculate_metrics,
    run_simulation,
)
from expert_poker_player.experiments import (
    EvaluationRecord,
    ExperimentRunSummary,
    ExperimentVariant,
    LearningCurvePoint,
    RLAlgorithm,
    build_validation_schedule,
)
from expert_poker_player.rewards import RewardType
from expert_poker_player.state_representation import (
    StateRepresentation,
)


def build_evaluation_record() -> EvaluationRecord:
    result = run_simulation(
        agent=RandomAgent(seed=123),
        config=build_validation_schedule(5),
    )

    return EvaluationRecord.from_metrics(
        calculate_metrics(result)
    )


def test_evaluation_record_preserves_metrics() -> None:
    record = build_evaluation_record()

    assert record.round_count == 5

    assert sum(
        record.outcome_counts.values()
    ) == 5

    assert record.standard_error >= 0.0


def test_learning_curve_point_serializes() -> None:
    point = LearningCurvePoint(
        completed_episodes=100,
        evaluation=build_evaluation_record(),
    )

    payload = point.to_dict()

    assert payload[
        "completed_episodes"
    ] == 100

    assert isinstance(
        payload["evaluation"],
        dict,
    )


def test_experiment_run_summary_serializes_identity() -> None:
    variant = ExperimentVariant(
        algorithm=RLAlgorithm.DQN,
        state_representation=StateRepresentation.RAW,
        reward_type=RewardType.NET_PROFIT,
    )

    summary = ExperimentRunSummary(
        variant=variant,
        training_seed=123,
        training_episodes=100,
        total_steps=200,
        optimizer_updates=50,
        training_config={
            "learning_rate": 0.001,
        },
        validation_curve=(),
    )

    payload = summary.to_dict()

    assert payload["schema_version"] == 1

    assert payload["variant"] == {
        "name": "dqn_raw_net_profit",
        "algorithm": "dqn",
        "state_representation": "raw",
        "reward_type": "net_profit",
    }


def test_experiment_run_rejects_unsorted_validation_curve() -> None:
    evaluation = build_evaluation_record()

    first = LearningCurvePoint(
        completed_episodes=20,
        evaluation=evaluation,
    )

    second = LearningCurvePoint(
        completed_episodes=10,
        evaluation=evaluation,
    )

    variant = ExperimentVariant(
        algorithm=RLAlgorithm.REINFORCE,
        state_representation=StateRepresentation.FEATURES,
        reward_type=RewardType.NET_PROFIT,
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        ExperimentRunSummary(
            variant=variant,
            training_seed=123,
            training_episodes=100,
            total_steps=200,
            optimizer_updates=10,
            training_config={},
            validation_curve=(
                first,
                second,
            ),
        )