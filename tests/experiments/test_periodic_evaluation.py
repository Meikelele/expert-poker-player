from expert_poker_player.dqn import QNetwork
from expert_poker_player.experiments import (
    DQNPeriodicEvaluator,
    PolicyGradientPeriodicEvaluator,
    build_validation_schedule,
)
from expert_poker_player.policy_gradient import (
    PolicyNetwork,
)
from expert_poker_player.state_representation import (
    RawStateEncoder,
)


def test_dqn_evaluator_ignores_non_checkpoint_episode() -> None:
    encoder = RawStateEncoder()

    evaluator = DQNPeriodicEvaluator(
        state_encoder=encoder,
        schedule=build_validation_schedule(5),
        checkpoints=(2, 4),
    )

    network = QNetwork(
        input_size=encoder.output_size,
        hidden_sizes=(16,),
    )

    evaluator(
        1,
        network,
    )

    assert evaluator.snapshots == ()


def test_dqn_evaluator_records_checkpoint() -> None:
    encoder = RawStateEncoder()

    evaluator = DQNPeriodicEvaluator(
        state_encoder=encoder,
        schedule=build_validation_schedule(5),
        checkpoints=(2, 4),
    )

    network = QNetwork(
        input_size=encoder.output_size,
        hidden_sizes=(16,),
    )

    evaluator(
        2,
        network,
    )

    assert len(
        evaluator.snapshots
    ) == 1

    snapshot = evaluator.snapshots[0]

    assert snapshot.completed_episodes == 2
    assert snapshot.metrics.round_count == 5


def test_dqn_evaluator_records_multiple_checkpoints() -> None:
    encoder = RawStateEncoder()

    evaluator = DQNPeriodicEvaluator(
        state_encoder=encoder,
        schedule=build_validation_schedule(3),
        checkpoints=(2, 4),
    )

    network = QNetwork(
        input_size=encoder.output_size,
        hidden_sizes=(16,),
    )

    for completed_episodes in range(
        1,
        5,
    ):
        evaluator(
            completed_episodes,
            network,
        )

    assert [
        snapshot.completed_episodes
        for snapshot in evaluator.snapshots
    ] == [
        2,
        4,
    ]

    assert all(
        snapshot.metrics.round_count == 3
        for snapshot in evaluator.snapshots
    )


def test_policy_gradient_evaluator_records_checkpoint() -> None:
    encoder = RawStateEncoder()

    evaluator = PolicyGradientPeriodicEvaluator(
        state_encoder=encoder,
        schedule=build_validation_schedule(5),
        checkpoints=(2,),
    )

    network = PolicyNetwork(
        input_size=encoder.output_size,
        hidden_sizes=(16,),
    )

    evaluator(
        2,
        network,
    )

    assert len(
        evaluator.snapshots
    ) == 1

    snapshot = evaluator.snapshots[0]

    assert snapshot.completed_episodes == 2
    assert snapshot.metrics.round_count == 5

import pytest


@pytest.mark.parametrize(
    "checkpoints",
    (
        (),
        (0,),
        (-1,),
    ),
)
def test_evaluator_rejects_invalid_checkpoints(
    checkpoints: tuple[int, ...],
) -> None:
    encoder = RawStateEncoder()

    with pytest.raises(
        ValueError
    ):
        DQNPeriodicEvaluator(
            state_encoder=encoder,
            schedule=build_validation_schedule(5),
            checkpoints=checkpoints,
        )


def test_evaluator_rejects_duplicate_checkpoints() -> None:
    encoder = RawStateEncoder()

    with pytest.raises(
        ValueError,
        match="checkpoints must be unique",
    ):
        DQNPeriodicEvaluator(
            state_encoder=encoder,
            schedule=build_validation_schedule(5),
            checkpoints=(2, 2),
        )

