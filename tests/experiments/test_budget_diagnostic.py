import pytest

from experiments.run_training_budget_diagnostic import (
    build_validation_checkpoints,
)


def test_budget_checkpoints_include_final_episode() -> None:
    checkpoints = build_validation_checkpoints(
        training_episodes=10_000,
        interval=4_000,
    )

    assert checkpoints == (
        4_000,
        8_000,
        10_000,
    )


def test_budget_checkpoints_do_not_duplicate_final_episode() -> None:
    checkpoints = build_validation_checkpoints(
        training_episodes=8_000,
        interval=4_000,
    )

    assert checkpoints == (
        4_000,
        8_000,
    )


@pytest.mark.parametrize(
    "training_episodes, interval",
    (
        (0, 4_000),
        (10_000, 0),
        (-1, 4_000),
        (10_000, -1),
    ),
)
def test_budget_checkpoints_require_positive_values(
    training_episodes: int,
    interval: int,
) -> None:
    with pytest.raises(
        ValueError
    ):
        build_validation_checkpoints(
            training_episodes=training_episodes,
            interval=interval,
        )
    