import pytest

from expert_poker_player.experiments import (
    build_deck_schedule,
    build_final_evaluation_schedule,
    build_validation_schedule,
)


def test_same_schedule_parameters_are_reproducible() -> None:
    first = build_deck_schedule(
        source_seed=123,
        round_count=10,
    )
    second = build_deck_schedule(
        source_seed=123,
        round_count=10,
    )

    assert first == second


def test_schedule_is_prefix_stable() -> None:
    short = build_validation_schedule(10)
    long = build_validation_schedule(20)

    assert (
        short.deck_seeds
        == long.deck_seeds[:10]
    )


def test_validation_and_final_schedules_are_distinct() -> None:
    validation = build_validation_schedule(100)
    final = build_final_evaluation_schedule(100)

    assert (
        validation.deck_seeds
        != final.deck_seeds
    )


@pytest.mark.parametrize(
    "round_count",
    (
        0,
        -1,
    ),
)
def test_schedule_requires_positive_round_count(
    round_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="round_count must be positive",
    ):
        build_validation_schedule(
            round_count
        )