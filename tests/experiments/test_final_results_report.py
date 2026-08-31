from pathlib import Path

import pytest

from experiments.generate_final_results_report import (
    TrainingRun,
    ValidationPoint,
    aggregate_learning_curves,
    aggregate_training_diagnostics,
    load_learning_curve,
    load_training_diagnostics,
)


def test_load_learning_curve(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path / "learning_curve.csv"
    )

    path.write_text(
        (
            "completed_episodes,"
            "estimated_ev\n"
            "4000,-0.4\n"
            "8000,-0.3\n"
        ),
        encoding="utf-8",
    )

    points = load_learning_curve(
        path
    )

    assert len(
        points
    ) == 2

    assert (
        points[0].completed_episodes
        == 4000
    )

    assert (
        points[0].estimated_ev
        == -0.4
    )

    assert (
        points[1].completed_episodes
        == 8000
    )


def test_aggregate_learning_curves() -> None:
    runs = tuple(
        TrainingRun(
            algorithm="dqn",
            state_representation="features",
            reward_type="net_profit",
            training_seed=seed,
            training_episodes=50_000,
            points=(
                ValidationPoint(
                    completed_episodes=4000,
                    estimated_ev=float(seed),
                ),
                ValidationPoint(
                    completed_episodes=8000,
                    estimated_ev=(
                        float(seed) + 1.0
                    ),
                ),
            ),
            training_diagnostics_path=(
                "unused_training_diagnostics.csv"
            ),
        )
        for seed in range(
            1,
            6,
        )
    )

    aggregate = (
        aggregate_learning_curves(
            runs
        )
    )

    assert len(
        aggregate
    ) == 2

    assert (
        aggregate[0].mean_ev
        == 3.0
    )

    assert (
        aggregate[1].mean_ev
        == 4.0
    )

    assert (
        aggregate[0]
        .standard_deviation
        > 0.0
    )


def test_aggregate_requires_five_runs() -> None:
    with pytest.raises(
        ValueError
    ):
        aggregate_learning_curves(
            ()
        )


def test_load_training_diagnostics(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path / "training_diagnostics.csv"
    )

    path.write_text(
        (
            "episode,"
            "gradient_norm\n"
            "1,0.5\n"
            "2,0.75\n"
        ),
        encoding="utf-8",
    )

    points = load_training_diagnostics(
        path
    )

    assert len(
        points
    ) == 2

    assert (
        points[0].episode
        == 1
    )

    assert (
        points[0].gradient_norm
        == 0.5
    )

    assert (
        points[1].episode
        == 2
    )


def _write_diagnostics_run(
    *,
    tmp_path: Path,
    seed: int,
    episode: int,
) -> TrainingRun:
    diagnostics_path = (
        tmp_path / f"diagnostics_{seed}.csv"
    )

    diagnostics_path.write_text(
        (
            "episode,"
            "gradient_norm\n"
            f"{episode},{float(seed)}\n"
            f"{episode + 1},{float(seed) + 1.0}\n"
        ),
        encoding="utf-8",
    )

    return TrainingRun(
        algorithm="dqn",
        state_representation="features",
        reward_type="net_profit",
        training_seed=seed,
        training_episodes=50_000,
        points=(
            ValidationPoint(
                completed_episodes=4000,
                estimated_ev=0.0,
            ),
        ),
        training_diagnostics_path=str(
            diagnostics_path
        ),
    )


def test_aggregate_training_diagnostics(
    tmp_path: Path,
) -> None:
    runs = tuple(
        _write_diagnostics_run(
            tmp_path=tmp_path,
            seed=seed,
            episode=1,
        )
        for seed in range(
            1,
            6,
        )
    )

    aggregate = (
        aggregate_training_diagnostics(
            runs
        )
    )

    assert len(
        aggregate
    ) == 2

    assert (
        aggregate[0].mean_gradient_norm
        == 3.0
    )

    assert (
        aggregate[1].mean_gradient_norm
        == 4.0
    )

    assert (
        aggregate[0]
        .standard_deviation
        > 0.0
    )


def test_aggregate_training_diagnostics_requires_five_runs() -> None:
    with pytest.raises(
        ValueError
    ):
        aggregate_training_diagnostics(
            ()
        )


def test_aggregate_training_diagnostics_uses_common_episodes(
    tmp_path: Path,
) -> None:
    # DQN's warmup boundary falls on a different episode per seed
    # (episode lengths are random), so runs may not share the exact
    # same episode set. Aggregation should fall back to whichever
    # episodes all five runs actually have in common.
    runs = tuple(
        _write_diagnostics_run(
            tmp_path=tmp_path,
            seed=seed,
            episode=(
                2
                if seed == 5
                else 1
            ),
        )
        for seed in range(
            1,
            6,
        )
    )

    aggregate = (
        aggregate_training_diagnostics(
            runs
        )
    )

    assert len(
        aggregate
    ) == 1

    assert (
        aggregate[0].episode
        == 2
    )

    assert aggregate[
        0
    ].mean_gradient_norm == pytest.approx(
        3.8
    )


def test_aggregate_training_diagnostics_rejects_no_common_episodes(
    tmp_path: Path,
) -> None:
    runs = tuple(
        _write_diagnostics_run(
            tmp_path=tmp_path,
            seed=seed,
            episode=(
                10
                if seed == 5
                else 1
            ),
        )
        for seed in range(
            1,
            6,
        )
    )

    with pytest.raises(
        ValueError,
        match="no common episodes",
    ):
        aggregate_training_diagnostics(
            runs
        )