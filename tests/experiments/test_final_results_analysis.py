from pathlib import Path

from experiments.analyze_final_results import (
    EvaluationTargetData,
    get_evaluation_ev,
    paired_seed_comparison,
    summarize_values,
)


def build_model_target(
    *,
    seed: int,
    ev: float,
) -> EvaluationTargetData:
    return EvaluationTargetData(
        identifier=f"model_{seed}",
        summary_path=Path(
            "summary.json"
        ),
        rounds_path=Path(
            "rounds.csv"
        ),
        summary={
            "target": {
                "training_seed": seed,
                "training_episodes": 50_000,
                "variant": {
                    "algorithm": "dqn",
                    "state_representation": (
                        "features"
                    ),
                    "reward_type": (
                        "net_profit"
                    ),
                },
            },
            "evaluation": {
                "estimated_ev": ev,
            },
        },
    )


def test_summarize_values_calculates_interval() -> None:
    result = summarize_values(
        (
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ),
        critical_value=2.0,
    )

    assert result.sample_count == 5
    assert result.mean == 3.0
    assert (
        result.standard_deviation
        > 0.0
    )
    assert (
        result.ci95_low
        < result.mean
    )
    assert (
        result.ci95_high
        > result.mean
    )


def test_get_evaluation_ev_reads_summary() -> None:
    target = build_model_target(
        seed=1,
        ev=-0.25,
    )

    assert (
        get_evaluation_ev(
            target
        )
        == -0.25
    )


def test_paired_seed_comparison_aligns_by_seed() -> None:
    left = tuple(
        build_model_target(
            seed=seed,
            ev=float(seed),
        )
        for seed in range(
            1,
            6,
        )
    )

    right = tuple(
        build_model_target(
            seed=seed,
            ev=float(seed) - 0.5,
        )
        for seed in reversed(
            range(
                1,
                6,
            )
        )
    )

    comparison = (
        paired_seed_comparison(
            left,
            right,
        )
    )

    assert (
        comparison.sample_count
        == 5
    )

    assert (
        comparison.mean
        == 0.5
    )

    assert (
        comparison.standard_deviation
        == 0.0
    )

    assert (
        comparison.ci95_low
        == 0.5
    )

    assert (
        comparison.ci95_high
        == 0.5
    )