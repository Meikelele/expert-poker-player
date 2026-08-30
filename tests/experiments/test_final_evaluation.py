import csv

from fractions import Fraction
from pathlib import Path

import experiments.run_final_evaluation as final_evaluation

from expert_poker_player.agents import (
    RuleBasedAgent,
)
from expert_poker_player.evaluation import (
    SimulationConfig,
    run_simulation,
)


def build_args(
    tmp_path: Path,
) -> final_evaluation.FinalEvaluationArgs:
    return final_evaluation.FinalEvaluationArgs(
        final_training_dir=(
            tmp_path / "final_training"
        ),
        extended_training_dir=(
            tmp_path / "extended_training"
        ),
        output_dir=(
            tmp_path / "final_evaluation"
        ),
    )


def test_final_evaluation_defines_all_targets(
    tmp_path: Path,
) -> None:
    args = build_args(
        tmp_path
    )

    targets = (
        final_evaluation.build_targets(
            args
        )
    )

    assert len(
        targets
    ) == 52

    baseline_targets = [
        target
        for target in targets
        if target.training_episodes is None
    ]

    final_training_targets = [
        target
        for target in targets
        if (
            target.training_episodes
            == 50_000
        )
    ]

    extended_training_targets = [
        target
        for target in targets
        if (
            target.training_episodes
            == 100_000
        )
    ]

    assert len(
        baseline_targets
    ) == 2

    assert len(
        final_training_targets
    ) == 40

    assert len(
        extended_training_targets
    ) == 10

    assert len(
        {
            target.identifier
            for target in targets
        }
    ) == 52


def test_missing_model_checkpoint_is_reported(
    tmp_path: Path,
) -> None:
    args = build_args(
        tmp_path
    )

    target = next(
        target
        for target
        in final_evaluation.build_targets(
            args
        )
        if target.checkpoint_path
        is not None
    )

    assert (
        final_evaluation.get_target_status(
            args=args,
            target=target,
        )
        is final_evaluation
        .FinalEvaluationStatus
        .MISSING_CHECKPOINT
    )


def test_completed_output_is_detected(
    tmp_path: Path,
) -> None:
    args = build_args(
        tmp_path
    )

    target = (
        final_evaluation
        .FinalEvaluationTarget(
            agent_kind=(
                final_evaluation
                .EvaluationAgentKind
                .RULE_BASED
            ),
            variant=None,
            training_seed=None,
            training_episodes=None,
            checkpoint_path=None,
        )
    )

    output_dir = (
        args.output_dir
        / target.relative_output_dir
    )

    output_dir.mkdir(
        parents=True,
    )

    (
        output_dir
        / "evaluation_rounds.csv"
    ).write_text(
        "",
        encoding="utf-8",
    )

    (
        output_dir
        / "summary.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    assert (
        final_evaluation.get_target_status(
            args=args,
            target=target,
        )
        is final_evaluation
        .FinalEvaluationStatus
        .COMPLETED
    )


def test_round_csv_preserves_pairing_data(
    tmp_path: Path,
) -> None:
    config = SimulationConfig(
        deck_seeds=(
            11,
            22,
            33,
        )
    )

    simulation = run_simulation(
        agent=RuleBasedAgent(),
        config=config,
    )

    output_path = (
        tmp_path / "rounds.csv"
    )

    final_evaluation.write_rounds(
        output_path,
        simulation,
    )

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(
                file
            )
        )

    assert len(
        rows
    ) == 3

    assert [
        int(row["deck_seed"])
        for row in rows
    ] == [
        11,
        22,
        33,
    ]

    for row, episode in zip(
        rows,
        simulation.episodes,
    ):
        assert (
            Fraction(
                row["net_profit"]
            )
            == episode.net_profit
        )

        assert (
            Fraction(
                row["total_staked"]
            )
            == episode.total_staked
        )

        assert (
            row["outcome"]
            == episode.outcome.name
        )


def test_manifest_describes_final_protocol(
    tmp_path: Path,
) -> None:
    args = build_args(
        tmp_path
    )

    manifest = (
        final_evaluation.build_manifest(
            args
        )
    )

    assert (
        manifest["round_count"]
        == 100_000
    )

    assert (
        manifest[
            "schedule_source_seed"
        ]
        == 20260830
    )

    assert (
        manifest["target_count"]
        == 52
    )

    assert (
        manifest[
            "missing_checkpoint_targets"
        ]
        == 50
    )