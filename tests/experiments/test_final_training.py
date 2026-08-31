from pathlib import Path
from typing import cast

import pytest

import experiments.run_final_training as final_training

from expert_poker_player.experiments import (
    FINAL_TRAINING_SEEDS,
    FINAL_VARIANTS,
    ExperimentRunSpec,
)

import experiments.run_extended_training as extended_training


def test_manifest_defines_exact_final_matrix(
    tmp_path: Path,
) -> None:
    args = final_training.FinalTrainingArgs(
        output_dir=tmp_path,
    )

    manifest = final_training.build_manifest(
        args
    )

    assert manifest["variant_count"] == 8
    assert manifest["training_seed_count"] == 5
    assert manifest["run_count"] == 40
    assert manifest["completed_runs"] == 0
    assert manifest["partial_runs"] == 0
    assert manifest["pending_runs"] == 40

    runs = cast(
        list[dict[str, object]],
        manifest["runs"],
    )

    assert len(runs) == 40

    assert {
        run["status"]
        for run in runs
    } == {
        final_training.FinalRunStatus.PENDING.value
    }

    assert len(
        {
            run["run_dir"]
            for run in runs
        }
    ) == 40


def test_run_status_tracks_artifact_completeness(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"

    assert (
        final_training.get_run_status(
            run_dir
        )
        is final_training.FinalRunStatus.PENDING
    )

    run_dir.mkdir()

    (
        run_dir / "summary.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    assert (
        final_training.get_run_status(
            run_dir
        )
        is final_training.FinalRunStatus.PARTIAL
    )

    (
        run_dir / "learning_curve.csv"
    ).write_text(
        "",
        encoding="utf-8",
    )

    (
        run_dir / "model.pt"
    ).write_bytes(
        b"checkpoint"
    )

    assert (
        final_training.get_run_status(
            run_dir
        )
        is final_training.FinalRunStatus.COMPLETED
    )


def test_completed_run_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = FINAL_VARIANTS[0]
    training_seed = FINAL_TRAINING_SEEDS[0]

    args = final_training.FinalTrainingArgs(
        output_dir=tmp_path,
    )

    run_dir = final_training.build_run_dir(
        output_dir=tmp_path,
        variant=variant,
        training_seed=training_seed,
    )

    run_dir.mkdir(
        parents=True,
    )

    (
        run_dir / "summary.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    (
        run_dir / "learning_curve.csv"
    ).write_text(
        "",
        encoding="utf-8",
    )

    (
        run_dir / "model.pt"
    ).write_bytes(
        b"checkpoint"
    )

    def fail_run_experiment(
        spec: ExperimentRunSpec,
    ) -> object:
        raise AssertionError(
            "completed run must be skipped"
        )

    monkeypatch.setattr(
        final_training,
        "run_experiment",
        fail_run_experiment,
    )

    final_training.run_variant(
        variant=variant,
        training_seed=training_seed,
        args=args,
    )


def test_partial_run_is_executed_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = FINAL_VARIANTS[0]
    training_seed = FINAL_TRAINING_SEEDS[0]

    args = final_training.FinalTrainingArgs(
        output_dir=tmp_path,
    )

    run_dir = final_training.build_run_dir(
        output_dir=tmp_path,
        variant=variant,
        training_seed=training_seed,
    )

    run_dir.mkdir(
        parents=True,
    )

    (
        run_dir / "summary.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    observed_specs: list[
        ExperimentRunSpec
    ] = []

    def fake_run_experiment(
        spec: ExperimentRunSpec,
    ) -> object:
        observed_specs.append(
            spec
        )
        return object()

    monkeypatch.setattr(
        final_training,
        "run_experiment",
        fake_run_experiment,
    )

    monkeypatch.setattr(
        final_training,
        "write_learning_curve",
        lambda path, result: None, # type: ignore
    )

    monkeypatch.setattr(
        final_training,
        "save_checkpoint",
        lambda path, variant, result: None, # type: ignore
    )

    monkeypatch.setattr(
        final_training,
        "write_summary",
        lambda path, result: None, # type: ignore
    )

    final_training.run_variant(
        variant=variant,
        training_seed=training_seed,
        args=args,
    )

    assert len(
        observed_specs
    ) == 1

    spec = observed_specs[0]

    assert (
        spec.training_episodes
        == final_training.FINAL_TRAINING_EPISODES
    )

    assert (
        spec.validation_checkpoints[-1]
        == final_training.FINAL_TRAINING_EPISODES
    )

    assert (
        spec.final_evaluation_schedule
        is None
    )


def test_final_orchestration_runs_exact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = final_training.FinalTrainingArgs(
        output_dir=tmp_path,
    )

    observed_runs: list[
        tuple[str, int]
    ] = []

    manifest_write_count = 0

    def fake_run_variant(
        *,
        variant: object,
        training_seed: int,
        args: object,
    ) -> None:
        observed_runs.append(
            (
                getattr(
                    variant,
                    "name",
                ),
                training_seed,
            )
        )

    def fake_write_manifest(
        args: object,
    ) -> None:
        nonlocal manifest_write_count

        manifest_write_count += 1

    monkeypatch.setattr(
        final_training,
        "run_variant",
        fake_run_variant,
    )

    monkeypatch.setattr(
        final_training,
        "write_manifest",
        fake_write_manifest,
    )

    final_training.run_final_training(
        args
    )

    expected_runs = [
        (
            variant.name,
            training_seed,
        )
        for variant in FINAL_VARIANTS
        for training_seed in FINAL_TRAINING_SEEDS
    ]

    assert observed_runs == expected_runs

    assert len(
        observed_runs
    ) == 40

    assert manifest_write_count == 41

def test_extended_training_uses_selected_variants() -> None:
    assert len(
        extended_training.EXTENDED_VARIANTS
    ) == 2

    dqn_variant = (
        extended_training.EXTENDED_VARIANTS[0]
    )

    reinforce_variant = (
        extended_training.EXTENDED_VARIANTS[1]
    )

    assert (
        dqn_variant.algorithm
        is extended_training.RLAlgorithm.DQN
    )

    assert (
        dqn_variant.state_representation
        is extended_training.StateRepresentation.FEATURES
    )

    assert (
        dqn_variant.reward_type
        is extended_training.RewardType.NET_PROFIT
    )

    assert (
        reinforce_variant.algorithm
        is extended_training.RLAlgorithm.REINFORCE
    )

    assert (
        reinforce_variant.state_representation
        is extended_training.StateRepresentation.FEATURES
    )

    assert (
        reinforce_variant.reward_type
        is extended_training.RewardType.STAKE_SCALED_NET_PROFIT
    )


def test_extended_manifest_defines_ten_runs(
    tmp_path: Path,
) -> None:
    args = (
        extended_training.ExtendedTrainingArgs(
            output_dir=tmp_path,
        )
    )

    manifest = extended_training.build_manifest(
        args
    )

    assert manifest["training_episodes"] == 100_000
    assert manifest["variant_count"] == 2
    assert manifest["training_seed_count"] == 5
    assert manifest["run_count"] == 10
    assert manifest["completed_runs"] == 0
    assert manifest["partial_runs"] == 0
    assert manifest["pending_runs"] == 10