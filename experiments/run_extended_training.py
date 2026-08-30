import json

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from experiments.run_final_training import (
    FINAL_VALIDATION_INTERVAL,
    FINAL_VALIDATION_ROUNDS,
    FinalRunStatus,
    build_validation_checkpoints,
    get_run_status,
    save_checkpoint,
    write_learning_curve,
    write_summary,
)

from expert_poker_player.experiments import (
    FINAL_TRAINING_SEEDS,
    ExperimentRunSpec,
    ExperimentVariant,
    RLAlgorithm,
    build_validation_schedule,
    run_experiment,
)
from expert_poker_player.rewards import RewardType
from expert_poker_player.state_representation import (
    StateRepresentation,
)


EXTENDED_TRAINING_EPISODES = 100_000

DEFAULT_OUTPUT_DIR = Path(
    "experiments/runs/extended_training"
)


EXTENDED_VARIANTS = (
    ExperimentVariant(
        algorithm=RLAlgorithm.DQN,
        state_representation=(
            StateRepresentation.FEATURES
        ),
        reward_type=RewardType.NET_PROFIT,
    ),
    ExperimentVariant(
        algorithm=RLAlgorithm.REINFORCE,
        state_representation=(
            StateRepresentation.FEATURES
        ),
        reward_type=(
            RewardType.STAKE_SCALED_NET_PROFIT
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class ExtendedTrainingArgs:
    output_dir: Path


def build_run_dir(
    *,
    output_dir: Path,
    variant: ExperimentVariant,
    training_seed: int,
) -> Path:
    return (
        output_dir
        / variant.name
        / f"seed_{training_seed}"
        / f"episodes_{EXTENDED_TRAINING_EPISODES}"
    )


def build_manifest(
    args: ExtendedTrainingArgs,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []

    completed_runs = 0
    partial_runs = 0

    for variant in EXTENDED_VARIANTS:
        for training_seed in FINAL_TRAINING_SEEDS:
            run_dir = build_run_dir(
                output_dir=args.output_dir,
                variant=variant,
                training_seed=training_seed,
            )

            status = get_run_status(
                run_dir
            )

            if status is FinalRunStatus.COMPLETED:
                completed_runs += 1
            elif status is FinalRunStatus.PARTIAL:
                partial_runs += 1

            runs.append(
                {
                    "variant": {
                        "name": variant.name,
                        "algorithm": (
                            variant.algorithm.value
                        ),
                        "state_representation": (
                            variant
                            .state_representation
                            .value
                        ),
                        "reward_type": (
                            variant.reward_type.value
                        ),
                    },
                    "training_seed": training_seed,
                    "status": status.value,
                    "run_dir": str(
                        run_dir
                    ),
                    "summary_path": str(
                        run_dir / "summary.json"
                    ),
                    "learning_curve_path": str(
                        run_dir
                        / "learning_curve.csv"
                    ),
                    "checkpoint_path": str(
                        run_dir / "model.pt"
                    ),
                }
            )

    run_count = len(
        runs
    )

    return {
        "schema_version": 1,
        "purpose": (
            "extended_training_analysis"
        ),
        "selection_basis": (
            "50k multi-seed validation results"
        ),
        "training_episodes": (
            EXTENDED_TRAINING_EPISODES
        ),
        "validation_rounds": (
            FINAL_VALIDATION_ROUNDS
        ),
        "validation_interval": (
            FINAL_VALIDATION_INTERVAL
        ),
        "variant_count": len(
            EXTENDED_VARIANTS
        ),
        "training_seed_count": len(
            FINAL_TRAINING_SEEDS
        ),
        "run_count": run_count,
        "completed_runs": completed_runs,
        "partial_runs": partial_runs,
        "pending_runs": (
            run_count
            - completed_runs
            - partial_runs
        ),
        "runs": runs,
    }


def write_manifest(
    args: ExtendedTrainingArgs,
) -> None:
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        args.output_dir / "manifest.json"
    )

    temporary_path = (
        args.output_dir / "manifest.json.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            build_manifest(args),
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        manifest_path
    )


def run_extended_training(
    args: ExtendedTrainingArgs,
) -> None:
    write_manifest(
        args
    )

    for variant in EXTENDED_VARIANTS:
        for training_seed in FINAL_TRAINING_SEEDS:
            run_variant(
                variant=variant,
                training_seed=training_seed,
                args=args,
            )

            write_manifest(
                args
            )


def run_variant(
    *,
    variant: ExperimentVariant,
    training_seed: int,
    args: ExtendedTrainingArgs,
) -> None:
    run_dir = build_run_dir(
        output_dir=args.output_dir,
        variant=variant,
        training_seed=training_seed,
    )

    if (
        get_run_status(run_dir)
        is FinalRunStatus.COMPLETED
    ):
        print(
            f"Skipping {variant.name}, "
            f"seed {training_seed}, "
            "run already completed"
        )
        return

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_checkpoints = (
        build_validation_checkpoints(
            training_episodes=(
                EXTENDED_TRAINING_EPISODES
            ),
            interval=FINAL_VALIDATION_INTERVAL,
        )
    )

    spec = ExperimentRunSpec(
        variant=variant,
        training_seed=training_seed,
        training_episodes=(
            EXTENDED_TRAINING_EPISODES
        ),
        validation_schedule=(
            build_validation_schedule(
                FINAL_VALIDATION_ROUNDS
            )
        ),
        validation_checkpoints=(
            validation_checkpoints
        ),
    )

    print()
    print(
        f"Running extended {variant.name}, "
        f"seed {training_seed}"
    )

    result = run_experiment(
        spec
    )

    write_learning_curve(
        run_dir / "learning_curve.csv",
        result,
    )

    save_checkpoint(
        run_dir / "model.pt",
        variant,
        result,
    )

    write_summary(
        run_dir / "summary.json",
        result,
    )


def parse_args() -> ExtendedTrainingArgs:
    parser = ArgumentParser(
        description=(
            "Run the selected 2 x 5 "
            "extended 100k training matrix."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    namespace = parser.parse_args()

    return ExtendedTrainingArgs(
        output_dir=cast(
            Path,
            namespace.output_dir,
        )
    )


def main() -> None:
    run_extended_training(
        parse_args()
    )


if __name__ == "__main__":
    main()