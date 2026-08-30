import csv
import json

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from expert_poker_player.dqn import (
    DQNConfig,
    QNetwork,
    save_dqn_checkpoint,
)
from expert_poker_player.experiments import (
    FINAL_TRAINING_SEEDS,
    FINAL_VARIANTS,
    ExperimentExecutionResult,
    ExperimentRunSpec,
    ExperimentVariant,
    RLAlgorithm,
    build_validation_schedule,
    run_experiment,
)
from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
    PolicyNetwork,
    save_policy_gradient_checkpoint,
)


FINAL_TRAINING_EPISODES = 50_000
FINAL_VALIDATION_ROUNDS = 2_000
FINAL_VALIDATION_INTERVAL = 4_000

DEFAULT_OUTPUT_DIR = Path(
    "experiments/runs/final_training"
)


@dataclass(frozen=True, slots=True)
class FinalTrainingArgs:
    output_dir: Path


def build_validation_checkpoints(
    *,
    training_episodes: int,
    interval: int,
) -> tuple[int, ...]:
    if type(training_episodes) is not int:
        raise TypeError(
            "training_episodes must be an integer"
        )

    if type(interval) is not int:
        raise TypeError(
            "interval must be an integer"
        )

    if training_episodes <= 0:
        raise ValueError(
            "training_episodes must be positive"
        )

    if interval <= 0:
        raise ValueError(
            "interval must be positive"
        )

    checkpoints = list(
        range(
            interval,
            training_episodes + 1,
            interval,
        )
    )

    if (
        not checkpoints
        or checkpoints[-1] != training_episodes
    ):
        checkpoints.append(
            training_episodes
        )

    return tuple(checkpoints)


def run_final_training(
    args: FinalTrainingArgs,
) -> None:
    for variant in FINAL_VARIANTS:
        for training_seed in FINAL_TRAINING_SEEDS:
            run_variant(
                variant=variant,
                training_seed=training_seed,
                args=args,
            )


def run_variant(
    *,
    variant: ExperimentVariant,
    training_seed: int,
    args: FinalTrainingArgs,
) -> None:
    run_dir = (
        args.output_dir
        / variant.name
        / f"seed_{training_seed}"
        / f"episodes_{FINAL_TRAINING_EPISODES}"
    )

    summary_path = (
        run_dir / "summary.json"
    )

    learning_curve_path = (
        run_dir / "learning_curve.csv"
    )

    checkpoint_path = (
        run_dir / "model.pt"
    )

    if (
        summary_path.exists()
        and learning_curve_path.exists()
        and checkpoint_path.exists()
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
                FINAL_TRAINING_EPISODES
            ),
            interval=FINAL_VALIDATION_INTERVAL,
        )
    )

    spec = ExperimentRunSpec(
        variant=variant,
        training_seed=training_seed,
        training_episodes=(
            FINAL_TRAINING_EPISODES
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
        f"Running {variant.name}, "
        f"seed {training_seed}"
    )

    result = run_experiment(
        spec
    )

    _write_learning_curve(
        learning_curve_path,
        result,
    )

    _save_checkpoint(
        checkpoint_path,
        variant,
        result,
    )

    _write_summary(
        summary_path,
        result,
    )


def _write_summary(
    path: Path,
    result: ExperimentExecutionResult,
) -> None:
    path.write_text(
        json.dumps(
            result.summary.to_dict(),
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_learning_curve(
    path: Path,
    result: ExperimentExecutionResult,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "completed_episodes",
                "estimated_ev",
                "standard_error",
                "mean_staked",
            ),
        )

        writer.writeheader()

        for point in result.summary.validation_curve:
            writer.writerow(
                {
                    "completed_episodes": (
                        point.completed_episodes
                    ),
                    "estimated_ev": (
                        point.evaluation.estimated_ev
                    ),
                    "standard_error": (
                        point.evaluation.standard_error
                    ),
                    "mean_staked": (
                        point.evaluation.mean_staked
                    ),
                }
            )


def _save_checkpoint(
    path: Path,
    variant: ExperimentVariant,
    result: ExperimentExecutionResult,
) -> None:
    if variant.algorithm is RLAlgorithm.DQN:
        if not isinstance(
            result.policy_network,
            QNetwork,
        ):
            raise TypeError(
                "expected QNetwork"
            )

        if not isinstance(
            result.training_config,
            DQNConfig,
        ):
            raise TypeError(
                "expected DQNConfig"
            )

        save_dqn_checkpoint(
            path,
            policy_network=result.policy_network,
            state_representation=(
                variant.state_representation
            ),
            reward_type=variant.reward_type,
            config=result.training_config,
        )

        return

    if not isinstance(
        result.policy_network,
        PolicyNetwork,
    ):
        raise TypeError(
            "expected PolicyNetwork"
        )

    if not isinstance(
        result.training_config,
        PolicyGradientConfig,
    ):
        raise TypeError(
            "expected PolicyGradientConfig"
        )

    save_policy_gradient_checkpoint(
        path,
        policy_network=result.policy_network,
        state_representation=(
            variant.state_representation
        ),
        reward_type=variant.reward_type,
        config=result.training_config,
    )


def parse_args() -> FinalTrainingArgs:
    parser = ArgumentParser(
        description=(
            "Run the final 8 x 5 RL training matrix."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    namespace = parser.parse_args()

    return FinalTrainingArgs(
        output_dir=cast(
            Path,
            namespace.output_dir,
        )
    )


def main() -> None:
    run_final_training(
        parse_args()
    )


if __name__ == "__main__":
    main()