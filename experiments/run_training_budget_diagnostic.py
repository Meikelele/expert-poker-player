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


DEFAULT_OUTPUT_DIR = Path(
    "experiments/runs/training_budget"
)

DEFAULT_TRAINING_EPISODES = 50_000
DEFAULT_VALIDATION_ROUNDS = 2_000
DEFAULT_VALIDATION_INTERVAL = 4_000

DEFAULT_TRAINING_SEED = 20260831


@dataclass(frozen=True, slots=True)
class BudgetDiagnosticArgs:
    training_episodes: int
    validation_rounds: int
    validation_interval: int
    training_seed: int
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

    return tuple(
        checkpoints
    )

def run_variant(
    *,
    variant: ExperimentVariant,
    args: BudgetDiagnosticArgs,
) -> None:
    run_dir = (
        args.output_dir
        / variant.name
        / f"seed_{args.training_seed}"
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
            "result already exists"
        )
        return

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints = build_validation_checkpoints(
        training_episodes=args.training_episodes,
        interval=args.validation_interval,
    )

    spec = ExperimentRunSpec(
        variant=variant,
        training_seed=args.training_seed,
        training_episodes=args.training_episodes,
        validation_schedule=(
            build_validation_schedule(
                args.validation_rounds
            )
        ),
        validation_checkpoints=checkpoints,
    )

    print()
    print(
        f"Running {variant.name}"
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

    # summary.json ostatni: pełni rolę markera ukończenia runu.
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

def parse_args() -> BudgetDiagnosticArgs:
    parser = ArgumentParser(
        description=(
            "Run the training-budget diagnostic "
            "for all RL variants."
        )
    )

    parser.add_argument(
        "--training-episodes",
        type=int,
        default=DEFAULT_TRAINING_EPISODES,
    )

    parser.add_argument(
        "--validation-rounds",
        type=int,
        default=DEFAULT_VALIDATION_ROUNDS,
    )

    parser.add_argument(
        "--validation-interval",
        type=int,
        default=DEFAULT_VALIDATION_INTERVAL,
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    namespace = parser.parse_args()

    return BudgetDiagnosticArgs(
        training_episodes=cast(
            int,
            namespace.training_episodes,
        ),
        validation_rounds=cast(
            int,
            namespace.validation_rounds,
        ),
        validation_interval=cast(
            int,
            namespace.validation_interval,
        ),
        training_seed=cast(
            int,
            namespace.training_seed,
        ),
        output_dir=cast(
            Path,
            namespace.output_dir,
        ),
    )

def main() -> None:
    args = parse_args()

    pg_batch_size = (
        PolicyGradientConfig().batch_size
    )

    if (
        args.validation_interval
        % pg_batch_size
        != 0
    ):
        raise ValueError(
            "validation_interval must be "
            "divisible by Policy Gradient batch size"
        )

    for variant in FINAL_VARIANTS:
        run_variant(
            variant=variant,
            args=args,
        )


if __name__ == "__main__":
    main()