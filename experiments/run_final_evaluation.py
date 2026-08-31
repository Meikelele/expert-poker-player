import csv
import json

from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

from experiments.run_extended_training import (
    EXTENDED_TRAINING_EPISODES,
    EXTENDED_VARIANTS,
    build_run_dir as build_extended_run_dir,
)
from experiments.run_final_training import (
    FINAL_TRAINING_EPISODES,
    build_run_dir as build_final_training_run_dir,
)

from expert_poker_player.agents import (
    Agent,
    RandomAgent,
    RuleBasedAgent,
)
from expert_poker_player.dqn import (
    DQNAgent,
    load_dqn_checkpoint,
)
from expert_poker_player.evaluation import (
    SimulationConfig,
    SimulationResult,
    calculate_metrics,
    run_simulation,
)
from expert_poker_player.experiments import (
    FINAL_EVALUATION_SCHEDULE_SEED,
    FINAL_TRAINING_SEEDS,
    FINAL_VARIANTS,
    EvaluationRecord,
    ExperimentVariant,
    RLAlgorithm,
    build_final_evaluation_schedule,
)
from expert_poker_player.policy_gradient import (
    PolicyGradientAgent,
    load_policy_gradient_checkpoint,
)
from expert_poker_player.rewards import RewardType
from expert_poker_player.state_representation import (
    StateRepresentation,
    build_state_encoder,
)


FINAL_EVALUATION_ROUNDS = 100_000
FINAL_RANDOM_AGENT_SEED = 20260901

DEFAULT_FINAL_TRAINING_DIR = Path(
    "experiments/runs/final_training"
)

DEFAULT_EXTENDED_TRAINING_DIR = Path(
    "experiments/runs/extended_training"
)

DEFAULT_OUTPUT_DIR = Path(
    "experiments/runs/final_evaluation"
)


class EvaluationAgentKind(str, Enum):
    RANDOM = "random"
    RULE_BASED = "rule_based"
    DQN = "dqn"
    REINFORCE = "reinforce"


class FinalEvaluationStatus(str, Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    COMPLETED = "completed"
    MISSING_CHECKPOINT = "missing_checkpoint"


@dataclass(frozen=True, slots=True)
class FinalEvaluationArgs:
    final_training_dir: Path
    extended_training_dir: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class FinalEvaluationTarget:
    agent_kind: EvaluationAgentKind
    variant: ExperimentVariant | None
    training_seed: int | None
    training_episodes: int | None
    checkpoint_path: Path | None

    @property
    def identifier(self) -> str:
        if self.variant is None:
            return (
                f"baseline_{self.agent_kind.value}"
            )

        if self.training_seed is None:
            raise ValueError(
                "model target requires training_seed"
            )

        if self.training_episodes is None:
            raise ValueError(
                "model target requires training_episodes"
            )

        return (
            f"{self.training_episodes}_"
            f"{self.variant.name}_"
            f"seed_{self.training_seed}"
        )

    @property
    def relative_output_dir(self) -> Path:
        if self.variant is None:
            return (
                Path("baselines")
                / self.agent_kind.value
            )

        if self.training_seed is None:
            raise ValueError(
                "model target requires training_seed"
            )

        if self.training_episodes is None:
            raise ValueError(
                "model target requires training_episodes"
            )

        return (
            Path(
                f"training_{self.training_episodes}"
            )
            / self.variant.name
            / f"seed_{self.training_seed}"
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "agent_kind": self.agent_kind.value,
            "training_seed": self.training_seed,
            "training_episodes": (
                self.training_episodes
            ),
            "checkpoint_path": (
                str(self.checkpoint_path)
                if self.checkpoint_path
                is not None
                else None
            ),
            "variant": (
                {
                    "name": self.variant.name,
                    "algorithm": (
                        self.variant.algorithm.value
                    ),
                    "state_representation": (
                        self.variant
                        .state_representation
                        .value
                    ),
                    "reward_type": (
                        self.variant.reward_type.value
                    ),
                }
                if self.variant is not None
                else None
            ),
        }


def build_targets(
    args: FinalEvaluationArgs,
) -> tuple[FinalEvaluationTarget, ...]:
    targets: list[
        FinalEvaluationTarget
    ] = [
        FinalEvaluationTarget(
            agent_kind=(
                EvaluationAgentKind.RANDOM
            ),
            variant=None,
            training_seed=None,
            training_episodes=None,
            checkpoint_path=None,
        ),
        FinalEvaluationTarget(
            agent_kind=(
                EvaluationAgentKind.RULE_BASED
            ),
            variant=None,
            training_seed=None,
            training_episodes=None,
            checkpoint_path=None,
        ),
    ]

    for variant in FINAL_VARIANTS:
        for training_seed in FINAL_TRAINING_SEEDS:
            run_dir = build_final_training_run_dir(
                output_dir=(
                    args.final_training_dir
                ),
                variant=variant,
                training_seed=training_seed,
            )

            targets.append(
                FinalEvaluationTarget(
                    agent_kind=(
                        _agent_kind_for_algorithm(
                            variant.algorithm
                        )
                    ),
                    variant=variant,
                    training_seed=training_seed,
                    training_episodes=(
                        FINAL_TRAINING_EPISODES
                    ),
                    checkpoint_path=(
                        run_dir / "model.pt"
                    ),
                )
            )

    for variant in EXTENDED_VARIANTS:
        for training_seed in FINAL_TRAINING_SEEDS:
            run_dir = build_extended_run_dir(
                output_dir=(
                    args.extended_training_dir
                ),
                variant=variant,
                training_seed=training_seed,
            )

            targets.append(
                FinalEvaluationTarget(
                    agent_kind=(
                        _agent_kind_for_algorithm(
                            variant.algorithm
                        )
                    ),
                    variant=variant,
                    training_seed=training_seed,
                    training_episodes=(
                        EXTENDED_TRAINING_EPISODES
                    ),
                    checkpoint_path=(
                        run_dir / "model.pt"
                    ),
                )
            )

    return tuple(
        targets
    )


def _agent_kind_for_algorithm(
    algorithm: RLAlgorithm,
) -> EvaluationAgentKind:
    if algorithm is RLAlgorithm.DQN:
        return EvaluationAgentKind.DQN

    if algorithm is RLAlgorithm.REINFORCE:
        return EvaluationAgentKind.REINFORCE

    raise ValueError(
        f"unsupported algorithm: {algorithm}"
    )


def get_target_status(
    *,
    args: FinalEvaluationArgs,
    target: FinalEvaluationTarget,
) -> FinalEvaluationStatus:
    output_dir = (
        args.output_dir
        / target.relative_output_dir
    )

    rounds_path = (
        output_dir / "evaluation_rounds.csv"
    )

    summary_path = (
        output_dir / "summary.json"
    )

    existing_count = sum(
        path.exists()
        for path in (
            rounds_path,
            summary_path,
        )
    )

    if existing_count == 2:
        return (
            FinalEvaluationStatus.COMPLETED
        )

    if existing_count == 1:
        return FinalEvaluationStatus.PARTIAL

    if (
        target.checkpoint_path is not None
        and not target.checkpoint_path.exists()
    ):
        return (
            FinalEvaluationStatus
            .MISSING_CHECKPOINT
        )

    return FinalEvaluationStatus.PENDING


def build_manifest(
    args: FinalEvaluationArgs,
) -> dict[str, object]:
    targets = build_targets(
        args
    )

    entries: list[
        dict[str, object]
    ] = []

    status_counts = {
        status: 0
        for status in FinalEvaluationStatus
    }

    for target in targets:
        status = get_target_status(
            args=args,
            target=target,
        )

        status_counts[
            status
        ] += 1

        entries.append(
            {
                **target.to_dict(),
                "status": status.value,
                "output_dir": str(
                    args.output_dir
                    / target.relative_output_dir
                ),
            }
        )

    return {
        "schema_version": 1,
        "purpose": "final_evaluation",
        "schedule_source_seed": (
            FINAL_EVALUATION_SCHEDULE_SEED
        ),
        "round_count": (
            FINAL_EVALUATION_ROUNDS
        ),
        "random_agent_seed": (
            FINAL_RANDOM_AGENT_SEED
        ),
        "target_count": len(
            targets
        ),
        "completed_targets": (
            status_counts[
                FinalEvaluationStatus.COMPLETED
            ]
        ),
        "partial_targets": (
            status_counts[
                FinalEvaluationStatus.PARTIAL
            ]
        ),
        "pending_targets": (
            status_counts[
                FinalEvaluationStatus.PENDING
            ]
        ),
        "missing_checkpoint_targets": (
            status_counts[
                FinalEvaluationStatus
                .MISSING_CHECKPOINT
            ]
        ),
        "targets": entries,
    }


def write_manifest(
    args: FinalEvaluationArgs,
) -> None:
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        args.output_dir / "manifest.json"
    )

    temporary_path = (
        args.output_dir
        / "manifest.json.tmp"
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


def build_agent(
    target: FinalEvaluationTarget,
) -> Agent:
    if (
        target.agent_kind
        is EvaluationAgentKind.RANDOM
    ):
        return RandomAgent(
            seed=FINAL_RANDOM_AGENT_SEED
        )

    if (
        target.agent_kind
        is EvaluationAgentKind.RULE_BASED
    ):
        return RuleBasedAgent()

    if target.checkpoint_path is None:
        raise ValueError(
            "model target requires checkpoint_path"
        )

    if (
        target.agent_kind
        is EvaluationAgentKind.DQN
    ):
        loaded = load_dqn_checkpoint(
            target.checkpoint_path
        )

        _validate_checkpoint_metadata(
            target=target,
            state_representation=(
                loaded.state_representation
            ),
            reward_type=loaded.reward_type,
            training_seed=loaded.training_seed,
            training_episodes=(
                loaded.config.training_episodes
            ),
        )

        loaded.policy_network.eval()

        return DQNAgent(
            q_network=loaded.policy_network,
            state_encoder=build_state_encoder(
                loaded.state_representation
            ),
            epsilon=0.0,
            seed=0,
        )

    if (
        target.agent_kind
        is EvaluationAgentKind.REINFORCE
    ):
        loaded = (
            load_policy_gradient_checkpoint(
                target.checkpoint_path
            )
        )

        _validate_checkpoint_metadata(
            target=target,
            state_representation=(
                loaded.state_representation
            ),
            reward_type=loaded.reward_type,
            training_seed=loaded.training_seed,
            training_episodes=(
                loaded.config.training_episodes
            ),
        )

        loaded.policy_network.eval()

        return PolicyGradientAgent(
            policy_network=(
                loaded.policy_network
            ),
            state_encoder=build_state_encoder(
                loaded.state_representation
            ),
            deterministic=True,
            seed=0,
        )

    raise ValueError(
        "unsupported evaluation agent kind"
    )


def _validate_checkpoint_metadata(
    *,
    target: FinalEvaluationTarget,
    state_representation: StateRepresentation,
    reward_type: RewardType,
    training_seed: int,
    training_episodes: int,
) -> None:
    if target.variant is None:
        raise ValueError(
            "model target requires variant"
        )

    if target.training_seed is None:
        raise ValueError(
            "model target requires training_seed"
        )

    if target.training_episodes is None:
        raise ValueError(
            "model target requires training_episodes"
        )

    if (
        state_representation
        is not target.variant.state_representation
    ):
        raise ValueError(
            "checkpoint state representation "
            "does not match evaluation target"
        )

    if (
        reward_type
        is not target.variant.reward_type
    ):
        raise ValueError(
            "checkpoint reward type "
            "does not match evaluation target"
        )

    if (
        training_seed
        != target.training_seed
    ):
        raise ValueError(
            "checkpoint training seed "
            "does not match evaluation target"
        )

    if (
        training_episodes
        != target.training_episodes
    ):
        raise ValueError(
            "checkpoint training episode count "
            "does not match evaluation target"
        )


def run_final_evaluation(
    args: FinalEvaluationArgs,
) -> None:
    schedule = (
        build_final_evaluation_schedule(
            FINAL_EVALUATION_ROUNDS
        )
    )

    write_manifest(
        args
    )

    for target in build_targets(
        args
    ):
        try:
            run_target(
                args=args,
                target=target,
                schedule=schedule,
            )
        finally:
            write_manifest(
                args
            )


def run_target(
    *,
    args: FinalEvaluationArgs,
    target: FinalEvaluationTarget,
    schedule: SimulationConfig,
) -> None:
    status = get_target_status(
        args=args,
        target=target,
    )

    if (
        status
        is FinalEvaluationStatus.COMPLETED
    ):
        print(
            f"Skipping {target.identifier}, "
            "evaluation already completed"
        )
        return

    if (
        status
        is FinalEvaluationStatus
        .MISSING_CHECKPOINT
    ):
        raise FileNotFoundError(
            f"Missing checkpoint for "
            f"{target.identifier}: "
            f"{target.checkpoint_path}"
        )

    output_dir = (
        args.output_dir
        / target.relative_output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Evaluating {target.identifier}"
    )

    agent = build_agent(
        target
    )

    simulation = run_simulation(
        agent=agent,
        config=schedule,
    )

    write_rounds(
        output_dir
        / "evaluation_rounds.csv",
        simulation,
    )

    _write_summary(
        output_dir / "summary.json",
        target=target,
        schedule=schedule,
        simulation=simulation,
    )


def write_rounds(
    path: Path,
    simulation: SimulationResult,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "round_index",
                "deck_seed",
                "net_profit",
                "total_staked",
                "outcome",
                "actions",
            ),
        )

        writer.writeheader()

        for (
            round_index,
            (
                deck_seed,
                episode,
            ),
        ) in enumerate(
            zip(
                simulation.config.deck_seeds,
                simulation.episodes,
            )
        ):
            writer.writerow(
                {
                    "round_index": round_index,
                    "deck_seed": deck_seed,
                    "net_profit": str(
                        episode.net_profit
                    ),
                    "total_staked": str(
                        episode.total_staked
                    ),
                    "outcome": (
                        episode.outcome.name
                    ),
                    "actions": "|".join(
                        action.name
                        for action
                        in episode.actions
                    ),
                }
            )


def _write_summary(
    path: Path,
    *,
    target: FinalEvaluationTarget,
    schedule: SimulationConfig,
    simulation: SimulationResult,
) -> None:
    metrics = calculate_metrics(
        simulation
    )

    evaluation = (
        EvaluationRecord.from_metrics(
            metrics
        )
    )

    summary = { # type: ignore
        "schema_version": 1,
        "target": target.to_dict(),
        "schedule": {
            "source_seed": (
                FINAL_EVALUATION_SCHEDULE_SEED
            ),
            "round_count": (
                schedule.round_count
            ),
        },
        "evaluation": (
            evaluation.to_dict()
        ),
    }

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> FinalEvaluationArgs:
    parser = ArgumentParser(
        description=(
            "Evaluate all final RL models and "
            "baselines on the common 100k "
            "final deck schedule."
        )
    )

    parser.add_argument(
        "--final-training-dir",
        type=Path,
        default=DEFAULT_FINAL_TRAINING_DIR,
    )

    parser.add_argument(
        "--extended-training-dir",
        type=Path,
        default=DEFAULT_EXTENDED_TRAINING_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    namespace = parser.parse_args()

    return FinalEvaluationArgs(
        final_training_dir=cast(
            Path,
            namespace.final_training_dir,
        ),
        extended_training_dir=cast(
            Path,
            namespace.extended_training_dir,
        ),
        output_dir=cast(
            Path,
            namespace.output_dir,
        ),
    )


def main() -> None:
    run_final_evaluation(
        parse_args()
    )


if __name__ == "__main__":
    main()