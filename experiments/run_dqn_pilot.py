import csv
import json

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import cast

from expert_poker_player.dqn import (
    DQNConfig,
    train_dqn,
)
from expert_poker_player.evaluation import (
    AgentMetrics,
    SimulationConfig,
    calculate_metrics,
    run_simulation,
)
from expert_poker_player.rewards import (
    RewardType,
    build_reward_function,
)
from expert_poker_player.state_representation import (
    StateRepresentation,
    build_state_encoder,
)
from expert_poker_player.uth import (
    Action,
    RoundOutcome,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/results"
)

DEFAULT_TRAINING_EPISODES = 10_000
DEFAULT_EVALUATION_ROUNDS = 10_000

DEFAULT_TRAINING_SEED = 20260823
DEFAULT_DECK_SCHEDULE_SEED = 20260815


@dataclass(
    frozen=True,
    slots=True,
)
class PilotArgs:
    training_episodes: int
    evaluation_rounds: int
    training_seed: int
    deck_seed: int
    output_dir: Path


@dataclass(
    frozen=True,
    slots=True,
)
class PilotVariant:
    state_representation: StateRepresentation
    reward_type: RewardType

    @property
    def name(self) -> str:
        return (
            f"{self.state_representation.value}_"
            f"{self.reward_type.value}"
        )


VARIANTS = (
    PilotVariant(
        state_representation=StateRepresentation.RAW,
        reward_type=RewardType.NET_PROFIT,
    ),
    PilotVariant(
        state_representation=StateRepresentation.RAW,
        reward_type=(
            RewardType.STAKE_SCALED_NET_PROFIT
        ),
    ),
    PilotVariant(
        state_representation=StateRepresentation.FEATURES,
        reward_type=RewardType.NET_PROFIT,
    ),
    PilotVariant(
        state_representation=StateRepresentation.FEATURES,
        reward_type=(
            RewardType.STAKE_SCALED_NET_PROFIT
        ),
    ),
)


def build_evaluation_config(
    *,
    round_count: int,
    schedule_seed: int,
) -> SimulationConfig:
    deck_random = Random(
        schedule_seed
    )

    deck_seeds = tuple(
        deck_random.getrandbits(63)
        for _ in range(round_count)
    )

    return SimulationConfig(
        deck_seeds=deck_seeds
    )


def build_training_config(
    *,
    training_episodes: int,
    seed: int,
) -> DQNConfig:
    return DQNConfig(
        training_episodes=training_episodes,
        seed=seed,
    )


def metrics_to_dict(
    metrics: AgentMetrics,
) -> dict[str, object]:
    return {
        "round_count": metrics.round_count,
        "total_net_profit": float(
            metrics.total_net_profit
        ),
        "estimated_ev": float(
            metrics.estimated_ev
        ),
        "total_staked": float(
            metrics.total_staked
        ),
        "mean_staked": float(
            metrics.mean_staked
        ),
        "standard_deviation": (
            metrics.standard_deviation
        ),
        "standard_error": (
            metrics.standard_error
        ),
        "outcome_counts": {
            outcome.name: metrics.outcome_counts[
                outcome
            ]
            for outcome in RoundOutcome
        },
        "action_counts": {
            action.name: metrics.action_counts[
                action
            ]
            for action in Action
        },
    }


def print_metrics(
    *,
    variant: PilotVariant,
    metrics: AgentMetrics,
) -> None:
    print()
    print(variant.name)
    print("-" * len(variant.name))

    print(
        f"EV:            "
        f"{float(metrics.estimated_ev):.6f} Ante"
    )
    print(
        f"Mean staked:   "
        f"{float(metrics.mean_staked):.6f} Ante"
    )
    print(
        f"Std deviation: "
        f"{metrics.standard_deviation:.6f}"
    )
    print(
        f"Std error:     "
        f"{metrics.standard_error:.6f}"
    )

    print("Actions:")

    for action in Action:
        print(
            f"  {action.name:<12} "
            f"{metrics.action_counts[action]}"
        )


def parse_args() -> PilotArgs:
    parser = ArgumentParser(
        description=(
            "Run the initial DQN pilot "
            "for all state and reward variants."
        )
    )

    parser.add_argument(
        "--training-episodes",
        type=int,
        default=DEFAULT_TRAINING_EPISODES,
    )

    parser.add_argument(
        "--evaluation-rounds",
        type=int,
        default=DEFAULT_EVALUATION_ROUNDS,
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
    )

    parser.add_argument(
        "--deck-seed",
        type=int,
        default=DEFAULT_DECK_SCHEDULE_SEED,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    namespace = parser.parse_args()

    training_episodes = cast(
        int,
        namespace.training_episodes,
    )

    evaluation_rounds = cast(
        int,
        namespace.evaluation_rounds,
    )

    training_seed = cast(
        int,
        namespace.training_seed,
    )

    deck_seed = cast(
        int,
        namespace.deck_seed,
    )

    output_dir = cast(
        Path,
        namespace.output_dir,
    )

    if training_episodes <= 0:
        parser.error(
            "--training-episodes must be positive"
        )

    if evaluation_rounds <= 0:
        parser.error(
            "--evaluation-rounds must be positive"
        )

    return PilotArgs(
        training_episodes=training_episodes,
        evaluation_rounds=evaluation_rounds,
        training_seed=training_seed,
        deck_seed=deck_seed,
        output_dir=output_dir,
    )


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_config = build_training_config(
        training_episodes=args.training_episodes,
        seed=args.training_seed,
    )

    evaluation_config = build_evaluation_config(
        round_count=args.evaluation_rounds,
        schedule_seed=args.deck_seed,
    )

    prefix = (
        f"dqn_pilot_"
        f"{args.training_episodes}_"
        f"{args.training_seed}_"
        f"{args.deck_seed}"
    )

    summary_path = (
        args.output_dir
        / f"{prefix}_summary.json"
    )

    training_path = (
        args.output_dir
        / f"{prefix}_training.csv"
    )

    episodes_path = (
        args.output_dir
        / f"{prefix}_episodes.csv"
    )

    summary_variants: dict[
        str,
        object,
    ] = {}

    with training_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as training_file, episodes_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as episodes_file:
        training_writer = csv.writer(
            training_file
        )

        evaluation_writer = csv.writer(
            episodes_file
        )

        training_writer.writerow(
            [
                "state_representation",
                "reward_type",
                "input_size",
                "episode",
                "total_reward",
                "steps",
                "optimizer_updates",
                "mean_loss",
                "epsilon",
            ]
        )

        evaluation_writer.writerow(
            [
                "round_index",
                "deck_seed",
                "state_representation",
                "reward_type",
                "net_profit",
                "total_staked",
                "outcome",
                "actions",
            ]
        )

        for variant in VARIANTS:
            print()
            print(
                f"Training {variant.name}"
            )

            training_result = train_dqn(
                state_encoder=build_state_encoder(
                    variant.state_representation
                ),
                reward_function=build_reward_function(
                    variant.reward_type
                ),
                config=training_config,
            )

            for stats in training_result.episode_stats:
                training_writer.writerow(
                    [
                        variant.state_representation.value,
                        variant.reward_type.value,
                        training_result.policy_network.input_size,
                        stats.episode,
                        stats.total_reward,
                        stats.steps,
                        stats.optimizer_updates,
                        stats.mean_loss,
                        stats.epsilon,
                    ]
                )

            training_result.agent.epsilon = 0.0
            training_result.policy_network.eval()

            evaluation_result = run_simulation(
                agent=training_result.agent,
                config=evaluation_config,
            )

            metrics = calculate_metrics(
                evaluation_result
            )

            for round_index, (
                deck_seed,
                episode,
            ) in enumerate(
                zip(
                    evaluation_result.config.deck_seeds,
                    evaluation_result.episodes,
                    strict=True,
                )
            ):
                evaluation_writer.writerow(
                    [
                        round_index,
                        deck_seed,
                        variant.state_representation.value,
                        variant.reward_type.value,
                        float(
                            episode.net_profit
                        ),
                        float(
                            episode.total_staked
                        ),
                        episode.outcome.name,
                        "|".join(
                            action.name
                            for action in episode.actions
                        ),
                    ]
                )

            print_metrics(
                variant=variant,
                metrics=metrics,
            )

            summary_variants[
                variant.name
            ] = {
                "state_representation": (
                    variant.state_representation.value
                ),
                "reward_type": (
                    variant.reward_type.value
                ),
                "input_size": (
                    training_result.policy_network.input_size
                ),
                "total_training_steps": (
                    training_result.total_steps
                ),
                "optimizer_updates": (
                    training_result.optimizer_updates
                ),
                "evaluation": metrics_to_dict(
                    metrics
                ),
            }

    summary: dict[str, object] = {
        "schema_version": 1,
        "experiment": "dqn_pilot",
        "training_seed": args.training_seed,
        "training_config": (
            training_config.to_dict()
        ),
        "evaluation_round_count": (
            evaluation_config.round_count
        ),
        "deck_schedule_seed": args.deck_seed,
        "deck_seeds": list(
            evaluation_config.deck_seeds
        ),
        "variants": summary_variants,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=2,
            sort_keys=True,
        )

        summary_file.write("\n")

    print()
    print("Saved results")
    print(f"Summary:  {summary_path}")
    print(f"Training: {training_path}")
    print(f"Episodes: {episodes_path}")


if __name__ == "__main__":
    main()