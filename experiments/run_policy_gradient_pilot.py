import csv
import json

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import cast

from expert_poker_player.evaluation import (
    AgentMetrics,
    SimulationConfig,
    calculate_metrics,
    run_simulation,
)
from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
    ProbeState,
    UntrainedControlResult,
    generate_probe_states,
    run_untrained_control,
    train_policy_gradient,
)
from expert_poker_player.rewards import (
    RewardType,
    build_reward_function,
)
from expert_poker_player.rl.actions import (
    ACTION_ORDER,
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

import math
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_TRAINING_EPISODES = 10_000
DEFAULT_EVALUATION_ROUNDS = 10_000

DEFAULT_TRAINING_SEED = 20260823
DEFAULT_DECK_SCHEDULE_SEED = 20260815

# Wspólny, niezależny od reprezentacji stanu i funkcji nagrody zestaw
# sond, żeby te same 100 rozdań można było porównywać między wariantami.
PROBE_STATES: tuple[ProbeState, ...] = generate_probe_states()


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
    learning_rate: float


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
    learning_rate: float,
    seed: int,
) -> PolicyGradientConfig:
    return PolicyGradientConfig(
        learning_rate=learning_rate,
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


def untrained_control_to_dict(
    *,
    control: UntrainedControlResult,
    input_size: int,
) -> dict[str, object]:
    return {
        "input_size": input_size,
        "action_counts": {
            action.name: control.action_counts[
                action
            ]
            for action in Action
        },
        "mean_max_probability": (
            control.mean_max_probability
        ),
        "mean_normalized_entropy": (
            sum(
                snapshot.normalized_entropy
                for snapshot in control.probe_snapshots
            )
            / len(
                control.probe_snapshots
            )
        ),
        "mean_preflop_action_probabilities": {
            action.value: probability
            for action, probability in zip(
                ACTION_ORDER,
                control.mean_preflop_probabilities,
            )
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
            "Run the initial Policy Gradient pilot "
            "for all state and reward variants."
        )
    )

    parser.add_argument(
        "--training-episodes",
        type=int,
        default=DEFAULT_TRAINING_EPISODES,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
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

    learning_rate = cast( # type: ignore
        float,
        namespace.learning_rate,
    )

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        parser.error(
            "--learning-rate must be positive and finite"
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
        learning_rate=learning_rate,
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
        learning_rate=args.learning_rate,
        seed=args.training_seed,
    )

    evaluation_config = build_evaluation_config(
        round_count=args.evaluation_rounds,
        schedule_seed=args.deck_seed,
    )

    prefix = (
        f"policy_gradient_pilot_"
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

    updates_path = (
        args.output_dir
        / f"{prefix}_updates.csv"
    )

    policy_diagnostics_path = (
        args.output_dir
        / f"{prefix}_policy_diagnostics.csv"
    )

    untrained_control_path = (
        args.output_dir
        / f"{prefix}_untrained_control.json"
    )

    summary_variants: dict[
        str,
        object,
    ] = {}

    untrained_control_variants: dict[
        str,
        object,
    ] = {}

    computed_state_representations: set[
        StateRepresentation
    ] = set()

    with training_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as training_file, episodes_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as episodes_file, updates_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as updates_file, policy_diagnostics_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as policy_diagnostics_file:
        training_writer = csv.writer(
            training_file
        )

        evaluation_writer = csv.writer(
            episodes_file
        )

        updates_writer = csv.writer(
            updates_file
        )

        policy_diagnostics_writer = csv.writer(
            policy_diagnostics_file
        )

        training_writer.writerow(
            [
                "state_representation",
                "reward_type",
                "input_size",
                "episode",
                "total_reward",
                "steps",
            ]
        )

        updates_writer.writerow(
            [
                "state_representation",
                "reward_type",
                "update",
                "first_episode",
                "last_episode",
                "batch_size",
                "loss",
                "gradient_norm",
                "cumulative_steps",
                "mean_episode_length",
                "mean_abs_return",
                "max_abs_return",
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

        policy_diagnostics_writer.writerow(
            [
                "state_representation",
                "reward_type",
                "training_seed",
                "update",
                "phase",
                "probe_index",
                "normalized_entropy",
                "max_probability",
                *(
                    f"prob_{action.value}"
                    for action in ACTION_ORDER
                ),
            ]
        )

        for variant in VARIANTS:
            print()
            print(
                f"Training {variant.name}"
            )

            if (
                variant.state_representation
                not in computed_state_representations
            ):
                computed_state_representations.add(
                    variant.state_representation
                )

                control = run_untrained_control(
                    state_encoder=build_state_encoder(
                        variant.state_representation
                    ),
                    config=training_config,
                    evaluation_config=evaluation_config,
                    probe_states=PROBE_STATES,
                )

                untrained_control_variants[
                    variant.state_representation.value
                ] = untrained_control_to_dict(
                    control=control,
                    input_size=build_state_encoder(
                        variant.state_representation
                    ).output_size,
                )

            training_result = (
                train_policy_gradient(
                    state_encoder=build_state_encoder(
                        variant.state_representation
                    ),
                    reward_function=build_reward_function(
                        variant.reward_type
                    ),
                    config=training_config,
                    probe_states=PROBE_STATES,
                )
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
                    ]
                )

            for update in training_result.update_stats:
                updates_writer.writerow(
                    [
                        variant.state_representation.value,
                        variant.reward_type.value,
                        update.update,
                        update.first_episode,
                        update.last_episode,
                        update.batch_size,
                        update.loss,
                        update.gradient_norm,
                        update.cumulative_steps,
                        update.mean_episode_length,
                        update.mean_abs_return,
                        update.max_abs_return,
                    ]
                )

            for snapshot in training_result.probe_snapshots:
                policy_diagnostics_writer.writerow(
                    [
                        variant.state_representation.value,
                        variant.reward_type.value,
                        args.training_seed,
                        snapshot.update,
                        snapshot.phase.value,
                        snapshot.probe_index,
                        snapshot.normalized_entropy,
                        snapshot.max_probability,
                        *snapshot.probabilities,
                    ]
                )

            training_result.agent.deterministic = True
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
        "experiment": "policy_gradient_pilot",
        "algorithm": "reinforce",
        "training_seed": args.training_seed,
        "randomness_design": (
            "common_training_seed_across_variants"
        ),
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

    untrained_control: dict[str, object] = {
        "training_seed": args.training_seed,
        "deck_schedule_seed": args.deck_seed,
        "evaluation_round_count": (
            evaluation_config.round_count
        ),
        "probe_count": len(
            {
                probe.probe_index
                for probe in PROBE_STATES
            }
        ),
        "variants": untrained_control_variants,
    }

    with untrained_control_path.open(
        "w",
        encoding="utf-8",
    ) as untrained_control_file:
        json.dump(
            untrained_control,
            untrained_control_file,
            indent=2,
            sort_keys=True,
        )

        untrained_control_file.write("\n")

    print()
    print("Saved results")
    print(f"Summary:            {summary_path}")
    print(f"Training:           {training_path}")
    print(f"Episodes:           {episodes_path}")
    print(f"Updates:            {updates_path}")
    print(f"Policy diagnostics: {policy_diagnostics_path}")
    print(f"Untrained control:  {untrained_control_path}")


if __name__ == "__main__":
    main()
