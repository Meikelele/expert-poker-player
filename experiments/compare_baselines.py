import csv
import json

from argparse import ArgumentParser
from pathlib import Path
from random import Random

from expert_poker_player.agents import (
    RandomAgent,
    RuleBasedAgent,
)
from expert_poker_player.evaluation import (
    AgentMetrics,
    SimulationConfig,
    SimulationResult,
    calculate_metrics,
    run_simulation,
)
from expert_poker_player.uth import (
    Action,
    RoundOutcome,
)

DEFAULT_OUTPUT_DIR = Path(
    "experiments/results"
)


DEFAULT_ROUNDS = 10_000
DEFAULT_DECK_SCHEDULE_SEED = 20260815
DEFAULT_RANDOM_AGENT_SEED = 20260816


def build_config(
    *,
    round_count: int,
    schedule_seed: int,
) -> SimulationConfig:
    """Buduje powtarzalny harmonogram rozdań."""

    deck_random = Random(schedule_seed)

    deck_seeds = tuple(
        deck_random.getrandbits(63)
        for _ in range(round_count)
    )

    return SimulationConfig(
        deck_seeds=deck_seeds,
    )

def metrics_to_dict(
    metrics: AgentMetrics,
) -> dict[str, object]:
    """Konwertuje metryki agenta do formatu zapisywalnego jako JSON."""

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

def save_summary(
    *,
    output_path: Path,
    config: SimulationConfig,
    deck_schedule_seed: int,
    random_agent_seed: int,
    random_metrics: AgentMetrics,
    rule_based_metrics: AgentMetrics,
) -> None:
    """Zapisuje konfigurację i zagregowane wyniki eksperymentu."""

    summary: dict[str, object] = {
        "schema_version": 1,
        "experiment": "baseline_comparison",
        "round_count": config.round_count,
        "deck_schedule_seed": deck_schedule_seed,
        "random_agent_seed": random_agent_seed,
        "deck_seeds": list(
            config.deck_seeds
        ),
        "agents": {
            "RandomAgent": metrics_to_dict(
                random_metrics
            ),
            "RuleBasedAgent": metrics_to_dict(
                rule_based_metrics
            ),
        },
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
            sort_keys=True,
        )

        output_file.write("\n")

def write_agent_episodes(
    *,
    writer: "csv.DictWriter[str]",
    agent_name: str,
    result: SimulationResult,
) -> None:
    """Zapisuje wyniki poszczególnych rozdań jednego agenta."""

    for round_index, (
        deck_seed,
        episode,
    ) in enumerate(
        zip(
            result.config.deck_seeds,
            result.episodes,
            strict=True,
        )
    ):
        writer.writerow(
            {
                "round_index": round_index,
                "deck_seed": deck_seed,
                "agent": agent_name,
                "net_profit": float(
                    episode.net_profit
                ),
                "total_staked": float(
                    episode.total_staked
                ),
                "outcome": episode.outcome.name,
                "actions": "|".join(
                    action.name
                    for action in episode.actions
                ),
            }
        )

def save_episodes(
    *,
    output_path: Path,
    random_result: SimulationResult,
    rule_based_result: SimulationResult,
) -> None:
    """Zapisuje wyniki wszystkich rozdań obu agentów."""

    fieldnames = [
        "round_index",
        "deck_seed",
        "agent",
        "net_profit",
        "total_staked",
        "outcome",
        "actions",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        write_agent_episodes(
            writer=writer,
            agent_name="RandomAgent",
            result=random_result,
        )

        write_agent_episodes(
            writer=writer,
            agent_name="RuleBasedAgent",
            result=rule_based_result,
        )

def save_results(
    *,
    output_dir: Path,
    config: SimulationConfig,
    deck_schedule_seed: int,
    random_agent_seed: int,
    random_result: SimulationResult,
    rule_based_result: SimulationResult,
    random_metrics: AgentMetrics,
    rule_based_metrics: AgentMetrics,
) -> tuple[Path, Path]:
    """Zapisuje kompletny wynik porównania baseline'ów."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"baseline_{config.round_count}_"
        f"{deck_schedule_seed}_"
        f"{random_agent_seed}"
    )

    summary_path = (
        output_dir
        / f"{prefix}_summary.json"
    )

    episodes_path = (
        output_dir
        / f"{prefix}_episodes.csv"
    )

    save_summary(
        output_path=summary_path,
        config=config,
        deck_schedule_seed=deck_schedule_seed,
        random_agent_seed=random_agent_seed,
        random_metrics=random_metrics,
        rule_based_metrics=rule_based_metrics,
    )

    save_episodes(
        output_path=episodes_path,
        random_result=random_result,
        rule_based_result=rule_based_result,
    )

    return (
        summary_path,
        episodes_path,
    )

def print_metrics(
    name: str,
    metrics: AgentMetrics,
) -> None:
    """Wyświetla podstawowe wyniki agenta."""

    print()
    print(name)
    print("-" * len(name))

    print(
        f"Rounds:              "
        f"{metrics.round_count}"
    )
    print(
        f"Total net profit:    "
        f"{float(metrics.total_net_profit):.6f} Ante"
    )
    print(
        f"Estimated EV:        "
        f"{float(metrics.estimated_ev):.6f} Ante"
    )
    print(
        f"Mean staked:         "
        f"{float(metrics.mean_staked):.6f} Ante"
    )
    print(
        f"Standard deviation:  "
        f"{metrics.standard_deviation:.6f}"
    )
    print(
        f"Standard error:      "
        f"{metrics.standard_error:.6f}"
    )

    print()
    print("Outcomes:")

    for outcome in RoundOutcome:
        count = metrics.outcome_counts[outcome]
        rate = count / metrics.round_count

        print(
            f"  {outcome.name:<16} "
            f"{count:>8} "
            f"{rate:>8.2%}"
        )

    print()
    print("Actions:")

    total_actions = sum(
        metrics.action_counts.values()
    )

    for action in Action:
        count = metrics.action_counts[action]

        rate = (
            count / total_actions
            if total_actions
            else 0.0
        )

        print(
            f"  {action.name:<16} "
            f"{count:>8} "
            f"{rate:>8.2%}"
        )


def parse_args():
    parser = ArgumentParser(
        description=(
            "Compare baseline UTH agents "
            "on the same deck schedule."
        )
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=(
            "number of rounds to simulate "
            f"(default: {DEFAULT_ROUNDS})"
        ),
    )
    parser.add_argument(
        "--deck-seed",
        type=int,
        default=DEFAULT_DECK_SCHEDULE_SEED,
        help="seed used to generate the deck schedule",
    )
    parser.add_argument(
        "--random-agent-seed",
        type=int,
        default=DEFAULT_RANDOM_AGENT_SEED,
        help="seed used by RandomAgent",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "directory used to store "
            "experiment results"
        ),
    )
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error(
            "--rounds must be greater than zero"
        )

    return args


def main() -> None:
    args = parse_args()

    config = build_config(
        round_count=args.rounds,
        schedule_seed=args.deck_seed,
    )

    random_result = run_simulation(
        agent=RandomAgent(
            seed=args.random_agent_seed,
        ),
        config=config,
    )

    rule_based_result = run_simulation(
        agent=RuleBasedAgent(),
        config=config,
    )

    random_metrics = calculate_metrics(
        random_result
    )
    rule_based_metrics = calculate_metrics(
        rule_based_result
    )

    print(
        "Baseline comparison"
    )
    print(
        f"Deck schedule seed:  {args.deck_seed}"
    )
    print(
        f"Random agent seed:   "
        f"{args.random_agent_seed}"
    )

    print_metrics(
        "RandomAgent",
        random_metrics,
    )
    print_metrics(
        "RuleBasedAgent",
        rule_based_metrics,
    )

    ev_difference = (
        rule_based_metrics.estimated_ev
        - random_metrics.estimated_ev
    )

    summary_path, episodes_path = save_results(
    output_dir=args.output_dir,
    config=config,
    deck_schedule_seed=args.deck_seed,
    random_agent_seed=args.random_agent_seed,
    random_result=random_result,
    rule_based_result=rule_based_result,
    random_metrics=random_metrics,
    rule_based_metrics=rule_based_metrics,
)

    print()
    print("Comparison")
    print("----------")
    print(
        "RuleBasedAgent - RandomAgent EV: "
        f"{float(ev_difference):.6f} Ante"
    )

    print()
    print("Saved results")
    print("-------------")
    print(
        f"Summary:  {summary_path}"
    )
    print(
        f"Episodes: {episodes_path}"
    )


if __name__ == "__main__":
    main()