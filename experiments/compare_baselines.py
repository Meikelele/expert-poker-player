from argparse import ArgumentParser
from random import Random

from expert_poker_player.agents import (
    RandomAgent,
    RuleBasedAgent,
)
from expert_poker_player.evaluation import (
    AgentMetrics,
    SimulationConfig,
    calculate_metrics,
    run_simulation,
)
from expert_poker_player.uth import (
    Action,
    RoundOutcome,
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

    print()
    print("Comparison")
    print("----------")
    print(
        "RuleBasedAgent - RandomAgent EV: "
        f"{float(ev_difference):.6f} Ante"
    )


if __name__ == "__main__":
    main()