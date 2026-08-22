import csv
import json

from pathlib import Path

from experiments.compare_baselines import (
    build_config,
    save_results,
)
from expert_poker_player.agents import (
    RandomAgent,
    RuleBasedAgent,
)
from expert_poker_player.evaluation import (
    calculate_metrics,
    run_simulation,
)


def test_saves_reproducible_comparison_results(
    tmp_path: Path,
) -> None:
    config = build_config(
        round_count=3,
        schedule_seed=123,
    )

    random_result = run_simulation(
        agent=RandomAgent(seed=456),
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

    summary_path, episodes_path = save_results(
        output_dir=tmp_path,
        config=config,
        deck_schedule_seed=123,
        random_agent_seed=456,
        random_result=random_result,
        rule_based_result=rule_based_result,
        random_metrics=random_metrics,
        rule_based_metrics=rule_based_metrics,
    )

    with summary_path.open(
        encoding="utf-8",
    ) as summary_file:
        summary = json.load(
            summary_file
        )

    assert summary["round_count"] == 3
    assert summary["deck_schedule_seed"] == 123
    assert summary["random_agent_seed"] == 456
    assert summary["deck_seeds"] == list(
        config.deck_seeds
    )

    assert set(summary["agents"]) == {
        "RandomAgent",
        "RuleBasedAgent",
    }

    with episodes_path.open(
        encoding="utf-8",
        newline="",
    ) as episodes_file:
        rows = list(
            csv.DictReader(
                episodes_file
            )
        )

    assert len(rows) == 6

    assert [
        row["agent"]
        for row in rows
    ] == [
        "RandomAgent",
        "RandomAgent",
        "RandomAgent",
        "RuleBasedAgent",
        "RuleBasedAgent",
        "RuleBasedAgent",
    ]

    assert [
        int(row["deck_seed"])
        for row in rows[:3]
    ] == list(
        config.deck_seeds
    )

    assert [
        int(row["deck_seed"])
        for row in rows[3:]
    ] == list(
        config.deck_seeds
    )
    
def test_build_config_is_reproducible() -> None:
    first = build_config(
        round_count=5,
        schedule_seed=123,
    )

    second = build_config(
        round_count=5,
        schedule_seed=123,
    )

    assert first == second