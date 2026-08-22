from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from statistics import stdev
from types import MappingProxyType
from typing import Mapping

from expert_poker_player.evaluation.models import SimulationResult
from expert_poker_player.uth import (
    Action,
    RoundOutcome,
)


@dataclass(frozen=True, slots=True)
class AgentMetrics:
    """Zagregowane metryki uzyskane z symulacji agenta."""

    round_count: int

    total_net_profit: Fraction
    mean_net_profit: Fraction

    total_staked: Fraction
    mean_staked: Fraction

    standard_deviation: float
    standard_error: float

    outcome_counts: Mapping[RoundOutcome, int]
    action_counts: Mapping[Action, int]

    @property
    def estimated_ev(self) -> Fraction:
        """Zwraca empiryczne EV w jednostkach Ante na rozdanie."""

        return self.mean_net_profit


def calculate_metrics(
    result: SimulationResult,
) -> AgentMetrics:
    """Oblicza zagregowane metryki zakończonej symulacji."""

    if not isinstance(result, SimulationResult): # type: ignore
        raise TypeError(
            "result must be an instance of SimulationResult"
        )

    round_count = result.round_count

    total_net_profit = sum(
        (
            episode.net_profit
            for episode in result.episodes
        ),
        start=Fraction(0),
    )

    total_staked = sum(
        (
            episode.total_staked
            for episode in result.episodes
        ),
        start=Fraction(0),
    )

    mean_net_profit = (
        total_net_profit
        / round_count
    )

    mean_staked = (
        total_staked
        / round_count
    )

    net_profits = [
        float(episode.net_profit)
        for episode in result.episodes
    ]

    standard_deviation = (
        stdev(net_profits)
        if round_count > 1
        else 0.0
    )

    standard_error = (
        standard_deviation
        / sqrt(round_count)
    )

    outcome_counts = {
        outcome: 0
        for outcome in RoundOutcome
    }

    action_counts = {
        action: 0
        for action in Action
    }

    for episode in result.episodes:
        outcome_counts[episode.outcome] += 1

        for action in episode.actions:
            action_counts[action] += 1

    return AgentMetrics(
        round_count=round_count,
        total_net_profit=total_net_profit,
        mean_net_profit=mean_net_profit,
        total_staked=total_staked,
        mean_staked=mean_staked,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        outcome_counts=MappingProxyType(
            outcome_counts
        ),
        action_counts=MappingProxyType(
            action_counts
        ),
    )