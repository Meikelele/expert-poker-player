from fractions import Fraction
from math import sqrt
from statistics import stdev

import pytest

from expert_poker_player.evaluation import (
    EpisodeResult,
    SimulationConfig,
    SimulationResult,
    calculate_metrics,
)
from expert_poker_player.uth import (
    Action,
    RoundOutcome,
    Settlement,
    WagerSettlement,
)


def make_wager(
    *,
    stake: int,
    net_profit: int,
) -> WagerSettlement:
    return WagerSettlement(
        stake=Fraction(stake),
        net_profit=Fraction(net_profit),
    )


def make_episode(
    *,
    actions: tuple[Action, ...],
    outcome: RoundOutcome,
    ante_profit: int,
    blind_profit: int,
    play_stake: int,
    play_profit: int,
) -> EpisodeResult:
    return EpisodeResult(
        actions=actions,
        outcome=outcome,
        settlement=Settlement(
            ante=make_wager(
                stake=1,
                net_profit=ante_profit,
            ),
            blind=make_wager(
                stake=1,
                net_profit=blind_profit,
            ),
            play=make_wager(
                stake=play_stake,
                net_profit=play_profit,
            ),
        ),
    )


def make_simulation_result() -> SimulationResult:
    episodes = (
        make_episode(
            actions=(Action.BET_4X,),
            outcome=RoundOutcome.PLAYER_WIN,
            ante_profit=1,
            blind_profit=1,
            play_stake=4,
            play_profit=4,
        ),
        make_episode(
            actions=(Action.BET_4X,),
            outcome=RoundOutcome.DEALER_WIN,
            ante_profit=-1,
            blind_profit=-1,
            play_stake=4,
            play_profit=-4,
        ),
        make_episode(
            actions=(
                Action.CHECK,
                Action.BET_2X,
            ),
            outcome=RoundOutcome.PUSH,
            ante_profit=0,
            blind_profit=0,
            play_stake=2,
            play_profit=0,
        ),
        make_episode(
            actions=(
                Action.CHECK,
                Action.CHECK,
                Action.FOLD,
            ),
            outcome=RoundOutcome.PLAYER_FOLD,
            ante_profit=-1,
            blind_profit=-1,
            play_stake=0,
            play_profit=0,
        ),
    )

    return SimulationResult(
        config=SimulationConfig(
            deck_seeds=(101, 202, 303, 404),
        ),
        episodes=episodes,
    )


def test_calculates_financial_metrics() -> None:
    metrics = calculate_metrics(
        make_simulation_result()
    )

    assert metrics.round_count == 4

    assert metrics.total_net_profit == Fraction(-2)
    assert metrics.mean_net_profit == Fraction(-1, 2)
    assert metrics.estimated_ev == Fraction(-1, 2)

    assert metrics.total_staked == Fraction(18)
    assert metrics.mean_staked == Fraction(9, 2)


def test_calculates_variability_metrics() -> None:
    metrics = calculate_metrics(
        make_simulation_result()
    )

    values = [6.0, -6.0, 0.0, -2.0]
    expected_deviation = stdev(values)

    assert metrics.standard_deviation == pytest.approx(
        expected_deviation
    )
    assert metrics.standard_error == pytest.approx(
        expected_deviation / sqrt(4)
    )


def test_counts_round_outcomes() -> None:
    metrics = calculate_metrics(
        make_simulation_result()
    )

    assert metrics.outcome_counts == {
        RoundOutcome.PLAYER_WIN: 1,
        RoundOutcome.DEALER_WIN: 1,
        RoundOutcome.PUSH: 1,
        RoundOutcome.PLAYER_FOLD: 1,
    }


def test_counts_agent_actions() -> None:
    metrics = calculate_metrics(
        make_simulation_result()
    )

    assert metrics.action_counts[Action.BET_4X] == 2
    assert metrics.action_counts[Action.BET_3X] == 0
    assert metrics.action_counts[Action.BET_2X] == 1
    assert metrics.action_counts[Action.BET_1X] == 0
    assert metrics.action_counts[Action.CHECK] == 3
    assert metrics.action_counts[Action.FOLD] == 1


def test_single_episode_has_zero_variability() -> None:
    episode = make_episode(
        actions=(Action.BET_4X,),
        outcome=RoundOutcome.PLAYER_WIN,
        ante_profit=1,
        blind_profit=1,
        play_stake=4,
        play_profit=4,
    )

    result = SimulationResult(
        config=SimulationConfig(
            deck_seeds=(101,),
        ),
        episodes=(episode,),
    )

    metrics = calculate_metrics(result)

    assert metrics.standard_deviation == 0.0
    assert metrics.standard_error == 0.0


def test_rejects_invalid_simulation_result() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "result must be an instance "
            "of SimulationResult"
        ),
    ):
        calculate_metrics(
            object(),  # type: ignore[arg-type]
        )