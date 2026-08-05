from fractions import Fraction

import pytest

from expert_poker_player.evaluation import (
    EpisodeResult,
    SimulationConfig,
    SimulationResult,
)
from expert_poker_player.uth import (
    Action,
    RoundOutcome,
    Settlement,
    WagerSettlement,
)


def make_settlement(
    *,
    play_stake: int,
) -> Settlement:
    return Settlement(
        ante=WagerSettlement(
            stake=Fraction(1),
            net_profit=Fraction(0),
        ),
        blind=WagerSettlement(
            stake=Fraction(1),
            net_profit=Fraction(0),
        ),
        play=WagerSettlement(
            stake=Fraction(play_stake),
            net_profit=Fraction(0),
        ),
    )

def make_episode_result() -> EpisodeResult:
    return EpisodeResult(
        actions=(Action.BET_4X,),
        outcome=RoundOutcome.PUSH,
        settlement=make_settlement(
            play_stake=4,
        ),
    )

@pytest.mark.parametrize(
    (
        "actions",
        "outcome",
        "play_stake",
        "expected_multiplier",
    ),
    [
        (
            (Action.BET_4X,),
            RoundOutcome.PUSH,
            4,
            4,
        ),
        (
            (Action.BET_3X,),
            RoundOutcome.PUSH,
            3,
            3,
        ),
        (
            (Action.CHECK, Action.BET_2X),
            RoundOutcome.PUSH,
            2,
            2,
        ),
        (
            (
                Action.CHECK,
                Action.CHECK,
                Action.BET_1X,
            ),
            RoundOutcome.PUSH,
            1,
            1,
        ),
        (
            (
                Action.CHECK,
                Action.CHECK,
                Action.FOLD,
            ),
            RoundOutcome.PLAYER_FOLD,
            0,
            None,
        ),
    ],
)
def test_accepts_completed_action_sequences(
    actions: tuple[Action, ...],
    outcome: RoundOutcome,
    play_stake: int,
    expected_multiplier: int | None,
) -> None:
    result = EpisodeResult(
        actions=actions,
        outcome=outcome,
        settlement=make_settlement(
            play_stake=play_stake,
        ),
    )

    assert result.decision_count == len(actions)
    assert result.play_multiplier == expected_multiplier
    assert result.folded is (
        outcome is RoundOutcome.PLAYER_FOLD
    )
    assert result.net_profit == 0
    assert result.total_staked == 2 + play_stake


def test_rejects_actions_that_are_not_a_tuple() -> None:
    with pytest.raises(
        TypeError,
        match="actions must be a tuple",
    ):
        EpisodeResult(
            actions=[Action.BET_4X],  # type: ignore[arg-type]
            outcome=RoundOutcome.PUSH,
            settlement=make_settlement(play_stake=4),
        )


def test_rejects_non_action_values() -> None:
    with pytest.raises(
        TypeError,
        match="actions must contain only Action values",
    ):
        EpisodeResult(
            actions=("bet_4x",),  # type: ignore[arg-type]
            outcome=RoundOutcome.PUSH,
            settlement=make_settlement(play_stake=4),
        )


def test_rejects_incomplete_or_impossible_sequence() -> None:
    with pytest.raises(
        ValueError,
        match="actions must represent a completed UTH round",
    ):
        EpisodeResult(
            actions=(Action.CHECK, Action.BET_1X),
            outcome=RoundOutcome.PUSH,
            settlement=make_settlement(play_stake=1),
        )


def test_rejects_invalid_outcome_type() -> None:
    with pytest.raises(
        TypeError,
        match="outcome must be an instance of RoundOutcome",
    ):
        EpisodeResult(
            actions=(Action.BET_4X,),
            outcome="push",  # type: ignore[arg-type]
            settlement=make_settlement(play_stake=4),
        )


def test_rejects_invalid_settlement_type() -> None:
    with pytest.raises(
        TypeError,
        match="settlement must be an instance of Settlement",
    ):
        EpisodeResult(
            actions=(Action.BET_4X,),
            outcome=RoundOutcome.PUSH,
            settlement=object(),  # type: ignore[arg-type]
        )


def test_fold_requires_player_fold_outcome() -> None:
    with pytest.raises(
        ValueError,
        match="fold action sequence requires PLAYER_FOLD outcome",
    ):
        EpisodeResult(
            actions=(
                Action.CHECK,
                Action.CHECK,
                Action.FOLD,
            ),
            outcome=RoundOutcome.PUSH,
            settlement=make_settlement(play_stake=0),
        )


def test_fold_rejects_play_stake() -> None:
    with pytest.raises(
        ValueError,
        match="folded episode cannot contain a Play stake",
    ):
        EpisodeResult(
            actions=(
                Action.CHECK,
                Action.CHECK,
                Action.FOLD,
            ),
            outcome=RoundOutcome.PLAYER_FOLD,
            settlement=make_settlement(play_stake=1),
        )


def test_bet_rejects_player_fold_outcome() -> None:
    with pytest.raises(
        ValueError,
        match="bet action sequence cannot have PLAYER_FOLD outcome",
    ):
        EpisodeResult(
            actions=(Action.BET_4X,),
            outcome=RoundOutcome.PLAYER_FOLD,
            settlement=make_settlement(play_stake=4),
        )


def test_play_stake_must_match_final_bet() -> None:
    with pytest.raises(
        ValueError,
        match="Play stake must match the final bet action",
    ):
        EpisodeResult(
            actions=(Action.BET_4X,),
            outcome=RoundOutcome.PUSH,
            settlement=make_settlement(play_stake=3),
        )

def test_simulation_config_accepts_deck_seeds() -> None:
    config = SimulationConfig(
        deck_seeds=(101, 202, 303),
    )

    assert config.deck_seeds == (101, 202, 303)
    assert config.round_count == 3


def test_simulation_config_rejects_non_tuple_seeds() -> None:
    with pytest.raises(
        TypeError,
        match="deck_seeds must be a tuple",
    ):
        SimulationConfig(
            deck_seeds=[101, 202],  # type: ignore[arg-type]
        )


def test_simulation_config_rejects_empty_seeds() -> None:
    with pytest.raises(
        ValueError,
        match="deck_seeds cannot be empty",
    ):
        SimulationConfig(
            deck_seeds=(),
        )


@pytest.mark.parametrize(
    "deck_seeds",
    [
        (101, "202"),
        (101, 2.5),
        (101, True),
    ],
)
def test_simulation_config_rejects_invalid_seed_values(
    deck_seeds: tuple[object, ...],
) -> None:
    with pytest.raises(
        TypeError,
        match="deck_seeds must contain only integers",
    ):
        SimulationConfig(
            deck_seeds=deck_seeds,  # type: ignore[arg-type]
        )


def test_simulation_result_accepts_matching_episodes() -> None:
    config = SimulationConfig(
        deck_seeds=(101, 202),
    )
    episodes = (
        make_episode_result(),
        make_episode_result(),
    )

    result = SimulationResult(
        config=config,
        episodes=episodes,
    )

    assert result.config is config
    assert result.episodes == episodes
    assert result.round_count == 2


def test_simulation_result_rejects_invalid_config() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "config must be an instance "
            "of SimulationConfig"
        ),
    ):
        SimulationResult(
            config=object(),  # type: ignore[arg-type]
            episodes=(),
        )


def test_simulation_result_rejects_non_tuple_episodes() -> None:
    config = SimulationConfig(
        deck_seeds=(101,),
    )

    with pytest.raises(
        TypeError,
        match="episodes must be a tuple",
    ):
        SimulationResult(
            config=config,
            episodes=[  # type: ignore[arg-type]
                make_episode_result(),
            ],
        )


def test_simulation_result_rejects_invalid_episode_values() -> None:
    config = SimulationConfig(
        deck_seeds=(101,),
    )

    with pytest.raises(
        TypeError,
        match=(
            "episodes must contain only "
            "EpisodeResult values"
        ),
    ):
        SimulationResult(
            config=config,
            episodes=(object(),),  # type: ignore[arg-type]
        )


def test_simulation_result_requires_one_episode_per_seed() -> None:
    config = SimulationConfig(
        deck_seeds=(101, 202),
    )

    with pytest.raises(
        ValueError,
        match=(
            "episode count must match "
            "the configured round count"
        ),
    ):
        SimulationResult(
            config=config,
            episodes=(make_episode_result(),),
        )