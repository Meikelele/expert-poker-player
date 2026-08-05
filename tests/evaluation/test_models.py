from fractions import Fraction

import pytest

from expert_poker_player.evaluation import EpisodeResult
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