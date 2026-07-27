import pytest

from expert_poker_player.uth import Action, GamePhase
from expert_poker_player.uth.rules import (
    legal_actions_for_phase,
)


@pytest.mark.parametrize(
    ("phase", "expected_actions"),
    [
        (
            GamePhase.PREFLOP,
            frozenset(
                {
                    Action.CHECK,
                    Action.BET_3X,
                    Action.BET_4X,
                }
            ),
        ),
        (
            GamePhase.FLOP,
            frozenset(
                {
                    Action.CHECK,
                    Action.BET_2X,
                }
            ),
        ),
        (
            GamePhase.RIVER,
            frozenset(
                {
                    Action.BET_1X,
                    Action.FOLD,
                }
            ),
        ),
        (
            GamePhase.TERMINAL,
            frozenset(),
        ),
    ], # type: ignore
)
def test_legal_actions_match_game_phase(
    phase: GamePhase,
    expected_actions: frozenset[Action],
) -> None:
    assert legal_actions_for_phase(phase) == expected_actions


def test_legal_actions_are_immutable() -> None:
    actions = legal_actions_for_phase(GamePhase.PREFLOP)

    assert isinstance(actions, frozenset)


def test_legal_actions_reject_invalid_phase_type() -> None:
    with pytest.raises(
        TypeError,
        match="phase must be an instance of GamePhase",
    ):
        legal_actions_for_phase("preflop")  # type: ignore[arg-type]