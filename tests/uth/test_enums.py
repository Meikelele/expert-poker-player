from expert_poker_player.uth import (
    Action,
    GamePhase,
    RoundOutcome,
    WagerOutcome,
)


def test_action_values_are_stable() -> None:
    assert Action.CHECK.value == "check"
    assert Action.BET_4X.value == "bet_4x"
    assert Action.BET_3X.value == "bet_3x"
    assert Action.BET_2X.value == "bet_2x"
    assert Action.BET_1X.value == "bet_1x"
    assert Action.FOLD.value == "fold"


def test_all_actions_have_unique_values() -> None:
    values = [action.value for action in Action]

    assert len(values) == len(set(values))


def test_game_phase_values_are_stable() -> None:
    assert GamePhase.PREFLOP.value == "preflop"
    assert GamePhase.FLOP.value == "flop"
    assert GamePhase.RIVER.value == "river"
    assert GamePhase.TERMINAL.value == "terminal"


def test_round_outcomes_are_defined_from_player_perspective() -> None:
    assert RoundOutcome.PLAYER_WIN.value == "player_win"
    assert RoundOutcome.DEALER_WIN.value == "dealer_win"
    assert RoundOutcome.PUSH.value == "push"
    assert RoundOutcome.PLAYER_FOLD.value == "player_fold"


def test_wager_outcomes_include_not_placed() -> None:
    assert WagerOutcome.WIN.value == "win"
    assert WagerOutcome.LOSS.value == "loss"
    assert WagerOutcome.PUSH.value == "push"
    assert WagerOutcome.NOT_PLACED.value == "not_placed"