from dataclasses import replace

import pytest

from expert_poker_player.cards import Card, Rank, Suit

from expert_poker_player.uth import (
    Action,
    GamePhase,
    RoundOutcome,
)
from expert_poker_player.uth.models import (
    RoundState,
    StepResult,
    UTHObservation,
    observation_from_state,
    step_result_from_state,
)
from expert_poker_player.uth.rules import (
    legal_actions_for_phase,
)
from expert_poker_player.uth.settlement import (
    settle_fold,
    settle_showdown,
)
from expert_poker_player.hands import evaluate_best_hand
from expert_poker_player.uth.showdown import ShowdownResult


PLAYER_CARDS = (
    Card(rank=Rank.ACE, suit=Suit.SPADES),
    Card(rank=Rank.KING, suit=Suit.HEARTS),
)

DEALER_CARDS = (
    Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
    Card(rank=Rank.JACK, suit=Suit.CLUBS),
)

COMMUNITY_CARDS = (
    Card(rank=Rank.TEN, suit=Suit.SPADES),
    Card(rank=Rank.NINE, suit=Suit.HEARTS),
    Card(rank=Rank.EIGHT, suit=Suit.DIAMONDS),
    Card(rank=Rank.SEVEN, suit=Suit.CLUBS),
    Card(rank=Rank.SIX, suit=Suit.SPADES),
)

BURNED_CARDS = (
    Card(rank=Rank.FIVE, suit=Suit.HEARTS),
    Card(rank=Rank.FOUR, suit=Suit.DIAMONDS),
)

def make_showdown() -> ShowdownResult:
    return ShowdownResult(
        player_hand=evaluate_best_hand(
            (
                *PLAYER_CARDS,
                *COMMUNITY_CARDS,
            )
        ),
        dealer_hand=evaluate_best_hand(
            (
                *DEALER_CARDS,
                *COMMUNITY_CARDS,
            )
        ),
    )

def make_state(
    phase: GamePhase,
) -> RoundState:
    if phase is GamePhase.PREFLOP:
        return RoundState(
            phase=phase,
            player_cards=PLAYER_CARDS,
            dealer_cards=DEALER_CARDS,
            community_cards=(),
            burned_cards=(),
        )

    if phase is GamePhase.FLOP:
        return RoundState(
            phase=phase,
            player_cards=PLAYER_CARDS,
            dealer_cards=DEALER_CARDS,
            community_cards=COMMUNITY_CARDS[:3],
            burned_cards=BURNED_CARDS[:1],
        )

    if phase is GamePhase.RIVER:
        return RoundState(
            phase=phase,
            player_cards=PLAYER_CARDS,
            dealer_cards=DEALER_CARDS,
            community_cards=COMMUNITY_CARDS,
            burned_cards=BURNED_CARDS,
        )

    return RoundState(
        phase=GamePhase.TERMINAL,
        player_cards=PLAYER_CARDS,
        dealer_cards=DEALER_CARDS,
        community_cards=COMMUNITY_CARDS,
        burned_cards=BURNED_CARDS,
        play_multiplier=None,
        outcome=RoundOutcome.PLAYER_FOLD,
        settlement=settle_fold(),
    )


@pytest.mark.parametrize(
    "phase",
    [
        GamePhase.PREFLOP,
        GamePhase.FLOP,
        GamePhase.RIVER,
        GamePhase.TERMINAL,
    ],
)
def test_creates_valid_round_state_for_each_phase(
    phase: GamePhase,
) -> None:
    state = make_state(phase)

    assert state.phase is phase


@pytest.mark.parametrize(
    "phase",
    [
        GamePhase.PREFLOP,
        GamePhase.FLOP,
        GamePhase.RIVER,
        GamePhase.TERMINAL,
    ],
)
def test_observation_contains_legal_actions_for_phase(
    phase: GamePhase,
) -> None:
    state = make_state(phase)

    observation = observation_from_state(state)

    assert observation.legal_actions == legal_actions_for_phase(
        phase
    )
    assert observation.terminated is (
        phase is GamePhase.TERMINAL
    )


def test_observation_does_not_expose_dealer_cards() -> None:
    state = make_state(GamePhase.PREFLOP)

    observation = observation_from_state(state)

    assert observation.player_cards == PLAYER_CARDS
    assert not hasattr(observation, "dealer_cards")
    assert not hasattr(observation, "burned_cards")


def test_non_terminal_step_result_has_no_settlement() -> None:
    state = make_state(GamePhase.FLOP)

    result = step_result_from_state(state)

    assert not result.terminated
    assert result.outcome is None
    assert result.settlement is None


def test_terminal_step_result_contains_final_result() -> None:
    state = make_state(GamePhase.TERMINAL)

    result = step_result_from_state(state)

    assert result.terminated
    assert result.outcome is RoundOutcome.PLAYER_FOLD
    assert result.settlement == settle_fold()


def test_creates_valid_terminal_showdown_state() -> None:
    showdown = make_showdown()

    settlement = settle_showdown(
        player_hand=showdown.player_hand.value,
        dealer_hand=showdown.dealer_hand.value,
        play_multiplier=4,
    )

    state = RoundState(
        phase=GamePhase.TERMINAL,
        player_cards=PLAYER_CARDS,
        dealer_cards=DEALER_CARDS,
        community_cards=COMMUNITY_CARDS,
        burned_cards=BURNED_CARDS,
        play_multiplier=4,
        outcome=showdown.outcome,
        settlement=settlement,
        showdown=showdown,
    )

    assert state.play_multiplier == 4
    assert state.outcome is showdown.outcome
    assert state.showdown == showdown
    assert state.settlement == settlement
    
def test_round_state_rejects_invalid_phase_type() -> None: # type: ignore
    with pytest.raises(
        TypeError,
        match="phase must be an instance of GamePhase",
    ):
        RoundState(
            phase="preflop",  # type: ignore[arg-type]
            player_cards=PLAYER_CARDS,
            dealer_cards=DEALER_CARDS,
            community_cards=(),
            burned_cards=(),
        )


def test_round_state_rejects_non_tuple_cards() -> None:
    with pytest.raises(
        TypeError,
        match="player_cards must be a tuple",
    ):
        RoundState(
            phase=GamePhase.PREFLOP,
            player_cards=list(PLAYER_CARDS),  # type: ignore[arg-type]
            dealer_cards=DEALER_CARDS,
            community_cards=(),
            burned_cards=(),
        )


def test_round_state_rejects_wrong_number_of_player_cards() -> None:
    with pytest.raises(
        ValueError,
        match="player_cards must contain exactly 2 cards",
    ):
        RoundState(
            phase=GamePhase.PREFLOP,
            player_cards=PLAYER_CARDS[:1],  # type: ignore[arg-type]
            dealer_cards=DEALER_CARDS,
            community_cards=(),
            burned_cards=(),
        )


def test_round_state_rejects_non_card_element() -> None:
    with pytest.raises(
        TypeError,
        match="player_cards must contain only Card values",
    ):
        RoundState(
            phase=GamePhase.PREFLOP,
            player_cards=(
                PLAYER_CARDS[0],
                "not a card", # type: ignore
            ),  # type: ignore[arg-type]
            dealer_cards=DEALER_CARDS,
            community_cards=(),
            burned_cards=(),
        )


def test_round_state_rejects_duplicate_cards_in_field() -> None:
    with pytest.raises(
        ValueError,
        match="player_cards cannot contain duplicate cards",
    ):
        RoundState(
            phase=GamePhase.PREFLOP,
            player_cards=(
                PLAYER_CARDS[0],
                PLAYER_CARDS[0],
            ),
            dealer_cards=DEALER_CARDS,
            community_cards=(),
            burned_cards=(),
        )


def test_round_state_rejects_duplicate_cards_between_fields() -> None:
    with pytest.raises(
        ValueError,
        match="round state cannot contain duplicate cards",
    ):
        RoundState(
            phase=GamePhase.PREFLOP,
            player_cards=PLAYER_CARDS,
            dealer_cards=(
                DEALER_CARDS[0],
                PLAYER_CARDS[0],
            ),
            community_cards=(),
            burned_cards=(),
        )


def test_round_state_rejects_wrong_community_card_count() -> None:
    with pytest.raises(
        ValueError,
        match="FLOP requires exactly 3 community cards",
    ):
        RoundState(
            phase=GamePhase.FLOP,
            player_cards=PLAYER_CARDS,
            dealer_cards=DEALER_CARDS,
            community_cards=COMMUNITY_CARDS[:2],
            burned_cards=BURNED_CARDS[:1],
        )


def test_round_state_rejects_wrong_burned_card_count() -> None:
    with pytest.raises(
        ValueError,
        match="FLOP requires exactly 1 burned cards",
    ):
        RoundState(
            phase=GamePhase.FLOP,
            player_cards=PLAYER_CARDS,
            dealer_cards=DEALER_CARDS,
            community_cards=COMMUNITY_CARDS[:3],
            burned_cards=(),
        )


@pytest.mark.parametrize(
    "play_multiplier",
    [
        "4",
        4.0,
        True,
    ],
)
def test_round_state_rejects_non_integer_play_multiplier(
    play_multiplier: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="play_multiplier must be an integer or None",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            play_multiplier=play_multiplier,
        )


def test_round_state_rejects_invalid_play_multiplier() -> None:
    with pytest.raises(
        ValueError,
        match="play_multiplier must be one of: 1, 2, 3, 4",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            play_multiplier=5,
        )


def test_non_terminal_state_rejects_play_multiplier() -> None:
    with pytest.raises(
        ValueError,
        match="non-terminal state cannot have a Play multiplier",
    ):
        replace(
            make_state(GamePhase.PREFLOP),
            play_multiplier=4,
        )


def test_non_terminal_state_rejects_outcome() -> None:
    with pytest.raises(
        ValueError,
        match="non-terminal state cannot have a round outcome",
    ):
        replace(
            make_state(GamePhase.PREFLOP),
            outcome=RoundOutcome.PLAYER_WIN,
        )


def test_non_terminal_state_rejects_settlement() -> None:
    with pytest.raises(
        ValueError,
        match="non-terminal state cannot have a settlement",
    ):
        replace(
            make_state(GamePhase.PREFLOP),
            settlement=settle_fold(),
        )


def test_terminal_state_requires_outcome() -> None:
    with pytest.raises(
        ValueError,
        match="terminal state requires a round outcome",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            outcome=None,
        )


def test_terminal_state_requires_settlement() -> None:
    with pytest.raises(
        ValueError,
        match="terminal state requires a settlement",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            settlement=None,
        )


def test_folded_state_rejects_play_multiplier() -> None:
    with pytest.raises(
        ValueError,
        match="folded round cannot have a Play multiplier",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            play_multiplier=1,
        )


def test_showdown_state_requires_play_multiplier() -> None:
    with pytest.raises(
        ValueError,
        match="showdown result requires a Play multiplier",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            outcome=RoundOutcome.PLAYER_WIN,
        )


def test_round_state_rejects_invalid_outcome_type() -> None:
    with pytest.raises(
        TypeError,
        match="outcome must be an instance",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            outcome="player_win",  # type: ignore[arg-type]
        )


def test_round_state_rejects_invalid_settlement_type() -> None:
    with pytest.raises(
        TypeError,
        match="settlement must be an instance",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            settlement="settlement",  # type: ignore[arg-type]
        )

def test_observation_rejects_non_frozenset_actions() -> None:
    with pytest.raises(
        TypeError,
        match="legal_actions must be a frozenset",
    ):
        UTHObservation(
            phase=GamePhase.PREFLOP,
            player_cards=PLAYER_CARDS,
            community_cards=(),
            legal_actions={
                Action.CHECK,
            },  # type: ignore[arg-type]
        )


def test_observation_rejects_invalid_action_element() -> None:
    with pytest.raises(
        TypeError,
        match="must contain only Action values",
    ):
        UTHObservation(
            phase=GamePhase.PREFLOP,
            player_cards=PLAYER_CARDS,
            community_cards=(),
            legal_actions=frozenset(
                {
                    "check",
                }
            ),  # type: ignore[arg-type]
        )


def test_observation_rejects_actions_from_wrong_phase() -> None:
    with pytest.raises(
        ValueError,
        match="do not match the current phase",
    ):
        UTHObservation(
            phase=GamePhase.PREFLOP,
            player_cards=PLAYER_CARDS,
            community_cards=(),
            legal_actions=frozenset(
                {
                    Action.BET_1X,
                    Action.FOLD,
                }
            ),
        )


def test_non_terminal_observation_rejects_play_multiplier() -> None:
    with pytest.raises(
        ValueError,
        match="non-terminal observation cannot have",
    ):
        UTHObservation(
            phase=GamePhase.PREFLOP,
            player_cards=PLAYER_CARDS,
            community_cards=(),
            legal_actions=legal_actions_for_phase(
                GamePhase.PREFLOP
            ),
            play_multiplier=4,
        )


def test_step_result_rejects_invalid_observation() -> None:
    with pytest.raises(
        TypeError,
        match="observation must be an instance",
    ):
        StepResult(
            observation="observation",  # type: ignore[arg-type]
        )


def test_non_terminal_step_rejects_outcome() -> None:
    observation = observation_from_state(
        make_state(GamePhase.PREFLOP)
    )

    with pytest.raises(
        ValueError,
        match="non-terminal step result cannot have "
        "a round outcome",
    ):
        StepResult(
            observation=observation,
            outcome=RoundOutcome.PLAYER_WIN,
        )


def test_non_terminal_step_rejects_settlement() -> None:
    observation = observation_from_state(
        make_state(GamePhase.PREFLOP)
    )

    with pytest.raises(
        ValueError,
        match="non-terminal step result cannot have "
        "a settlement",
    ):
        StepResult(
            observation=observation,
            settlement=settle_fold(),
        )


def test_terminal_step_requires_outcome() -> None:
    observation = observation_from_state(
        make_state(GamePhase.TERMINAL)
    )

    with pytest.raises(
        ValueError,
        match="terminal step result requires a round outcome",
    ):
        StepResult(
            observation=observation,
            settlement=settle_fold(),
        )


def test_terminal_step_requires_settlement() -> None:
    observation = observation_from_state(
        make_state(GamePhase.TERMINAL)
    )

    with pytest.raises(
        ValueError,
        match="terminal step result requires a settlement",
    ):
        StepResult(
            observation=observation,
            outcome=RoundOutcome.PLAYER_FOLD,
        )


def test_observation_from_state_rejects_invalid_type() -> None:
    with pytest.raises(
        TypeError,
        match="state must be an instance of RoundState",
    ):
        observation_from_state("state")  # type: ignore[arg-type]


def test_step_result_from_state_rejects_invalid_type() -> None:
    with pytest.raises(
        TypeError,
        match="state must be an instance of RoundState",
    ):
        step_result_from_state("state")  # type: ignore[arg-type]

def test_observation_rejects_invalid_phase_type() -> None:
    with pytest.raises(
        TypeError,
        match="phase must be an instance of GamePhase",
    ):
        UTHObservation(
            phase="preflop",  # type: ignore[arg-type]
            player_cards=PLAYER_CARDS,
            community_cards=(),
            legal_actions=legal_actions_for_phase(
                GamePhase.PREFLOP
            ),
        )

def test_observation_rejects_wrong_community_card_count() -> None:
    with pytest.raises(
        ValueError,
        match="FLOP observation requires exactly "
        "3 community cards",
    ):
        UTHObservation(
            phase=GamePhase.FLOP,
            player_cards=PLAYER_CARDS,
            community_cards=COMMUNITY_CARDS[:2],
            legal_actions=legal_actions_for_phase(
                GamePhase.FLOP
            ),
        )

def test_observation_rejects_duplicate_visible_cards() -> None:
    community_cards = (
        PLAYER_CARDS[0],
        COMMUNITY_CARDS[1],
        COMMUNITY_CARDS[2],
    )

    with pytest.raises(
        ValueError,
        match="observation cannot contain duplicate cards",
    ):
        UTHObservation(
            phase=GamePhase.FLOP,
            player_cards=PLAYER_CARDS,
            community_cards=community_cards,
            legal_actions=legal_actions_for_phase(
                GamePhase.FLOP
            ),
        )

def test_non_terminal_state_rejects_showdown() -> None:
    with pytest.raises(
        ValueError,
        match="non-terminal state cannot have showdown details",
    ):
        replace(
            make_state(GamePhase.PREFLOP),
            showdown=make_showdown(),
        )

def test_folded_state_rejects_showdown() -> None:
    with pytest.raises(
        ValueError,
        match="folded round cannot have a showdown result",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            showdown=make_showdown(),
        )

def test_terminal_showdown_requires_details() -> None:
    with pytest.raises(
        ValueError,
        match="showdown result requires showdown details",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            play_multiplier=4,
            outcome=RoundOutcome.DEALER_WIN,
        )

def test_round_state_rejects_invalid_showdown_type() -> None:
    with pytest.raises(
        TypeError,
        match="showdown must be an instance",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            showdown="showdown",  # type: ignore[arg-type]
        )

def test_round_outcome_must_match_showdown() -> None:
    showdown = make_showdown()

    opposite_outcome = (
        RoundOutcome.PLAYER_WIN
        if showdown.outcome is RoundOutcome.DEALER_WIN
        else RoundOutcome.DEALER_WIN
    )

    with pytest.raises(
        ValueError,
        match="round outcome must match showdown outcome",
    ):
        replace(
            make_state(GamePhase.TERMINAL),
            play_multiplier=4,
            outcome=opposite_outcome,
            showdown=showdown,
        )

def test_non_terminal_step_rejects_showdown() -> None:
    observation = observation_from_state(
        make_state(GamePhase.PREFLOP)
    )

    with pytest.raises(
        ValueError,
        match="non-terminal step result cannot have "
        "showdown details",
    ):
        StepResult(
            observation=observation,
            showdown=make_showdown(),
        )

def test_folded_step_rejects_showdown() -> None:
    observation = observation_from_state(
        make_state(GamePhase.TERMINAL)
    )

    with pytest.raises(
        ValueError,
        match="folded step result cannot have showdown details",
    ):
        StepResult(
            observation=observation,
            outcome=RoundOutcome.PLAYER_FOLD,
            settlement=settle_fold(),
            showdown=make_showdown(),
        )

def test_showdown_step_requires_details() -> None:
    observation = observation_from_state(
        make_state(GamePhase.TERMINAL)
    )

    with pytest.raises(
        ValueError,
        match="showdown step result requires showdown details",
    ):
        StepResult(
            observation=observation,
            outcome=RoundOutcome.DEALER_WIN,
            settlement=settle_fold(),
        )

def test_step_outcome_must_match_showdown() -> None:
    observation = observation_from_state(
        make_state(GamePhase.TERMINAL)
    )
    showdown = make_showdown()

    opposite_outcome = (
        RoundOutcome.PLAYER_WIN
        if showdown.outcome is RoundOutcome.DEALER_WIN
        else RoundOutcome.DEALER_WIN
    )

    with pytest.raises(
        ValueError,
        match="step outcome must match showdown outcome",
    ):
        StepResult(
            observation=observation,
            outcome=opposite_outcome,
            settlement=settle_fold(),
            showdown=showdown,
        )
