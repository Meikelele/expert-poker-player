from fractions import Fraction
import pytest

from expert_poker_player.cards import Card, Rank, Suit
from expert_poker_player.uth import (
    Action,
    FixedDeck,
    GamePhase,
    IllegalActionError,
    RoundFinishedError,
    RoundNotStartedError,
    RoundOutcome,
    UTHGame,
)

ROUND_DRAW_ORDER = (
    # Player 1
    Card(rank=Rank.ACE, suit=Suit.SPADES),

    # Dealer 1
    Card(rank=Rank.KING, suit=Suit.HEARTS),

    # Player 2
    Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),

    # Dealer 2
    Card(rank=Rank.JACK, suit=Suit.CLUBS),

    # Burn przed flopem
    Card(rank=Rank.TEN, suit=Suit.SPADES),

    # Flop
    Card(rank=Rank.NINE, suit=Suit.HEARTS),
    Card(rank=Rank.EIGHT, suit=Suit.DIAMONDS),
    Card(rank=Rank.SEVEN, suit=Suit.CLUBS),

    # Burn przed turnem i riverem
    Card(rank=Rank.SIX, suit=Suit.SPADES),

    # Turn i river
    Card(rank=Rank.FIVE, suit=Suit.HEARTS),
    Card(rank=Rank.FOUR, suit=Suit.DIAMONDS),
)

INITIAL_DRAW_ORDER = (
    Card(rank=Rank.ACE, suit=Suit.SPADES),
    Card(rank=Rank.KING, suit=Suit.HEARTS),
    Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
    Card(rank=Rank.JACK, suit=Suit.CLUBS),
)

def start_fixed_game() -> tuple[UTHGame, FixedDeck]:
    game = UTHGame()
    deck = FixedDeck(ROUND_DRAW_ORDER)

    game.reset(card_source=deck)

    return game, deck

def test_preflop_check_reveals_flop() -> None:
    game, deck = start_fixed_game()

    result = game.step(Action.CHECK)

    assert not result.terminated
    assert result.outcome is None
    assert result.settlement is None

    assert game.state.phase is GamePhase.FLOP
    assert game.state.burned_cards == (
        ROUND_DRAW_ORDER[4],
    )
    assert game.state.community_cards == (
        ROUND_DRAW_ORDER[5],
        ROUND_DRAW_ORDER[6],
        ROUND_DRAW_ORDER[7],
    )

    assert result.observation.legal_actions == frozenset(
        {
            Action.CHECK,
            Action.BET_2X,
        }
    )

    assert len(deck) == 3

def test_flop_check_reveals_turn_and_river() -> None:
    game, deck = start_fixed_game()

    game.step(Action.CHECK)
    result = game.step(Action.CHECK)

    assert not result.terminated
    assert game.state.phase is GamePhase.RIVER

    assert game.state.burned_cards == (
        ROUND_DRAW_ORDER[4],
        ROUND_DRAW_ORDER[8],
    )

    assert game.state.community_cards == (
        ROUND_DRAW_ORDER[5],
        ROUND_DRAW_ORDER[6],
        ROUND_DRAW_ORDER[7],
        ROUND_DRAW_ORDER[9],
        ROUND_DRAW_ORDER[10],
    )

    assert result.observation.legal_actions == frozenset(
        {
            Action.BET_1X,
            Action.FOLD,
        }
    )

    assert len(deck) == 0

@pytest.mark.parametrize(
    ("action", "expected_multiplier"),
    [
        (Action.BET_3X, 3),
        (Action.BET_4X, 4),
    ],
)
def test_preflop_bet_finishes_round(
    action: Action,
    expected_multiplier: int,
) -> None:
    game, deck = start_fixed_game()

    result = game.step(action)

    assert result.terminated
    assert result.observation.phase is GamePhase.TERMINAL
    assert result.observation.legal_actions == frozenset()

    assert result.outcome is RoundOutcome.PLAYER_WIN
    assert result.settlement is not None

    assert game.state.play_multiplier == expected_multiplier

    assert result.settlement.ante.net_profit == Fraction(0)
    assert result.settlement.blind.net_profit == Fraction(0)
    assert result.settlement.play.net_profit == Fraction(
        expected_multiplier
    )
    assert result.settlement.total_net_profit == Fraction(
        expected_multiplier
    )

    assert len(deck) == 0

def test_flop_bet_2x_finishes_round() -> None:
    game, deck = start_fixed_game()

    game.step(Action.CHECK)
    result = game.step(Action.BET_2X)

    assert result.terminated
    assert result.outcome is RoundOutcome.PLAYER_WIN
    assert result.settlement is not None

    assert game.state.phase is GamePhase.TERMINAL
    assert game.state.play_multiplier == 2

    assert result.settlement.play.stake == Fraction(2)
    assert result.settlement.play.net_profit == Fraction(2)
    assert result.settlement.total_net_profit == Fraction(2)

    assert len(deck) == 0

def test_river_bet_1x_finishes_round() -> None:
    game, _ = start_fixed_game()

    game.step(Action.CHECK)
    game.step(Action.CHECK)

    result = game.step(Action.BET_1X)

    assert result.terminated
    assert result.outcome is RoundOutcome.PLAYER_WIN
    assert result.settlement is not None

    assert game.state.phase is GamePhase.TERMINAL
    assert game.state.play_multiplier == 1

    assert result.settlement.play.stake == Fraction(1)
    assert result.settlement.play.net_profit == Fraction(1)
    assert result.settlement.total_net_profit == Fraction(1)

def test_river_fold_loses_ante_and_blind() -> None:
    game, _ = start_fixed_game()

    game.step(Action.CHECK)
    game.step(Action.CHECK)

    result = game.step(Action.FOLD)

    assert result.terminated
    assert result.outcome is RoundOutcome.PLAYER_FOLD
    assert result.settlement is not None
    assert result.showdown is None
    assert game.state.showdown is None

    assert game.state.phase is GamePhase.TERMINAL
    assert game.state.play_multiplier is None

    assert result.settlement.ante.net_profit == Fraction(-1)
    assert result.settlement.blind.net_profit == Fraction(-1)
    assert result.settlement.play.stake == Fraction(0)
    assert result.settlement.total_net_profit == Fraction(-2)

def test_step_requires_started_round() -> None:
    game = UTHGame()

    with pytest.raises(
        RoundNotStartedError,
        match=r"call reset\(\) first",
    ):
        game.step(Action.CHECK)

def test_step_rejects_invalid_action_type() -> None:
    game, _ = start_fixed_game()

    with pytest.raises(
        TypeError,
        match="action must be an instance of Action",
    ):
        game.step("check")  # type: ignore[arg-type]

def test_step_rejects_action_illegal_during_phase() -> None:
    game, deck = start_fixed_game()
    initial_state = game.state
    cards_before_action = deck.cards

    with pytest.raises(
        IllegalActionError,
        match="bet_2x is illegal during preflop",
    ):
        game.step(Action.BET_2X)

    assert game.state == initial_state
    assert deck.cards == cards_before_action

def test_step_rejects_action_after_round_finished() -> None:
    game, _ = start_fixed_game()

    game.step(Action.BET_4X)

    with pytest.raises(
        RoundFinishedError,
        match="round has already finished",
    ):
        game.step(Action.CHECK)

def test_game_is_not_started_before_reset() -> None:
    game = UTHGame()

    assert not game.is_started

def test_state_requires_started_round() -> None:
    game = UTHGame()

    with pytest.raises(
        RoundNotStartedError,
        match=r"call reset\(\) first",
    ):
        _ = game.state

def test_observation_requires_started_round() -> None:
    game = UTHGame()

    with pytest.raises(
        RoundNotStartedError,
        match=r"call reset\(\) first",
    ):
        _ = game.observation

def test_reset_starts_preflop_round() -> None:
    game = UTHGame()

    observation = game.reset()

    assert game.is_started
    assert game.state.phase is GamePhase.PREFLOP
    assert observation.phase is GamePhase.PREFLOP
    assert observation.community_cards == ()
    assert not observation.terminated

def test_reset_returns_preflop_legal_actions() -> None:
    game = UTHGame()

    observation = game.reset()

    assert observation.legal_actions == frozenset(
        {
            Action.CHECK,
            Action.BET_3X,
            Action.BET_4X,
        }
    )

def test_reset_with_fixed_deck_uses_declared_draw_order() -> None:
    game = UTHGame()
    deck = FixedDeck(INITIAL_DRAW_ORDER)

    observation = game.reset(card_source=deck)

    assert observation.player_cards == (
        INITIAL_DRAW_ORDER[0],
        INITIAL_DRAW_ORDER[2],
    )

    assert game.state.dealer_cards == (
        INITIAL_DRAW_ORDER[1],
        INITIAL_DRAW_ORDER[3],
    )

    assert len(deck) == 0

def test_observation_does_not_expose_dealer_cards() -> None:
    game = UTHGame()

    observation = game.reset(
        card_source=FixedDeck(INITIAL_DRAW_ORDER)
    )

    assert not hasattr(observation, "dealer_cards")
    assert not hasattr(observation, "burned_cards")

def test_same_seed_produces_same_sequence_of_rounds() -> None:
    first_game = UTHGame(seed=42)
    second_game = UTHGame(seed=42)

    first_observation_a = first_game.reset()
    first_state_a = first_game.state

    first_observation_b = second_game.reset()
    first_state_b = second_game.state

    second_observation_a = first_game.reset()
    second_state_a = first_game.state

    second_observation_b = second_game.reset()
    second_state_b = second_game.state

    assert first_observation_a == first_observation_b
    assert first_state_a == first_state_b

    assert second_observation_a == second_observation_b
    assert second_state_a == second_state_b

def test_consecutive_rounds_use_fresh_decks() -> None:
    game = UTHGame(seed=42)

    first_observation = game.reset()
    first_dealer_cards = game.state.dealer_cards

    second_observation = game.reset()
    second_dealer_cards = game.state.dealer_cards

    first_deal = (
        *first_observation.player_cards,
        *first_dealer_cards,
    )
    second_deal = (
        *second_observation.player_cards,
        *second_dealer_cards,
    )

    assert first_deal != second_deal

@pytest.mark.parametrize(
    "seed",
    [
        "42",
        42.0,
        True,
    ],
)
def test_game_rejects_invalid_seed_type(
    seed: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="seed must be an integer or None",
    ):
        UTHGame(seed=seed)  # type: ignore[arg-type]

def test_reset_rejects_invalid_card_source() -> None:
    game = UTHGame()

    with pytest.raises(
        TypeError,
        match="card_source must implement CardSource",
    ):
        game.reset(
            card_source=object(),  # type: ignore[arg-type]
        )

def test_terminal_showdown_contains_best_five_cards() -> None:
    game, _ = start_fixed_game()

    result = game.step(Action.BET_4X)

    assert result.showdown is not None
    assert game.state.showdown is not None

    assert len(result.showdown.player_hand.cards) == 5
    assert len(result.showdown.dealer_hand.cards) == 5

    assert (
        result.showdown.player_hand
        == game.state.showdown.player_hand
    )
    assert (
        result.showdown.dealer_hand
        == game.state.showdown.dealer_hand
    )
    assert (
        result.showdown.outcome
        is result.outcome
    )

def test_history_is_disabled_by_default() -> None:
    game, _ = start_fixed_game()

    game.step(Action.CHECK)

    assert not game.history_enabled
    assert game.trace is None

def test_enabled_history_is_empty_before_reset() -> None:
    game = UTHGame(record_history=True)

    assert game.history_enabled
    assert game.trace is None

def test_trace_records_agent_observations_and_actions() -> None:
    game = UTHGame(record_history=True)
    game.reset(card_source=FixedDeck(ROUND_DRAW_ORDER))

    game.step(Action.CHECK)
    game.step(Action.CHECK)

    trace = game.trace

    assert trace is not None
    assert trace.round_id == 1
    assert not trace.completed
    assert len(trace.decisions) == 2

    first_decision = trace.decisions[0]

    assert first_decision.observation.phase is GamePhase.PREFLOP
    assert first_decision.action is Action.CHECK
    assert not hasattr(
        first_decision.observation,
        "dealer_cards",
    )

    second_decision = trace.decisions[1]

    assert second_decision.observation.phase is GamePhase.FLOP
    assert second_decision.action is Action.CHECK

def test_trace_contains_completed_fold_round() -> None:
    game = UTHGame(record_history=True)
    game.reset(card_source=FixedDeck(ROUND_DRAW_ORDER))

    game.step(Action.CHECK)
    game.step(Action.CHECK)
    game.step(Action.FOLD)

    trace = game.trace

    assert trace is not None
    assert trace.completed
    assert trace.outcome is RoundOutcome.PLAYER_FOLD
    assert trace.settlement is not None
    assert trace.showdown is None

    assert len(trace.decisions) == 3
    assert trace.decisions[-1].action is Action.FOLD

def test_trace_contains_completed_showdown_round() -> None:
    game = UTHGame(record_history=True)
    game.reset(card_source=FixedDeck(ROUND_DRAW_ORDER))

    game.step(Action.BET_4X)

    trace = game.trace

    assert trace is not None
    assert trace.completed
    assert trace.outcome is RoundOutcome.PLAYER_WIN
    assert trace.settlement is not None
    assert trace.showdown is not None

    assert len(trace.decisions) == 1
    assert trace.decisions[0].action is Action.BET_4X

    assert not hasattr(
        trace.decisions[0].observation,
        "dealer_cards",
    )
    assert trace.state.dealer_cards == game.state.dealer_cards

def test_illegal_action_is_not_recorded() -> None:
    game = UTHGame(record_history=True)
    game.reset(card_source=FixedDeck(ROUND_DRAW_ORDER))

    with pytest.raises(IllegalActionError):
        game.step(Action.BET_2X)

    trace = game.trace

    assert trace is not None
    assert trace.decisions == ()

def test_reset_starts_new_trace() -> None:
    game = UTHGame(record_history=True)

    game.reset(card_source=FixedDeck(ROUND_DRAW_ORDER))
    game.step(Action.CHECK)

    first_trace = game.trace

    assert first_trace is not None
    assert first_trace.round_id == 1
    assert len(first_trace.decisions) == 1

    game.reset(card_source=FixedDeck(ROUND_DRAW_ORDER))

    second_trace = game.trace

    assert second_trace is not None
    assert second_trace.round_id == 2
    assert second_trace.decisions == ()

@pytest.mark.parametrize(
    "record_history",
    [
        1,
        "true",
        None,
    ],
)
def test_game_rejects_invalid_record_history_type(
    record_history: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="record_history must be a boolean",
    ):
        UTHGame(
            record_history=record_history,  # type: ignore[arg-type]
        )
