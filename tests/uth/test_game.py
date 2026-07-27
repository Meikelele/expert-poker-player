import pytest

from expert_poker_player.cards import Card, Rank, Suit
from expert_poker_player.uth import (
    Action,
    FixedDeck,
    GamePhase,
    RoundNotStartedError,
    UTHGame,
)


INITIAL_DRAW_ORDER = (
    Card(rank=Rank.ACE, suit=Suit.SPADES),
    Card(rank=Rank.KING, suit=Suit.HEARTS),
    Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
    Card(rank=Rank.JACK, suit=Suit.CLUBS),
)


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