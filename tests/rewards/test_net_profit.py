import pytest

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.rewards import (
    NetProfitReward,
)
from expert_poker_player.uth import (
    Action,
    FixedDeck,
    StepResult,
    UTHGame,
)


def player_flush_win_deck() -> FixedDeck:
    return FixedDeck(
        [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.CLUBS),
            Card(Rank.TWO, Suit.DIAMONDS),
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.EIGHT, Suit.SPADES),
            Card(Rank.JACK, Suit.DIAMONDS),
            Card(Rank.THREE, Suit.SPADES),
        ]
    )


def dealer_win_deck() -> FixedDeck:
    return FixedDeck(
        [
            Card(Rank.TWO, Suit.SPADES),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.TEN, Suit.SPADES),
            Card(Rank.JACK, Suit.CLUBS),
            Card(Rank.KING, Suit.DIAMONDS),
        ]
    )


def push_deck() -> FixedDeck:
    return FixedDeck(
        [
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.CLUBS),
            Card(Rank.FOUR, Suit.DIAMONDS),
            Card(Rank.FIVE, Suit.DIAMONDS),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.TEN, Suit.SPADES),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
            Card(Rank.SEVEN, Suit.CLUBS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.ACE, Suit.SPADES),
        ]
    )


def play_preflop_bet(
    deck: FixedDeck,
) -> StepResult:
    game = UTHGame()

    game.reset(
        card_source=deck
    )

    return game.step(
        Action.BET_4X
    )

def test_non_terminal_step_returns_zero_reward() -> None:
    game = UTHGame()

    game.reset(
        card_source=dealer_win_deck()
    )

    result = game.step(
        Action.CHECK
    )

    reward = NetProfitReward().calculate_reward(
        result
    )

    assert result.terminated is False
    assert reward == 0.0


def test_player_win_returns_net_profit() -> None:
    result = play_preflop_bet(
        player_flush_win_deck()
    )

    reward = NetProfitReward().calculate_reward(
        result
    )

    assert result.settlement is not None

    assert reward == float(
        result.settlement.total_net_profit
    )

    assert reward == 5.5


def test_dealer_win_returns_negative_net_profit() -> None:
    result = play_preflop_bet(
        dealer_win_deck()
    )

    reward = NetProfitReward().calculate_reward(
        result
    )

    assert reward == -6.0


def test_push_returns_zero_reward() -> None:
    result = play_preflop_bet(
        push_deck()
    )

    reward = NetProfitReward().calculate_reward(
        result
    )

    assert reward == 0.0

def test_fold_returns_fold_net_profit() -> None:
    game = UTHGame()

    game.reset(
        card_source=dealer_win_deck()
    )

    game.step(
        Action.CHECK
    )

    game.step(
        Action.CHECK
    )

    result = game.step(
        Action.FOLD
    )

    reward = NetProfitReward().calculate_reward(
        result
    )

    assert reward == -2.0

def test_rejects_invalid_step_result() -> None:
    with pytest.raises(
        TypeError,
        match="step_result must be an instance of StepResult",
    ):
        NetProfitReward().calculate_reward(
            "invalid"  # type: ignore[arg-type]
        )