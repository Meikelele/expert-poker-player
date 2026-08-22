from fractions import Fraction

import pytest

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.rewards import (
    MAX_TOTAL_STAKE,
    NetProfitReward,
    StakeScaledNetProfitReward,
)
from expert_poker_player.uth import (
    Action,
    FixedDeck,
    UTHGame,
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

def test_max_total_stake_is_six_ante_units() -> None:
    assert MAX_TOTAL_STAKE == Fraction(6)

def test_non_terminal_reward_remains_zero() -> None:
    game = UTHGame()

    game.reset(
        card_source=dealer_win_deck()
    )

    result = game.step(
        Action.CHECK
    )

    reward = (
        StakeScaledNetProfitReward()
        .calculate_reward(result)
    )

    assert reward == 0.0

def test_max_standard_loss_scales_to_minus_one() -> None:
    game = UTHGame()

    game.reset(
        card_source=dealer_win_deck()
    )

    result = game.step(
        Action.BET_4X
    )

    raw_reward = NetProfitReward().calculate_reward(
        result
    )

    scaled_reward = (
        StakeScaledNetProfitReward()
        .calculate_reward(result)
    )

    assert raw_reward == -6.0
    assert scaled_reward == -1.0

def test_fold_reward_is_scaled_by_six() -> None:
    game = UTHGame()

    game.reset(
        card_source=dealer_win_deck()
    )

    game.step(Action.CHECK)
    game.step(Action.CHECK)

    result = game.step(
        Action.FOLD
    )

    reward = (
        StakeScaledNetProfitReward()
        .calculate_reward(result)
    )

    assert reward == pytest.approx(
        -2 / 6
    )

def test_scaled_reward_is_net_profit_divided_by_six() -> None:
    game = UTHGame()

    game.reset(
        card_source=dealer_win_deck()
    )

    result = game.step(
        Action.BET_4X
    )

    net_profit = NetProfitReward().calculate_reward(
        result
    )

    scaled = (
        StakeScaledNetProfitReward()
        .calculate_reward(result)
    )

    assert scaled == pytest.approx(
        net_profit / 6
    )

def test_rejects_invalid_step_result() -> None:
    with pytest.raises(
        TypeError,
        match="step_result must be an instance of StepResult",
    ):
        StakeScaledNetProfitReward().calculate_reward(
            "invalid"  # type: ignore[arg-type]
        )

    