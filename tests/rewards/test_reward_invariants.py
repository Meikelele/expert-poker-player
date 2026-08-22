import pytest

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.rewards import (
    NetProfitReward,
    StakeScaledNetProfitReward,
)
from expert_poker_player.uth import (
    Action,
    FixedDeck,
    StepResult,
    UTHGame,
)


def player_win_deck() -> FixedDeck:
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


def terminal_preflop_result(
    deck: FixedDeck,
) -> StepResult:
    game = UTHGame()

    game.reset(
        card_source=deck
    )

    return game.step(
        Action.BET_4X
    )


def fold_result() -> StepResult:
    game = UTHGame()

    game.reset(
        card_source=dealer_win_deck()
    )

    game.step(Action.CHECK)
    game.step(Action.CHECK)

    return game.step(
        Action.FOLD
    )

@pytest.mark.parametrize(
    "result_factory",
    [
        lambda: terminal_preflop_result(
            player_win_deck()
        ),
        lambda: terminal_preflop_result(
            dealer_win_deck()
        ),
        lambda: terminal_preflop_result(
            push_deck()
        ),
        fold_result,
    ],
)

def test_scaled_reward_is_linear_transform_of_net_profit(
    result_factory,
) -> None:
    result = result_factory()

    raw_reward = NetProfitReward().calculate_reward(
        result
    )

    scaled_reward = (
        StakeScaledNetProfitReward()
        .calculate_reward(result)
    )

    assert scaled_reward == pytest.approx(
        raw_reward / 6
    )

def player_win_result() -> StepResult:
    return terminal_preflop_result(
        player_win_deck()
    )


def dealer_win_result() -> StepResult:
    return terminal_preflop_result(
        dealer_win_deck()
    )


def push_result() -> StepResult:
    return terminal_preflop_result(
        push_deck()
    )

@pytest.mark.parametrize(
    "result_factory",
    [
        player_win_result,
        dealer_win_result,
        push_result,
        fold_result,
    ],
)

def test_scaling_preserves_reward_sign(
    result_factory,
) -> None:
    result = result_factory()

    raw = NetProfitReward().calculate_reward(
        result
    )

    scaled = StakeScaledNetProfitReward().calculate_reward(
        result
    )

    assert (raw > 0) == (scaled > 0)
    assert (raw < 0) == (scaled < 0)

def test_scaling_preserves_zero_reward() -> None:
    result = push_result()

    raw = NetProfitReward().calculate_reward(
        result
    )

    scaled = (
        StakeScaledNetProfitReward()
        .calculate_reward(result)
    )

    assert raw == 0.0
    assert scaled == 0.0

def test_scaling_preserves_reward_ordering() -> None:
    results = (
        dealer_win_result(),
        fold_result(),
        push_result(),
        player_win_result(),
    )

    raw_reward = NetProfitReward()
    scaled_reward = StakeScaledNetProfitReward()

    raw_values = [
        raw_reward.calculate_reward(result)
        for result in results
    ]

    scaled_values = [
        scaled_reward.calculate_reward(result)
        for result in results
    ]

    assert raw_values == sorted(raw_values)

    assert scaled_values == sorted(
        scaled_values
    )

def test_scaling_preserves_pairwise_ordering() -> None:
    results = (
        dealer_win_result(),
        fold_result(),
        push_result(),
        player_win_result(),
    )

    raw_function = NetProfitReward()
    scaled_function = (
        StakeScaledNetProfitReward()
    )

    raw_values = [
        raw_function.calculate_reward(result)
        for result in results
    ]

    scaled_values = [
        scaled_function.calculate_reward(
            result
        )
        for result in results
    ]

    for left_index in range(
        len(results)
    ):
        for right_index in range(
            len(results)
        ):
            assert (
                raw_values[left_index]
                < raw_values[right_index]
            ) == (
                scaled_values[left_index]
                < scaled_values[right_index]
            )

        