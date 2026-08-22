import pytest

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.rewards import (
    RewardType,
    build_reward_function,
)
from expert_poker_player.uth import (
    Action,
    FixedDeck,
    UTHGame,
)


def build_deck() -> FixedDeck:
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

@pytest.mark.parametrize(
    "reward_type",
    [
        RewardType.NET_PROFIT,
        RewardType.STAKE_SCALED_NET_PROFIT,
    ],
)
def test_reward_is_zero_until_terminal_step(
    reward_type: RewardType,
) -> None:
    game = UTHGame()

    reward_function = build_reward_function(
        reward_type
    )

    game.reset(
        card_source=build_deck()
    )

    flop_result = game.step(
        Action.CHECK
    )

    assert flop_result.terminated is False

    assert reward_function.calculate_reward(
        flop_result
    ) == 0.0

    river_result = game.step(
        Action.CHECK
    )

    assert river_result.terminated is False

    assert reward_function.calculate_reward(
        river_result
    ) == 0.0

    terminal_result = game.step(
        Action.BET_1X
    )

    assert terminal_result.terminated is True
    assert terminal_result.settlement is not None

    reward = reward_function.calculate_reward(
        terminal_result
    )

    assert reward != 0.0

def test_reward_variants_use_same_terminal_settlement() -> None:
    game = UTHGame()

    game.reset(
        card_source=build_deck()
    )

    game.step(Action.CHECK)
    game.step(Action.CHECK)

    result = game.step(
        Action.BET_1X
    )

    assert result.terminated is True
    assert result.settlement is not None

    net_profit_reward = build_reward_function(
        RewardType.NET_PROFIT
    ).calculate_reward(
        result
    )

    scaled_reward = build_reward_function(
        RewardType.STAKE_SCALED_NET_PROFIT
    ).calculate_reward(
        result
    )

    expected_net_profit = float(
        result.settlement.total_net_profit
    )

    assert net_profit_reward == expected_net_profit

    assert scaled_reward == pytest.approx(
        expected_net_profit / 6
    )

@pytest.mark.parametrize(
    (
        "reward_type",
        "expected_reward",
    ),
    [
        (
            RewardType.NET_PROFIT,
            -2.0,
        ),
        (
            RewardType.STAKE_SCALED_NET_PROFIT,
            -2 / 6,
        ),
    ],
)
def test_fold_produces_terminal_reward(
    reward_type: RewardType,
    expected_reward: float,
) -> None:
    game = UTHGame()

    reward_function = build_reward_function(
        reward_type
    )

    game.reset(
        card_source=build_deck()
    )

    game.step(Action.CHECK)
    game.step(Action.CHECK)

    result = game.step(
        Action.FOLD
    )

    assert result.terminated is True
    assert result.settlement is not None

    reward = reward_function.calculate_reward(
        result
    )

    assert reward == pytest.approx(
        expected_reward
    )

def test_reward_pipeline_uses_step_result_only() -> None:
    game = UTHGame()

    game.reset(
        card_source=build_deck()
    )

    result = game.step(
        Action.BET_4X
    )

    reward_function = build_reward_function(
        RewardType.NET_PROFIT
    )

    reward = reward_function.calculate_reward(
        result
    )

    assert result.settlement is not None

    assert reward == float(
        result.settlement.total_net_profit
    )

