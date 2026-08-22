import pytest

from expert_poker_player.agents import (
    Agent,
    RuleBasedAgent,
)
from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.evaluation import (
    SimulationConfig,
    run_simulation,
)
from expert_poker_player.uth import (
    Action,
    GamePhase,
    UTHGame,
    UTHObservation,
    legal_actions_for_phase,
)


def make_card(
    rank: Rank,
    suit: Suit,
) -> Card:
    return Card(
        rank=rank,
        suit=suit,
    )


def make_observation(
    *,
    phase: GamePhase,
    player_cards: tuple[Card, Card],
    community_cards: tuple[Card, ...] = (),
) -> UTHObservation:
    return UTHObservation(
        phase=phase,
        player_cards=player_cards,
        community_cards=community_cards,
        legal_actions=legal_actions_for_phase(phase),
    )


def test_rule_based_agent_satisfies_agent_protocol() -> None:
    agent = RuleBasedAgent()

    assert isinstance(agent, Agent)


def test_preflop_raises_strong_hand() -> None:
    observation = make_observation(
        phase=GamePhase.PREFLOP,
        player_cards=(
            make_card(Rank.ACE, Suit.HEARTS),
            make_card(Rank.TWO, Suit.CLUBS),
        ),
    )

    agent = RuleBasedAgent()

    assert agent.select_action(observation) is Action.BET_4X


def test_preflop_checks_weak_hand() -> None:
    observation = make_observation(
        phase=GamePhase.PREFLOP,
        player_cards=(
            make_card(Rank.TEN, Suit.HEARTS),
            make_card(Rank.NINE, Suit.CLUBS),
        ),
    )

    agent = RuleBasedAgent()

    assert agent.select_action(observation) is Action.CHECK


def test_flop_raises_two_pair() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            make_card(Rank.NINE, Suit.HEARTS),
            make_card(Rank.EIGHT, Suit.SPADES),
        ),
        community_cards=(
            make_card(Rank.NINE, Suit.CLUBS),
            make_card(Rank.EIGHT, Suit.DIAMONDS),
            make_card(Rank.TWO, Suit.HEARTS),
        ),
    )

    agent = RuleBasedAgent()

    assert agent.select_action(observation) is Action.BET_2X


def test_flop_checks_weak_hand() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            make_card(Rank.NINE, Suit.HEARTS),
            make_card(Rank.THREE, Suit.CLUBS),
        ),
        community_cards=(
            make_card(Rank.KING, Suit.SPADES),
            make_card(Rank.SEVEN, Suit.DIAMONDS),
            make_card(Rank.TWO, Suit.HEARTS),
        ),
    )

    agent = RuleBasedAgent()

    assert agent.select_action(observation) is Action.CHECK


def test_river_bets_with_fewer_than_twenty_one_outs() -> None:
    observation = make_observation(
        phase=GamePhase.RIVER,
        player_cards=(
            make_card(Rank.QUEEN, Suit.HEARTS),
            make_card(Rank.SEVEN, Suit.CLUBS),
        ),
        community_cards=(
            make_card(Rank.KING, Suit.SPADES),
            make_card(Rank.NINE, Suit.HEARTS),
            make_card(Rank.EIGHT, Suit.DIAMONDS),
            make_card(Rank.FIVE, Suit.CLUBS),
            make_card(Rank.FOUR, Suit.SPADES),
        ),
    )

    agent = RuleBasedAgent()

    assert agent.select_action(observation) is Action.BET_1X


def test_river_folds_with_at_least_twenty_one_outs() -> None:
    observation = make_observation(
        phase=GamePhase.RIVER,
        player_cards=(
            make_card(Rank.NINE, Suit.HEARTS),
            make_card(Rank.THREE, Suit.CLUBS),
        ),
        community_cards=(
            make_card(Rank.KING, Suit.SPADES),
            make_card(Rank.SEVEN, Suit.HEARTS),
            make_card(Rank.TWO, Suit.DIAMONDS),
            make_card(Rank.ACE, Suit.CLUBS),
            make_card(Rank.TEN, Suit.SPADES),
        ),
    )

    agent = RuleBasedAgent()

    assert agent.select_action(observation) is Action.FOLD


def test_rejects_invalid_observation() -> None:
    agent = RuleBasedAgent()

    with pytest.raises(
        TypeError,
        match=(
            "observation must be an instance "
            "of UTHObservation"
        ),
    ):
        agent.select_action(
            object(),  # type: ignore[arg-type]
        )


def test_rejects_terminal_observation() -> None:
    game = UTHGame(seed=123)

    game.reset()
    game.step(Action.CHECK)
    game.step(Action.CHECK)

    terminal_observation = game.step(
        Action.FOLD
    ).observation

    agent = RuleBasedAgent()

    with pytest.raises(
        ValueError,
        match=(
            "cannot select an action for "
            "a terminal observation"
        ),
    ):
        agent.select_action(terminal_observation)


def test_agent_completes_reproducible_simulation() -> None:
    config = SimulationConfig(
        deck_seeds=(
            101,
            202,
            303,
            404,
            505,
        ),
    )

    first_result = run_simulation(
        agent=RuleBasedAgent(),
        config=config,
    )
    second_result = run_simulation(
        agent=RuleBasedAgent(),
        config=config,
    )

    assert first_result == second_result

    assert all(
        episode.actions
        for episode in first_result.episodes
    )

    assert all(
        action in {
            Action.CHECK,
            Action.BET_4X,
            Action.BET_2X,
            Action.BET_1X,
            Action.FOLD,
        }
        for episode in first_result.episodes
        for action in episode.actions
    )