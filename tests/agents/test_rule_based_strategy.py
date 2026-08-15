import pytest

from expert_poker_player.agents.rule_based_strategy import (
    count_dealer_outs,
    should_raise_flop,
    should_raise_preflop,
    should_raise_river,
)
from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.uth import (
    GamePhase,
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


@pytest.mark.parametrize(
    (
        "player_cards",
        "expected",
    ),
    [
        (
            (
                make_card(Rank.THREE, Suit.HEARTS),
                make_card(Rank.THREE, Suit.SPADES),
            ),
            True,
        ),
        (
            (
                make_card(Rank.TWO, Suit.HEARTS),
                make_card(Rank.TWO, Suit.SPADES),
            ),
            False,
        ),
        (
            (
                make_card(Rank.ACE, Suit.HEARTS),
                make_card(Rank.TWO, Suit.CLUBS),
            ),
            True,
        ),
        (
            (
                make_card(Rank.KING, Suit.HEARTS),
                make_card(Rank.TWO, Suit.HEARTS),
            ),
            True,
        ),
        (
            (
                make_card(Rank.KING, Suit.HEARTS),
                make_card(Rank.FOUR, Suit.CLUBS),
            ),
            False,
        ),
        (
            (
                make_card(Rank.KING, Suit.HEARTS),
                make_card(Rank.FIVE, Suit.CLUBS),
            ),
            True,
        ),
        (
            (
                make_card(Rank.QUEEN, Suit.HEARTS),
                make_card(Rank.SIX, Suit.HEARTS),
            ),
            True,
        ),
        (
            (
                make_card(Rank.QUEEN, Suit.HEARTS),
                make_card(Rank.SEVEN, Suit.CLUBS),
            ),
            False,
        ),
        (
            (
                make_card(Rank.QUEEN, Suit.HEARTS),
                make_card(Rank.EIGHT, Suit.CLUBS),
            ),
            True,
        ),
        (
            (
                make_card(Rank.JACK, Suit.HEARTS),
                make_card(Rank.EIGHT, Suit.HEARTS),
            ),
            True,
        ),
        (
            (
                make_card(Rank.JACK, Suit.HEARTS),
                make_card(Rank.NINE, Suit.CLUBS),
            ),
            False,
        ),
        (
            (
                make_card(Rank.JACK, Suit.HEARTS),
                make_card(Rank.TEN, Suit.CLUBS),
            ),
            True,
        ),
        (
            (
                make_card(Rank.TEN, Suit.HEARTS),
                make_card(Rank.NINE, Suit.HEARTS),
            ),
            False,
        ),
    ],
)
def test_preflop_strategy(
    player_cards: tuple[Card, Card],
    expected: bool,
) -> None:
    observation = make_observation(
        phase=GamePhase.PREFLOP,
        player_cards=player_cards,
    )

    assert should_raise_preflop(observation) is expected


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

    assert should_raise_flop(observation)


def test_flop_raises_hidden_pair() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            make_card(Rank.KING, Suit.HEARTS),
            make_card(Rank.SEVEN, Suit.SPADES),
        ),
        community_cards=(
            make_card(Rank.KING, Suit.CLUBS),
            make_card(Rank.TWO, Suit.DIAMONDS),
            make_card(Rank.THREE, Suit.HEARTS),
        ),
    )

    assert should_raise_flop(observation)


def test_flop_does_not_raise_pocket_deuces_alone() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            make_card(Rank.TWO, Suit.HEARTS),
            make_card(Rank.TWO, Suit.SPADES),
        ),
        community_cards=(
            make_card(Rank.KING, Suit.CLUBS),
            make_card(Rank.SEVEN, Suit.DIAMONDS),
            make_card(Rank.FOUR, Suit.HEARTS),
        ),
    )

    assert not should_raise_flop(observation)


def test_flop_raises_four_to_flush_with_hidden_ten() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            make_card(Rank.TEN, Suit.HEARTS),
            make_card(Rank.THREE, Suit.CLUBS),
        ),
        community_cards=(
            make_card(Rank.TWO, Suit.HEARTS),
            make_card(Rank.SEVEN, Suit.HEARTS),
            make_card(Rank.KING, Suit.HEARTS),
        ),
    )

    assert should_raise_flop(observation)


def test_flop_rejects_four_to_flush_with_hidden_nine() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            make_card(Rank.NINE, Suit.HEARTS),
            make_card(Rank.THREE, Suit.CLUBS),
        ),
        community_cards=(
            make_card(Rank.TWO, Suit.HEARTS),
            make_card(Rank.SEVEN, Suit.HEARTS),
            make_card(Rank.KING, Suit.HEARTS),
        ),
    )

    assert not should_raise_flop(observation)


def test_preflop_strategy_rejects_wrong_phase() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            make_card(Rank.ACE, Suit.HEARTS),
            make_card(Rank.KING, Suit.SPADES),
        ),
        community_cards=(
            make_card(Rank.TWO, Suit.CLUBS),
            make_card(Rank.THREE, Suit.DIAMONDS),
            make_card(Rank.FOUR, Suit.HEARTS),
        ),
    )

    with pytest.raises(
        ValueError,
        match="expected preflop observation",
    ):
        should_raise_preflop(observation)


def test_flop_strategy_rejects_invalid_observation() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "observation must be an instance "
            "of UTHObservation"
        ),
    ):
        should_raise_flop(
            object(),  # type: ignore[arg-type]
        )

def test_river_raises_with_nineteen_dealer_outs() -> None:
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

    assert count_dealer_outs(observation) == 19
    assert should_raise_river(observation)

def test_river_folds_with_twenty_three_dealer_outs() -> None:
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

    assert count_dealer_outs(observation) == 23
    assert not should_raise_river(observation)

def test_river_raises_hidden_pair() -> None:
    observation = make_observation(
        phase=GamePhase.RIVER,
        player_cards=(
            make_card(Rank.KING, Suit.HEARTS),
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

    assert should_raise_river(observation)

def test_river_raises_pocket_deuces() -> None:
    observation = make_observation(
        phase=GamePhase.RIVER,
        player_cards=(
            make_card(Rank.TWO, Suit.HEARTS),
            make_card(Rank.TWO, Suit.CLUBS),
        ),
        community_cards=(
            make_card(Rank.KING, Suit.SPADES),
            make_card(Rank.SEVEN, Suit.HEARTS),
            make_card(Rank.FOUR, Suit.DIAMONDS),
            make_card(Rank.ACE, Suit.CLUBS),
            make_card(Rank.TEN, Suit.SPADES),
        ),
    )

    assert should_raise_river(observation)

def test_river_raises_two_pair() -> None:
    observation = make_observation(
        phase=GamePhase.RIVER,
        player_cards=(
            make_card(Rank.NINE, Suit.HEARTS),
            make_card(Rank.EIGHT, Suit.SPADES),
        ),
        community_cards=(
            make_card(Rank.NINE, Suit.CLUBS),
            make_card(Rank.EIGHT, Suit.DIAMONDS),
            make_card(Rank.TWO, Suit.HEARTS),
            make_card(Rank.KING, Suit.CLUBS),
            make_card(Rank.FOUR, Suit.SPADES),
        ),
    )

    assert should_raise_river(observation)

def test_river_strategy_rejects_wrong_phase() -> None:
    observation = make_observation(
        phase=GamePhase.PREFLOP,
        player_cards=(
            make_card(Rank.ACE, Suit.HEARTS),
            make_card(Rank.KING, Suit.SPADES),
        ),
    )

    with pytest.raises(
        ValueError,
        match="expected river observation",
    ):
        should_raise_river(observation)

