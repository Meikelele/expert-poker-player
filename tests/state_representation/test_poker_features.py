import pytest

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.hands import HandRank
from expert_poker_player.state_representation import (
    HAND_RANK_FEATURE_COUNT,
    POKER_FEATURE_COUNT,
    extract_poker_features,
)
from expert_poker_player.uth import (
    GamePhase,
    UTHObservation,
    legal_actions_for_phase,
)


def make_observation(
    *,
    phase: GamePhase,
    player_cards: tuple[Card, Card],
    community_cards: tuple[Card, ...],
) -> UTHObservation:
    return UTHObservation(
        phase=phase,
        player_cards=player_cards,
        community_cards=community_cards,
        legal_actions=legal_actions_for_phase(
            phase
        ),
    )


def test_poker_feature_count_is_fixed() -> None:
    assert POKER_FEATURE_COUNT == 20
    assert HAND_RANK_FEATURE_COUNT == 9

def test_extracts_preflop_hole_card_features() -> None:
    observation = make_observation(
        phase=GamePhase.PREFLOP,
        player_cards=(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES),
        ),
        community_cards=(),
    )

    features = extract_poker_features(
        observation
    )

    assert len(features) == POKER_FEATURE_COUNT

    assert features[0] == 1.0
    assert features[1] == pytest.approx(
        11 / 12
    )
    assert features[2] == 0.0
    assert features[3] == 1.0
    assert features[4] == pytest.approx(
        1 / 12
    )

    assert features[
        5:
        5 + HAND_RANK_FEATURE_COUNT
    ] == (
        0.0,
    ) * HAND_RANK_FEATURE_COUNT

def test_detects_pocket_pair() -> None:
    observation = make_observation(
        phase=GamePhase.PREFLOP,
        player_cards=(
            Card(Rank.QUEEN, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.HEARTS),
        ),
        community_cards=(),
    )

    features = extract_poker_features(
        observation
    )

    assert features[2] == 1.0
    assert features[4] == 0.0

def test_encodes_flop_hand_rank() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
        ),
        community_cards=(
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.KING, Suit.DIAMONDS),
        ),
    )

    features = extract_poker_features(
        observation
    )

    hand_rank = features[
        5:
        5 + HAND_RANK_FEATURE_COUNT
    ]

    assert hand_rank[
        HandRank.FULL_HOUSE.value
    ] == 1.0

    assert sum(hand_rank) == 1.0

def test_encodes_best_river_hand_rank() -> None:
    observation = make_observation(
        phase=GamePhase.RIVER,
        player_cards=(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES),
        ),
        community_cards=(
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.JACK, Suit.SPADES),
            Card(Rank.TEN, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
        ),
    )

    features = extract_poker_features(
        observation
    )

    hand_rank = features[
        5:
        5 + HAND_RANK_FEATURE_COUNT
    ]

    assert hand_rank[
        HandRank.STRAIGHT_FLUSH.value
    ] == 1.0

def test_structure_features_are_normalized() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
        ),
        community_cards=(
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TWO, Suit.CLUBS),
        ),
    )

    features = extract_poker_features(
        observation
    )

    structure_start = (
        5
        + HAND_RANK_FEATURE_COUNT
    )

    structure = features[
        structure_start:
    ]

    assert len(structure) == 6
    assert all(
        0.0 <= value <= 1.0
        for value in structure
    )

def test_detects_paired_board() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
        ),
        community_cards=(
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.TWO, Suit.DIAMONDS),
            Card(Rank.FIVE, Suit.HEARTS),
        ),
    )

    features = extract_poker_features(
        observation
    )

    assert features[-1] == 1.0

def test_rejects_terminal_observation() -> None:
    observation = UTHObservation(
        phase=GamePhase.TERMINAL,
        player_cards=(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
        ),
        community_cards=(
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
        ),
        legal_actions=legal_actions_for_phase(
            GamePhase.TERMINAL
        ),
        play_multiplier=4,
    )

    with pytest.raises(
        ValueError,
        match="cannot extract features",
    ):
        extract_poker_features(observation)


def test_rejects_invalid_observation() -> None:
    with pytest.raises(
        TypeError,
        match="observation must be an instance",
    ):
        extract_poker_features(
            "invalid"  # type: ignore[arg-type]
        )

def test_extracts_exact_draw_progress() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        player_cards=(
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
        ),
        community_cards=(
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TWO, Suit.CLUBS),
        ),
    )

    features = extract_poker_features(
        observation
    )

    structure_start = (
        5
        + HAND_RANK_FEATURE_COUNT
    )

    structure = features[
        structure_start:
    ]

    flush_progress = structure[3]
    straight_progress = structure[4]

    assert flush_progress == pytest.approx(
        4 / 5
    )

    assert straight_progress == pytest.approx(
        4 / 5
    )


    