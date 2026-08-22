import pytest

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.state_representation import (
    FEATURE_STATE_SIZE,
    POKER_FEATURE_COUNT,
    RAW_STATE_SIZE,
    FeatureStateEncoder,
    RawStateEncoder,
    StateEncoder,
    extract_poker_features,
)
from expert_poker_player.uth import (
    GamePhase,
    UTHObservation,
    legal_actions_for_phase,
)


def make_flop_observation() -> UTHObservation:
    return UTHObservation(
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
        legal_actions=legal_actions_for_phase(
            GamePhase.FLOP
        ),
    )


def test_feature_encoder_satisfies_protocol() -> None:
    encoder = FeatureStateEncoder()

    assert isinstance(
        encoder,
        StateEncoder,
    )


def test_feature_encoder_has_fixed_output_size() -> None:
    encoder = FeatureStateEncoder()

    assert RAW_STATE_SIZE == 373
    assert POKER_FEATURE_COUNT == 20
    assert FEATURE_STATE_SIZE == 393
    assert encoder.output_size == 393


def test_feature_encoder_extends_raw_representation() -> None:
    observation = make_flop_observation()

    raw = RawStateEncoder().encode(
        observation
    )

    feature = FeatureStateEncoder().encode(
        observation
    )

    assert feature[:RAW_STATE_SIZE] == raw


def test_feature_encoder_appends_poker_features() -> None:
    observation = make_flop_observation()

    expected = extract_poker_features(
        observation
    )

    encoded = FeatureStateEncoder().encode(
        observation
    )

    assert encoded[RAW_STATE_SIZE:] == expected


def test_feature_encoder_is_deterministic() -> None:
    observation = make_flop_observation()
    encoder = FeatureStateEncoder()

    first = encoder.encode(observation)
    second = encoder.encode(observation)

    assert first == second

def test_feature_encoder_rejects_terminal_observation() -> None:
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
        match="cannot encode a terminal observation",
    ):
        FeatureStateEncoder().encode(
            observation
        )

