from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.state_representation import (
    FEATURE_STATE_SIZE,
    RAW_STATE_SIZE,
    FeatureStateEncoder,
    RawStateEncoder,
)
from expert_poker_player.uth import (
    Action,
    FixedDeck,
    GamePhase,
    UTHGame,
)


def build_fixed_deck() -> FixedDeck:
    return FixedDeck(
        [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.CLUBS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.EIGHT, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.FOUR, Suit.CLUBS),
        ]
    )


def test_encodes_real_observation_pipeline() -> None:
    game = UTHGame()

    raw_encoder = RawStateEncoder()
    feature_encoder = FeatureStateEncoder()

    preflop = game.reset(
        card_source=build_fixed_deck()
    )

    assert preflop.phase is GamePhase.PREFLOP

    raw_preflop = raw_encoder.encode(preflop)
    feature_preflop = feature_encoder.encode(
        preflop
    )

    assert len(raw_preflop) == RAW_STATE_SIZE
    assert (
        len(feature_preflop)
        == FEATURE_STATE_SIZE
    )

    flop_result = game.step(
        Action.CHECK
    )

    flop = flop_result.observation

    assert flop.phase is GamePhase.FLOP

    raw_flop = raw_encoder.encode(flop)
    feature_flop = feature_encoder.encode(
        flop
    )

    assert len(raw_flop) == RAW_STATE_SIZE
    assert (
        len(feature_flop)
        == FEATURE_STATE_SIZE
    )

    river_result = game.step(
        Action.CHECK
    )

    river = river_result.observation

    assert river.phase is GamePhase.RIVER

    raw_river = raw_encoder.encode(river)
    feature_river = feature_encoder.encode(
        river
    )

    assert len(raw_river) == RAW_STATE_SIZE
    assert (
        len(feature_river)
        == FEATURE_STATE_SIZE
    )

    assert raw_preflop != raw_flop
    assert raw_flop != raw_river

    assert feature_preflop != feature_flop
    assert feature_flop != feature_river

def build_alternative_hidden_deck() -> FixedDeck:
    return FixedDeck(
        [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.SIX, Suit.SPADES),
            Card(Rank.SEVEN, Suit.DIAMONDS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.EIGHT, Suit.CLUBS),
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.FOUR, Suit.CLUBS),
        ]
    )
def test_encoding_does_not_depend_on_hidden_cards() -> None:
    first_game = UTHGame()
    second_game = UTHGame()

    first_preflop = first_game.reset(
        card_source=build_fixed_deck()
    )

    second_preflop = second_game.reset(
        card_source=build_alternative_hidden_deck()
    )

    encoders = (
        RawStateEncoder(),
        FeatureStateEncoder(),
    )

    for encoder in encoders:
        assert encoder.encode(
            first_preflop
        ) == encoder.encode(
            second_preflop
        )

    first_flop = first_game.step(
        Action.CHECK
    ).observation

    second_flop = second_game.step(
        Action.CHECK
    ).observation

    for encoder in encoders:
        assert encoder.encode(
            first_flop
        ) == encoder.encode(
            second_flop
        )

    first_river = first_game.step(
        Action.CHECK
    ).observation

    second_river = second_game.step(
        Action.CHECK
    ).observation

    for encoder in encoders:
        assert encoder.encode(
            first_river
        ) == encoder.encode(
            second_river
        )

def test_encoding_pipeline_is_reproducible() -> None:
    first_game = UTHGame()
    second_game = UTHGame()

    first_observation = first_game.reset(
        card_source=build_fixed_deck()
    )

    second_observation = second_game.reset(
        card_source=build_fixed_deck()
    )

    raw_encoder = RawStateEncoder()
    feature_encoder = FeatureStateEncoder()

    assert raw_encoder.encode(
        first_observation
    ) == raw_encoder.encode(
        second_observation
    )

    assert feature_encoder.encode(
        first_observation
    ) == feature_encoder.encode(
        second_observation
    )

