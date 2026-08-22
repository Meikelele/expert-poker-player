import pytest

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.state_representation import (
    CARD_COUNT,
    RAW_STATE_SIZE,
    RawStateEncoder,
    StateEncoder,
    encode_card,
)
from expert_poker_player.uth import (
    Action,
    GamePhase,
    UTHObservation,
    legal_actions_for_phase,
)


def make_observation(
    *,
    phase: GamePhase,
    community_cards: tuple[Card, ...],
) -> UTHObservation:
    return UTHObservation(
        phase=phase,
        player_cards=(
            Card(
                rank=Rank.ACE,
                suit=Suit.SPADES,
            ),
            Card(
                rank=Rank.KING,
                suit=Suit.HEARTS,
            ),
        ),
        community_cards=community_cards,
        legal_actions=legal_actions_for_phase(
            phase
        ),
    )

def test_raw_encoder_satisfies_protocol() -> None:
    encoder = RawStateEncoder()

    assert isinstance(
        encoder,
        StateEncoder,
    )


def test_raw_encoder_has_fixed_output_size() -> None:
    encoder = RawStateEncoder()

    assert encoder.output_size == 373
    assert RAW_STATE_SIZE == 373

def test_encodes_preflop_observation() -> None:
    observation = make_observation(
        phase=GamePhase.PREFLOP,
        community_cards=(),
    )

    encoded = RawStateEncoder().encode(
        observation
    )

    assert len(encoded) == RAW_STATE_SIZE

    assert encoded[:CARD_COUNT] == encode_card(
        observation.player_cards[0]
    )

    assert encoded[
        CARD_COUNT:2 * CARD_COUNT
    ] == encode_card(
        observation.player_cards[1]
    )

    board_start = 2 * CARD_COUNT
    board_end = 7 * CARD_COUNT

    assert encoded[
        board_start:board_end
    ] == (
        0.0,
    ) * (5 * CARD_COUNT)

@pytest.mark.parametrize(
    ("phase", "expected_phase_vector"),
    [
        (
            GamePhase.PREFLOP,
            (1.0, 0.0, 0.0),
        ),
        (
            GamePhase.FLOP,
            (0.0, 1.0, 0.0),
        ),
        (
            GamePhase.RIVER,
            (0.0, 0.0, 1.0),
        ),
    ],
)
def test_encodes_phase(
    phase: GamePhase,
    expected_phase_vector: tuple[
        float,
        float,
        float,
    ],
) -> None:
    community_cards_by_phase: dict[
        GamePhase, tuple[Card, ...]
    ] = {
        GamePhase.PREFLOP: (),
        GamePhase.FLOP: (
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
        ),
        GamePhase.RIVER: (
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
        ),
    }

    community_cards = community_cards_by_phase[phase]

    observation = make_observation(
        phase=phase,
        community_cards=community_cards,
    )

    encoded = RawStateEncoder().encode(
        observation
    )

    phase_start = 7 * CARD_COUNT
    phase_end = phase_start + 3

    assert encoded[
        phase_start:phase_end
    ] == expected_phase_vector

def test_encodes_legal_action_mask() -> None:
    observation = make_observation(
        phase=GamePhase.FLOP,
        community_cards=(
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
        ),
    )

    encoded = RawStateEncoder().encode(
        observation
    )

    action_mask = encoded[-len(Action):]

    assert action_mask == tuple(
        1.0
        if action in observation.legal_actions
        else 0.0
        for action in Action
    )

def test_flop_uses_zero_vectors_for_unrevealed_cards() -> None:
    flop = (
        Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.FOUR, Suit.HEARTS),
    )

    observation = make_observation(
        phase=GamePhase.FLOP,
        community_cards=flop,
    )

    encoded = RawStateEncoder().encode(
        observation
    )

    board_start = 2 * CARD_COUNT

    expected_board = (
        *encode_card(flop[0]),
        *encode_card(flop[1]),
        *encode_card(flop[2]),
        *((0.0,) * CARD_COUNT),
        *((0.0,) * CARD_COUNT),
    )

    assert encoded[
        board_start:
        board_start + 5 * CARD_COUNT
    ] == expected_board

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
        match="cannot encode a terminal observation",
    ):
        RawStateEncoder().encode(observation)

def test_rejects_invalid_observation() -> None:
    with pytest.raises(
        TypeError,
        match="observation must be an instance",
    ):
        RawStateEncoder().encode(
            "invalid"  # type: ignore[arg-type]
        )
