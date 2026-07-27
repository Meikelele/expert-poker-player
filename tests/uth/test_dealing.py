import pytest

from expert_poker_player.cards import Card, Deck, Rank, Suit
from expert_poker_player.uth import (
    FixedDeck,
    GamePhase,
    InvalidPhaseTransitionError,
)
from expert_poker_player.uth.dealing import (
    deal_initial_cards,
    reveal_flop,
    reveal_turn_and_river,
)


DRAW_ORDER = (
    # Karty własne: P1, D1, P2, D2
    Card(rank=Rank.ACE, suit=Suit.SPADES),
    Card(rank=Rank.KING, suit=Suit.HEARTS),
    Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
    Card(rank=Rank.JACK, suit=Suit.CLUBS),

    # Pierwsza spalona karta
    Card(rank=Rank.TEN, suit=Suit.SPADES),

    # Flop
    Card(rank=Rank.NINE, suit=Suit.HEARTS),
    Card(rank=Rank.EIGHT, suit=Suit.DIAMONDS),
    Card(rank=Rank.SEVEN, suit=Suit.CLUBS),

    # Druga spalona karta
    Card(rank=Rank.SIX, suit=Suit.SPADES),

    # Turn i river
    Card(rank=Rank.FIVE, suit=Suit.HEARTS),
    Card(rank=Rank.FOUR, suit=Suit.DIAMONDS),
)


def test_deals_complete_round_in_expected_order() -> None:
    deck = FixedDeck(DRAW_ORDER)

    preflop_state = deal_initial_cards(deck)

    assert preflop_state.phase is GamePhase.PREFLOP
    assert preflop_state.player_cards == (
        DRAW_ORDER[0],
        DRAW_ORDER[2],
    )
    assert preflop_state.dealer_cards == (
        DRAW_ORDER[1],
        DRAW_ORDER[3],
    )
    assert preflop_state.community_cards == ()
    assert preflop_state.burned_cards == ()
    assert len(deck) == 7

    flop_state = reveal_flop(
        preflop_state,
        deck,
    )

    assert flop_state.phase is GamePhase.FLOP
    assert flop_state.player_cards == preflop_state.player_cards
    assert flop_state.dealer_cards == preflop_state.dealer_cards
    assert flop_state.burned_cards == (
        DRAW_ORDER[4],
    )
    assert flop_state.community_cards == (
        DRAW_ORDER[5],
        DRAW_ORDER[6],
        DRAW_ORDER[7],
    )
    assert len(deck) == 3

    river_state = reveal_turn_and_river(
        flop_state,
        deck,
    )

    assert river_state.phase is GamePhase.RIVER
    assert river_state.player_cards == preflop_state.player_cards
    assert river_state.dealer_cards == preflop_state.dealer_cards
    assert river_state.burned_cards == (
        DRAW_ORDER[4],
        DRAW_ORDER[8],
    )
    assert river_state.community_cards == (
        DRAW_ORDER[5],
        DRAW_ORDER[6],
        DRAW_ORDER[7],
        DRAW_ORDER[9],
        DRAW_ORDER[10],
    )
    assert len(deck) == 0


def test_same_seed_produces_same_initial_deal() -> None:
    first_state = deal_initial_cards(
        Deck(seed=42)
    )
    second_state = deal_initial_cards(
        Deck(seed=42)
    )

    assert first_state == second_state


def test_deal_initial_cards_rejects_invalid_card_source() -> None:
    with pytest.raises(
        TypeError,
        match="card_source must implement CardSource",
    ):
        deal_initial_cards(object())  # type: ignore[arg-type]


def test_reveal_flop_rejects_invalid_state_type() -> None:
    with pytest.raises(
        TypeError,
        match="state must be an instance of RoundState",
    ):
        reveal_flop(
            "not a state",  # type: ignore[arg-type]
            FixedDeck(DRAW_ORDER),
        )


def test_reveal_flop_requires_preflop_phase() -> None:
    deck = FixedDeck(DRAW_ORDER)

    preflop_state = deal_initial_cards(deck)
    flop_state = reveal_flop(preflop_state, deck)

    with pytest.raises(
        InvalidPhaseTransitionError,
        match="expected phase preflop",
    ):
        reveal_flop(flop_state, deck)


def test_reveal_turn_and_river_requires_flop_phase() -> None:
    deck = FixedDeck(DRAW_ORDER)
    preflop_state = deal_initial_cards(deck)

    with pytest.raises(
        InvalidPhaseTransitionError,
        match="expected phase flop",
    ):
        reveal_turn_and_river(
            preflop_state,
            deck,
        )


def test_dealing_rejects_source_returning_non_card() -> None:
    class InvalidCardSource:
        def draw(self) -> Card:
            return "not a card"  # type: ignore[return-value]

    with pytest.raises(
        TypeError,
        match="card source must return an instance of Card",
    ):
        deal_initial_cards(InvalidCardSource())