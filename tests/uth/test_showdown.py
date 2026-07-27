import pytest

from expert_poker_player.cards import Card, Rank, Suit
from expert_poker_player.hands import evaluate_best_hand
from expert_poker_player.uth import RoundOutcome
from expert_poker_player.uth.showdown import ShowdownResult
from expert_poker_player.hands import EvaluatedHand

BOARD = (
    Card(rank=Rank.TWO, suit=Suit.CLUBS),
    Card(rank=Rank.FIVE, suit=Suit.DIAMONDS),
    Card(rank=Rank.NINE, suit=Suit.SPADES),
    Card(rank=Rank.JACK, suit=Suit.CLUBS),
    Card(rank=Rank.KING, suit=Suit.DIAMONDS),
)

PAIR_OF_ACES_CARDS = (
    Card(rank=Rank.ACE, suit=Suit.SPADES),
    Card(rank=Rank.ACE, suit=Suit.HEARTS),
)

PAIR_OF_KINGS_CARDS = (
    Card(rank=Rank.KING, suit=Suit.SPADES),
    Card(rank=Rank.QUEEN, suit=Suit.HEARTS),
)

HIGH_CARD_CARDS = (
    Card(rank=Rank.QUEEN, suit=Suit.SPADES),
    Card(rank=Rank.EIGHT, suit=Suit.HEARTS),
)


def evaluated_hand(
    hole_cards: tuple[Card, Card],
) -> EvaluatedHand:
    return evaluate_best_hand(
        (
            *hole_cards,
            *BOARD,
        )
    )


def test_showdown_reports_player_win() -> None:
    result = ShowdownResult(
        player_hand=evaluated_hand(PAIR_OF_ACES_CARDS),
        dealer_hand=evaluated_hand(PAIR_OF_KINGS_CARDS),
    )

    assert result.outcome is RoundOutcome.PLAYER_WIN
    assert result.dealer_qualified


def test_showdown_reports_dealer_win() -> None:
    result = ShowdownResult(
        player_hand=evaluated_hand(PAIR_OF_KINGS_CARDS),
        dealer_hand=evaluated_hand(PAIR_OF_ACES_CARDS),
    )

    assert result.outcome is RoundOutcome.DEALER_WIN
    assert result.dealer_qualified


def test_showdown_reports_unqualified_dealer() -> None:
    result = ShowdownResult(
        player_hand=evaluated_hand(PAIR_OF_ACES_CARDS),
        dealer_hand=evaluated_hand(HIGH_CARD_CARDS),
    )

    assert result.outcome is RoundOutcome.PLAYER_WIN
    assert not result.dealer_qualified


def test_showdown_reports_push() -> None:
    shared_board = (
        Card(rank=Rank.ACE, suit=Suit.SPADES),
        Card(rank=Rank.KING, suit=Suit.HEARTS),
        Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
        Card(rank=Rank.JACK, suit=Suit.CLUBS),
        Card(rank=Rank.TEN, suit=Suit.SPADES),
    )

    player_hand = evaluate_best_hand(
        (
            Card(rank=Rank.TWO, suit=Suit.HEARTS),
            Card(rank=Rank.THREE, suit=Suit.HEARTS),
            *shared_board,
        )
    )

    dealer_hand = evaluate_best_hand(
        (
            Card(rank=Rank.FOUR, suit=Suit.CLUBS),
            Card(rank=Rank.FIVE, suit=Suit.CLUBS),
            *shared_board,
        )
    )

    result = ShowdownResult(
        player_hand=player_hand,
        dealer_hand=dealer_hand,
    )

    assert result.outcome is RoundOutcome.PUSH
    assert result.dealer_qualified


def test_showdown_rejects_invalid_player_hand() -> None:
    valid_hand = evaluated_hand(PAIR_OF_ACES_CARDS)

    with pytest.raises(
        TypeError,
        match="player_hand must be an instance of EvaluatedHand",
    ):
        ShowdownResult(
            player_hand="invalid",  # type: ignore[arg-type]
            dealer_hand=valid_hand,
        )


def test_showdown_rejects_invalid_dealer_hand() -> None:
    valid_hand = evaluated_hand(PAIR_OF_ACES_CARDS)

    with pytest.raises(
        TypeError,
        match="dealer_hand must be an instance of EvaluatedHand",
    ):
        ShowdownResult(
            player_hand=valid_hand,
            dealer_hand="invalid",  # type: ignore[arg-type]
        )