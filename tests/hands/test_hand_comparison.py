from expert_poker_player.cards import Card, Rank, Suit
from expert_poker_player.hands import (
    HandRank,
    HandValue,
    evaluate_best_hand,
)


def make_cards(
    *specifications: tuple[Rank, Suit],
) -> list[Card]:
    """Tworzy karty na podstawie par: wartość i kolor."""

    return [
        Card(rank=rank, suit=suit)
        for rank, suit in specifications
    ]


def evaluate_value(
    *specifications: tuple[Rank, Suit],
) -> HandValue:
    """Tworzy karty i zwraca wartość najlepszego układu."""

    cards = make_cards(*specifications)
    return evaluate_best_hand(cards).value


def test_flush_beats_straight_even_when_straight_has_higher_cards() -> None:
    low_flush = evaluate_value(
        (Rank.NINE, Suit.HEARTS),
        (Rank.SEVEN, Suit.HEARTS),
        (Rank.FIVE, Suit.HEARTS),
        (Rank.THREE, Suit.HEARTS),
        (Rank.TWO, Suit.HEARTS),
    )

    ace_high_straight = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.JACK, Suit.CLUBS),
        (Rank.TEN, Suit.SPADES),
    )

    assert low_flush.rank is HandRank.FLUSH
    assert ace_high_straight.rank is HandRank.STRAIGHT
    assert low_flush > ace_high_straight


def test_higher_pair_wins() -> None:
    pair_of_aces = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.FOUR, Suit.SPADES),
    )

    pair_of_kings = evaluate_value(
        (Rank.KING, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.CLUBS),
        (Rank.EIGHT, Suit.SPADES),
    )

    assert pair_of_aces > pair_of_kings


def test_first_kicker_breaks_tie_between_equal_pairs() -> None:
    pair_of_aces_with_king = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.FOUR, Suit.SPADES),
    )

    pair_of_aces_with_queen = evaluate_value(
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.HEARTS),
        (Rank.EIGHT, Suit.CLUBS),
    )

    assert pair_of_aces_with_king > pair_of_aces_with_queen


def test_last_kicker_is_compared_when_previous_values_are_equal() -> None:
    first_hand = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.FIVE, Suit.SPADES),
    )

    second_hand = evaluate_value(
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.KING, Suit.SPADES),
        (Rank.NINE, Suit.HEARTS),
        (Rank.FOUR, Suit.CLUBS),
    )

    assert first_hand > second_hand


def test_higher_top_pair_wins_between_two_pair_hands() -> None:
    aces_and_kings = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.KING, Suit.CLUBS),
        (Rank.TWO, Suit.SPADES),
    )

    kings_and_queens = evaluate_value(
        (Rank.KING, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.CLUBS),
        (Rank.ACE, Suit.SPADES),
    )

    assert aces_and_kings > kings_and_queens


def test_lower_pair_breaks_tie_between_equal_top_pairs() -> None:
    aces_and_kings = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.KING, Suit.CLUBS),
        (Rank.TWO, Suit.SPADES),
    )

    aces_and_queens = evaluate_value(
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.QUEEN, Suit.HEARTS),
        (Rank.KING, Suit.CLUBS),
    )

    assert aces_and_kings > aces_and_queens


def test_kicker_breaks_tie_between_identical_two_pairs() -> None:
    aces_and_kings_with_queen = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.KING, Suit.CLUBS),
        (Rank.QUEEN, Suit.SPADES),
    )

    aces_and_kings_with_jack = evaluate_value(
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.KING, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.JACK, Suit.CLUBS),
    )

    assert aces_and_kings_with_queen > aces_and_kings_with_jack


def test_kicker_breaks_tie_between_equal_three_of_a_kind() -> None:
    three_queens_with_ace = evaluate_value(
        (Rank.QUEEN, Suit.SPADES),
        (Rank.QUEEN, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.SEVEN, Suit.SPADES),
    )

    three_queens_with_king = evaluate_value(
        (Rank.QUEEN, Suit.CLUBS),
        (Rank.QUEEN, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.KING, Suit.SPADES),
        (Rank.JACK, Suit.CLUBS),
    )

    assert three_queens_with_ace > three_queens_with_king


def test_six_high_straight_beats_wheel_straight() -> None:
    six_high_straight = evaluate_value(
        (Rank.SIX, Suit.SPADES),
        (Rank.FIVE, Suit.HEARTS),
        (Rank.FOUR, Suit.DIAMONDS),
        (Rank.THREE, Suit.CLUBS),
        (Rank.TWO, Suit.SPADES),
    )

    wheel_straight = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.FIVE, Suit.HEARTS),
        (Rank.FOUR, Suit.DIAMONDS),
        (Rank.THREE, Suit.CLUBS),
        (Rank.TWO, Suit.SPADES),
    )

    assert six_high_straight.tiebreak == (Rank.SIX.value,)
    assert wheel_straight.tiebreak == (Rank.FIVE.value,)
    assert six_high_straight > wheel_straight


def test_flush_comparison_checks_more_than_highest_card() -> None:
    first_flush = evaluate_value(
        (Rank.ACE, Suit.HEARTS),
        (Rank.JACK, Suit.HEARTS),
        (Rank.NINE, Suit.HEARTS),
        (Rank.SIX, Suit.HEARTS),
        (Rank.THREE, Suit.HEARTS),
    )

    second_flush = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.JACK, Suit.SPADES),
        (Rank.NINE, Suit.SPADES),
        (Rank.FIVE, Suit.SPADES),
        (Rank.FOUR, Suit.SPADES),
    )

    assert first_flush > second_flush

def test_full_house_is_compared_by_three_of_a_kind_first() -> None:
    kings_full_of_twos = evaluate_value(
        (Rank.KING, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.TWO, Suit.CLUBS),
        (Rank.TWO, Suit.SPADES),
    )

    queens_full_of_aces = evaluate_value(
        (Rank.QUEEN, Suit.SPADES),
        (Rank.QUEEN, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.ACE, Suit.SPADES),
    )

    assert kings_full_of_twos > queens_full_of_aces


def test_pair_breaks_tie_between_full_houses_with_equal_triplets() -> None:
    aces_full_of_kings = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.KING, Suit.CLUBS),
        (Rank.KING, Suit.SPADES),
    )

    aces_full_of_queens = evaluate_value(
        (Rank.ACE, Suit.CLUBS),
        (Rank.ACE, Suit.HEARTS),
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.QUEEN, Suit.CLUBS),
    )

    assert aces_full_of_kings > aces_full_of_queens


def test_kicker_breaks_tie_between_equal_four_of_a_kind() -> None:
    four_nines_with_ace = evaluate_value(
        (Rank.NINE, Suit.SPADES),
        (Rank.NINE, Suit.HEARTS),
        (Rank.NINE, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.ACE, Suit.SPADES),
    )

    four_nines_with_king = evaluate_value(
        (Rank.NINE, Suit.SPADES),
        (Rank.NINE, Suit.HEARTS),
        (Rank.NINE, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.KING, Suit.SPADES),
    )

    assert four_nines_with_ace > four_nines_with_king


def test_royal_flush_beats_king_high_straight_flush() -> None:
    royal_flush = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.SPADES),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.SPADES),
        (Rank.TEN, Suit.SPADES),
    )

    king_high_straight_flush = evaluate_value(
        (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.HEARTS),
        (Rank.JACK, Suit.HEARTS),
        (Rank.TEN, Suit.HEARTS),
        (Rank.NINE, Suit.HEARTS),
    )

    assert royal_flush.is_royal_flush
    assert not king_high_straight_flush.is_royal_flush
    assert royal_flush > king_high_straight_flush


def test_same_rank_values_with_different_suits_are_equal() -> None:
    first_hand = evaluate_value(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.FOUR, Suit.SPADES),
    )

    second_hand = evaluate_value(
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.KING, Suit.HEARTS),
        (Rank.NINE, Suit.SPADES),
        (Rank.FOUR, Suit.CLUBS),
    )

    assert first_hand == second_hand
    assert not first_hand > second_hand
    assert not first_hand < second_hand

def test_best_hand_on_board_produces_exact_tie() -> None:
    board = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.CLUBS),
        (Rank.JACK, Suit.HEARTS),
        (Rank.TEN, Suit.SPADES),
    )

    player_cards = make_cards(
        (Rank.TWO, Suit.CLUBS),
        (Rank.THREE, Suit.CLUBS),
    )

    dealer_cards = make_cards(
        (Rank.FOUR, Suit.DIAMONDS),
        (Rank.FIVE, Suit.DIAMONDS),
    )

    player_hand = evaluate_best_hand(
        [*player_cards, *board]
    )

    dealer_hand = evaluate_best_hand(
        [*dealer_cards, *board]
    )

    assert player_hand.value.rank is HandRank.STRAIGHT
    assert dealer_hand.value.rank is HandRank.STRAIGHT
    assert player_hand.value == dealer_hand.value
    assert set(player_hand.cards) == set(board)
    assert set(dealer_hand.cards) == set(board)


def test_hole_card_kicker_breaks_tie_on_two_pair_board() -> None:
    board = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.KING, Suit.CLUBS),
        (Rank.TWO, Suit.SPADES),
    )

    player_cards = make_cards(
        (Rank.QUEEN, Suit.SPADES),
        (Rank.THREE, Suit.HEARTS),
    )

    dealer_cards = make_cards(
        (Rank.JACK, Suit.SPADES),
        (Rank.TEN, Suit.HEARTS),
    )

    player_hand = evaluate_best_hand(
        [*player_cards, *board]
    )

    dealer_hand = evaluate_best_hand(
        [*dealer_cards, *board]
    )

    assert player_hand.value.rank is HandRank.TWO_PAIR
    assert dealer_hand.value.rank is HandRank.TWO_PAIR

    assert player_hand.value.tiebreak == (
        Rank.ACE.value,
        Rank.KING.value,
        Rank.QUEEN.value,
    )

    assert dealer_hand.value.tiebreak == (
        Rank.ACE.value,
        Rank.KING.value,
        Rank.JACK.value,
    )

    assert player_hand.value > dealer_hand.value


def test_player_does_not_have_to_use_any_hole_card() -> None:
    board = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.SPADES),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.SPADES),
        (Rank.TEN, Suit.SPADES),
    )

    hole_cards = make_cards(
        (Rank.TWO, Suit.HEARTS),
        (Rank.THREE, Suit.DIAMONDS),
    )

    result = evaluate_best_hand(
        [*hole_cards, *board]
    )

    assert result.value.is_royal_flush
    assert set(result.cards) == set(board)


def test_player_can_use_both_hole_cards() -> None:
    board = make_cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.SEVEN, Suit.DIAMONDS),
        (Rank.FOUR, Suit.CLUBS),
        (Rank.TWO, Suit.SPADES),
    )

    hole_cards = make_cards(
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
    )

    result = evaluate_best_hand(
        [*hole_cards, *board]
    )

    assert result.value.rank is HandRank.TWO_PAIR
    assert result.value.tiebreak == (
        Rank.ACE.value,
        Rank.KING.value,
        Rank.SEVEN.value,
    )

    assert set(hole_cards).issubset(set(result.cards))

def test_selects_stronger_full_house_from_two_triplets() -> None:
    cards = make_cards(
        (Rank.KING, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.QUEEN, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.TWO, Suit.CLUBS),
    )

    result = evaluate_best_hand(cards)

    assert result.value.rank is HandRank.FULL_HOUSE
    assert result.value.tiebreak == (
        Rank.KING.value,
        Rank.QUEEN.value,
    )