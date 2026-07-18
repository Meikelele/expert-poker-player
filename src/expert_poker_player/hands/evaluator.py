# wykonuje wlasciwa analize rak
from collections import Counter
from collections.abc import Iterable, Sequence

from expert_poker_player.cards import Card, Rank
from expert_poker_player.hands.hand_rank import HandRank
from expert_poker_player.hands.hand_value import HandValue


def evaluate_five_card_hand(cards: Sequence[Card]) -> HandValue:
    """
    Ocenia układ składający się z dokładnie pięciu kart.

    Zwraca kategorię układu oraz wartości rozstrzygające remis.
    """

    hand = tuple(cards)
    _validate_five_card_hand(hand)

    rank_counts = Counter(card.rank.value for card in hand)
    rank_values_desc = sorted(
        (card.rank.value for card in hand),
        reverse=True,
    )

    is_flush = len({card.suit for card in hand}) == 1
    straight_high_card = _get_straight_high_card(rank_counts.keys())

    four_of_a_kind = sorted(
        (
            rank_value
            for rank_value, count in rank_counts.items()
            if count == 4
        ),
        reverse=True,
    )

    three_of_a_kind = sorted(
        (
            rank_value
            for rank_value, count in rank_counts.items()
            if count == 3
        ),
        reverse=True,
    )

    pairs = sorted(
        (
            rank_value
            for rank_value, count in rank_counts.items()
            if count == 2
        ),
        reverse=True,
    )

    single_cards = sorted(
        (
            rank_value
            for rank_value, count in rank_counts.items()
            if count == 1
        ),
        reverse=True,
    )

    if is_flush and straight_high_card is not None:
        return HandValue(
            rank=HandRank.STRAIGHT_FLUSH,
            tiebreak=(straight_high_card,),
        )

    if four_of_a_kind:
        return HandValue(
            rank=HandRank.FOUR_OF_A_KIND,
            tiebreak=(
                four_of_a_kind[0],
                single_cards[0],
            ),
        )

    if three_of_a_kind and pairs:
        return HandValue(
            rank=HandRank.FULL_HOUSE,
            tiebreak=(
                three_of_a_kind[0],
                pairs[0],
            ),
        )

    if is_flush:
        return HandValue(
            rank=HandRank.FLUSH,
            tiebreak=tuple(rank_values_desc),
        )

    if straight_high_card is not None:
        return HandValue(
            rank=HandRank.STRAIGHT,
            tiebreak=(straight_high_card,),
        )

    if three_of_a_kind:
        return HandValue(
            rank=HandRank.THREE_OF_A_KIND,
            tiebreak=(
                three_of_a_kind[0],
                *single_cards,
            ),
        )

    if len(pairs) == 2:
        return HandValue(
            rank=HandRank.TWO_PAIR,
            tiebreak=(
                pairs[0],
                pairs[1],
                single_cards[0],
            ),
        )

    if len(pairs) == 1:
        return HandValue(
            rank=HandRank.ONE_PAIR,
            tiebreak=(
                pairs[0],
                *single_cards,
            ),
        )

    return HandValue(
        rank=HandRank.HIGH_CARD,
        tiebreak=tuple(rank_values_desc),
    )


def _validate_five_card_hand(cards: tuple[Card, ...]) -> None:
    """Sprawdza, czy przekazano dokładnie pięć unikalnych kart."""

    if len(cards) != 5:
        raise ValueError(
            f"A five-card hand must contain exactly 5 cards, "
            f"but received {len(cards)}"
        )

    if not all(isinstance(card, Card) for card in cards): # type: ignore
        raise TypeError("all elements must be instances of Card")

    if len(set(cards)) != 5:
        raise ValueError("a poker hand cannot contain duplicate cards")


def _get_straight_high_card(rank_values: Iterable[int]) -> int | None:
    """
    Zwraca wartość najwyższej karty strita.

    Zwraca None, jeśli wartości nie tworzą strita.
    """

    unique_values = sorted(set(rank_values), reverse=True)

    if len(unique_values) != 5:
        return None

    # Specjalny przypadek: A-2-3-4-5.
    # As działa tutaj jako karta poniżej dwójki.
    wheel = [
        Rank.ACE.value,
        Rank.FIVE.value,
        Rank.FOUR.value,
        Rank.THREE.value,
        Rank.TWO.value,
    ]

    if unique_values == wheel:
        return Rank.FIVE.value

    highest_value = unique_values[0]
    lowest_value = unique_values[-1]

    if highest_value - lowest_value == 4:
        return highest_value

    return None