from collections import Counter

from expert_poker_player.cards import (
    Card,
    Rank,
    Suit,
)
from expert_poker_player.hands import (
    HandRank,
    evaluate_five_card_hand,
)
from expert_poker_player.uth import (
    GamePhase,
    UTHObservation,
)


def should_raise_preflop(
    observation: UTHObservation,
) -> bool:
    """Sprawdza, czy strategia bazowa zaleca zakład 4x."""

    _validate_observation_phase(
        observation,
        expected_phase=GamePhase.PREFLOP,
    )

    first_card, second_card = observation.player_cards

    if first_card.rank is second_card.rank:
        return first_card.rank >= Rank.THREE

    high_card, low_card = sorted(
        observation.player_cards,
        key=lambda card: card.rank,
        reverse=True,
    )

    suited = high_card.suit is low_card.suit

    if high_card.rank is Rank.ACE:
        return True

    if high_card.rank is Rank.KING:
        return (
            suited
            or low_card.rank >= Rank.FIVE
        )

    if high_card.rank is Rank.QUEEN:
        threshold = (
            Rank.SIX
            if suited
            else Rank.EIGHT
        )

        return low_card.rank >= threshold

    if high_card.rank is Rank.JACK:
        threshold = (
            Rank.EIGHT
            if suited
            else Rank.TEN
        )

        return low_card.rank >= threshold

    return False


def should_raise_flop(
    observation: UTHObservation,
) -> bool:
    """Sprawdza, czy strategia bazowa zaleca zakład 2x."""

    _validate_observation_phase(
        observation,
        expected_phase=GamePhase.FLOP,
    )

    visible_cards = (
        *observation.player_cards,
        *observation.community_cards,
    )

    hand_value = evaluate_five_card_hand(
        visible_cards
    )

    if hand_value.rank >= HandRank.TWO_PAIR:
        return True

    if _has_hidden_pair(
        player_cards=observation.player_cards,
        visible_cards=visible_cards,
    ):
        return True

    return _has_four_to_flush_with_hidden_ten(
        player_cards=observation.player_cards,
        visible_cards=visible_cards,
    )


def _has_hidden_pair(
    *,
    player_cards: tuple[Card, Card],
    visible_cards: tuple[Card, ...],
) -> bool:
    if (
        player_cards[0].rank is Rank.TWO
        and player_cards[1].rank is Rank.TWO
    ):
        return False

    rank_counts = Counter(
        card.rank
        for card in visible_cards
    )

    return any(
        rank_counts[card.rank] >= 2
        for card in player_cards
    )


def _has_four_to_flush_with_hidden_ten(
    *,
    player_cards: tuple[Card, Card],
    visible_cards: tuple[Card, ...],
) -> bool:
    for suit in Suit:
        suited_cards = tuple(
            card
            for card in visible_cards
            if card.suit is suit
        )

        if len(suited_cards) < 4:
            continue

        if any(
            card.suit is suit
            and card.rank >= Rank.TEN
            for card in player_cards
        ):
            return True

    return False


def _validate_observation_phase(
    observation: UTHObservation,
    *,
    expected_phase: GamePhase,
) -> None:
    if not isinstance(observation, UTHObservation): # type: ignore
        raise TypeError(
            "observation must be an instance "
            "of UTHObservation"
        )

    if observation.phase is not expected_phase:
        raise ValueError(
            f"expected {expected_phase.value} observation, "
            f"received {observation.phase.value}"
        )