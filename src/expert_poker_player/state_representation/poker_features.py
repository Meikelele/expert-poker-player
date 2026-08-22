from collections import Counter
from typing import Final

from expert_poker_player.cards import (
    Card,
    Rank,
    # Suit,
)
from expert_poker_player.hands import (
    HandRank,
    evaluate_best_hand,
    evaluate_five_card_hand,
)
from expert_poker_player.state_representation.protocol import (
    StateVector,
)
from expert_poker_player.uth import UTHObservation


_MIN_RANK_VALUE: Final = Rank.TWO.value
_MAX_RANK_VALUE: Final = Rank.ACE.value
_RANK_RANGE: Final = (
    _MAX_RANK_VALUE
    - _MIN_RANK_VALUE
)

_MAX_PAIR_COUNT: Final = 3
_MAX_TRIPS_COUNT: Final = 2
_MAX_QUADS_COUNT: Final = 1

_FLUSH_CARD_COUNT: Final = 5
_STRAIGHT_CARD_COUNT: Final = 5

HAND_RANK_FEATURE_COUNT: Final = len(HandRank)

POKER_FEATURE_COUNT: Final = (
    5
    + HAND_RANK_FEATURE_COUNT
    + 6
)

_STRAIGHT_WINDOWS: Final[
    tuple[frozenset[Rank], ...]
] = (
    frozenset(
        {
            Rank.ACE,
            Rank.TWO,
            Rank.THREE,
            Rank.FOUR,
            Rank.FIVE,
        }
    ),
    *(
        frozenset(
            Rank(value)
            for value in range(
                start,
                start + _STRAIGHT_CARD_COUNT,
            )
        )
        for start in range(
            Rank.TWO.value,
            Rank.TEN.value + 1,
        )
    ),
)


def extract_poker_features(
    observation: UTHObservation,
) -> StateVector:
    """Wyznacza cechy pokerowe z informacji widocznych agentowi."""

    if not isinstance(
        observation,
        UTHObservation,
    ): # type: ignore
        raise TypeError(
            "observation must be an instance "
            "of UTHObservation"
        )

    if observation.terminated:
        raise ValueError(
            "cannot extract features from "
            "a terminal observation"
        )

    visible_cards = (
        *observation.player_cards,
        *observation.community_cards,
    )

    high_card, low_card = sorted(
        observation.player_cards,
        key=lambda card: card.rank,
        reverse=True,
    )

    hole_features = (
        _normalize_rank(high_card.rank),
        _normalize_rank(low_card.rank),
        float(
            high_card.rank is low_card.rank
        ),
        float(
            high_card.suit is low_card.suit
        ),
        abs(
            high_card.rank.value
            - low_card.rank.value
        )
        / _RANK_RANGE,
    )

    hand_rank_features = _encode_hand_rank(
        visible_cards
    )

    structure_features = _extract_structure_features(
        community_cards=observation.community_cards,
        visible_cards=visible_cards,
    )

    features = (
        *hole_features,
        *hand_rank_features,
        *structure_features,
    )

    if len(features) != POKER_FEATURE_COUNT:
        raise RuntimeError(
            "poker feature extractor produced "
            "an invalid output size"
        )

    return features


def _normalize_rank(
    rank: Rank,
) -> float:
    return (
        rank.value
        - _MIN_RANK_VALUE
    ) / _RANK_RANGE


def _encode_hand_rank(
    visible_cards: tuple[Card, ...],
) -> StateVector:
    if len(visible_cards) < 5:
        return (
            0.0,
        ) * HAND_RANK_FEATURE_COUNT

    if len(visible_cards) == 5:
        hand_rank = evaluate_five_card_hand(
            visible_cards
        ).rank
    else:
        hand_rank = evaluate_best_hand(
            visible_cards
        ).value.rank

    return tuple(
        1.0
        if rank is hand_rank
        else 0.0
        for rank in HandRank
    )


def _extract_structure_features(
    *,
    community_cards: tuple[Card, ...],
    visible_cards: tuple[Card, ...],
) -> StateVector:
    rank_counts = Counter(
        card.rank
        for card in visible_cards
    )

    suit_counts = Counter(
        card.suit
        for card in visible_cards
    )

    pair_count = sum(
        count == 2
        for count in rank_counts.values()
    )

    trips_count = sum(
        count == 3
        for count in rank_counts.values()
    )

    quads_count = sum(
        count == 4
        for count in rank_counts.values()
    )

    max_suit_count = max(
        suit_counts.values(),
        default=0,
    )

    visible_ranks = {
        card.rank
        for card in visible_cards
    }

    max_straight_cards = max(
        (
            len(
                visible_ranks
                & straight_window
            )
            for straight_window
            in _STRAIGHT_WINDOWS
        ),
        default=0,
    )

    community_ranks = [
        card.rank
        for card in community_cards
    ]

    board_paired = (
        len(set(community_ranks))
        < len(community_ranks)
    )

    return (
        pair_count / _MAX_PAIR_COUNT,
        trips_count / _MAX_TRIPS_COUNT,
        quads_count / _MAX_QUADS_COUNT,
        min(
            max_suit_count,
            _FLUSH_CARD_COUNT,
        )
        / _FLUSH_CARD_COUNT,
        min(
            max_straight_cards,
            _STRAIGHT_CARD_COUNT,
        )
        / _STRAIGHT_CARD_COUNT,
        float(board_paired),
    )