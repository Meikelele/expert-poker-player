from expert_poker_player.hands import HandRank


def test_hand_ranks_are_ordered_by_poker_strength() -> None:
    assert (
        HandRank.HIGH_CARD
        < HandRank.ONE_PAIR
        < HandRank.TWO_PAIR
        < HandRank.THREE_OF_A_KIND
        < HandRank.STRAIGHT
        < HandRank.FLUSH
        < HandRank.FULL_HOUSE
        < HandRank.FOUR_OF_A_KIND
        < HandRank.STRAIGHT_FLUSH
    )


def test_straight_flush_is_the_strongest_category() -> None:
    assert HandRank.STRAIGHT_FLUSH > HandRank.FOUR_OF_A_KIND


def test_one_pair_is_stronger_than_high_card() -> None:
    assert HandRank.ONE_PAIR > HandRank.HIGH_CARD