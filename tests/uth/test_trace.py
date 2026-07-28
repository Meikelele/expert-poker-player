import pytest

from expert_poker_player.cards import Card, Rank, Suit
from expert_poker_player.uth import (
    Action,
    DecisionRecord,
    GamePhase,
    RoundState,
    RoundTrace,
    UTHObservation,
    legal_actions_for_phase,
    # RoundOutcome,
    # settle_fold,
)


PLAYER_CARDS = (
    Card(rank=Rank.ACE, suit=Suit.SPADES),
    Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
)

DEALER_CARDS = (
    Card(rank=Rank.KING, suit=Suit.HEARTS),
    Card(rank=Rank.JACK, suit=Suit.CLUBS),
)

COMMUNITY_CARDS = (
    Card(rank=Rank.TWO, suit=Suit.CLUBS),
    Card(rank=Rank.FIVE, suit=Suit.DIAMONDS),
    Card(rank=Rank.SEVEN, suit=Suit.SPADES),
    Card(rank=Rank.NINE, suit=Suit.HEARTS),
    Card(rank=Rank.TEN, suit=Suit.CLUBS),
)


def make_preflop_observation() -> UTHObservation:
    return UTHObservation(
        phase=GamePhase.PREFLOP,
        player_cards=PLAYER_CARDS,
        community_cards=(),
        legal_actions=legal_actions_for_phase(
            GamePhase.PREFLOP
        ),
    )


def make_preflop_state() -> RoundState:
    return RoundState(
        phase=GamePhase.PREFLOP,
        player_cards=PLAYER_CARDS,
        dealer_cards=DEALER_CARDS,
        community_cards=(),
        burned_cards=(),
    )


def test_creates_valid_decision_record() -> None:
    observation = make_preflop_observation()

    decision = DecisionRecord(
        observation=observation,
        action=Action.CHECK,
    )

    assert decision.observation == observation
    assert decision.action is Action.CHECK


def test_decision_record_rejects_invalid_observation() -> None:
    with pytest.raises(
        TypeError,
        match="observation must be an instance of UTHObservation",
    ):
        DecisionRecord(
            observation="observation",  # type: ignore[arg-type]
            action=Action.CHECK,
        )


def test_decision_record_rejects_invalid_action_type() -> None:
    with pytest.raises(
        TypeError,
        match="action must be an instance of Action",
    ):
        DecisionRecord(
            observation=make_preflop_observation(),
            action="check",  # type: ignore[arg-type]
        )


def test_decision_record_rejects_illegal_action() -> None:
    with pytest.raises(
        ValueError,
        match="recorded action must be legal",
    ):
        DecisionRecord(
            observation=make_preflop_observation(),
            action=Action.BET_2X,
        )


def test_round_trace_exposes_current_state() -> None:
    decision = DecisionRecord(
        observation=make_preflop_observation(),
        action=Action.CHECK,
    )
    state = make_preflop_state()

    trace = RoundTrace(
        round_id=1,
        decisions=(decision,),
        state=state,
    )

    assert trace.round_id == 1
    assert trace.decisions == (decision,)
    assert trace.state == state
    assert not trace.completed
    assert trace.outcome is None
    assert trace.settlement is None
    assert trace.showdown is None


@pytest.mark.parametrize(
    "round_id",
    [
        "1",
        1.0,
        True,
    ],
)
def test_round_trace_rejects_non_integer_round_id(
    round_id: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="round_id must be an integer",
    ):
        RoundTrace(
            round_id=round_id,  # type: ignore[arg-type]
            decisions=(),
            state=make_preflop_state(),
        )


@pytest.mark.parametrize(
    "round_id",
    [
        0,
        -1,
    ],
)
def test_round_trace_requires_positive_round_id(
    round_id: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="round_id must be positive",
    ):
        RoundTrace(
            round_id=round_id,
            decisions=(),
            state=make_preflop_state(),
        )


def test_round_trace_rejects_non_tuple_decisions() -> None:
    with pytest.raises(
        TypeError,
        match="decisions must be a tuple",
    ):
        RoundTrace(
            round_id=1,
            decisions=[],  # type: ignore[arg-type]
            state=make_preflop_state(),
        )


def test_round_trace_rejects_invalid_decision_element() -> None:
    with pytest.raises(
        TypeError,
        match="must contain only DecisionRecord values",
    ):
        RoundTrace(
            round_id=1,
            decisions=("decision",),  # type: ignore[arg-type]
            state=make_preflop_state(),
        )


def test_round_trace_rejects_invalid_state() -> None:
    with pytest.raises(
        TypeError,
        match="state must be an instance of RoundState",
    ):
        RoundTrace(
            round_id=1,
            decisions=(),
            state="state",  # type: ignore[arg-type]
        )

def test_decision_record_rejects_terminal_observation() -> None:
    terminal_observation = UTHObservation(
        phase=GamePhase.TERMINAL,
        player_cards=PLAYER_CARDS,
        community_cards=COMMUNITY_CARDS,
        legal_actions=frozenset(),
    )

    with pytest.raises(
        ValueError,
        match="cannot record a decision for a terminal observation",
    ):
        DecisionRecord(
            observation=terminal_observation,
            action=Action.CHECK,
        )