from expert_poker_player.cards import Card
from expert_poker_player.uth.card_source import CardSource
from expert_poker_player.uth.enums import GamePhase
from expert_poker_player.uth.errors import (
    InvalidPhaseTransitionError,
)
from expert_poker_player.uth.models import RoundState


def deal_initial_cards(
    card_source: CardSource,
) -> RoundState:
    """
    Rozdaje po dwie karty graczowi i krupierowi.

    Kolejność rozdawania:
    player, dealer, player, dealer.
    """

    source = _require_card_source(card_source)

    player_first_card = _draw_card(source)
    dealer_first_card = _draw_card(source)
    player_second_card = _draw_card(source)
    dealer_second_card = _draw_card(source)

    return RoundState(
        phase=GamePhase.PREFLOP,
        player_cards=(
            player_first_card,
            player_second_card,
        ),
        dealer_cards=(
            dealer_first_card,
            dealer_second_card,
        ),
        community_cards=(),
        burned_cards=(),
    )


def reveal_flop(
    state: RoundState,
    card_source: CardSource,
) -> RoundState:
    """Spala jedną kartę i odkrywa trzy karty flopa."""

    current_state = _require_round_state(state)
    source = _require_card_source(card_source)

    _require_phase(
        current_state,
        expected_phase=GamePhase.PREFLOP,
    )

    burned_card = _draw_card(source)
    flop = _draw_many(source, count=3)

    return RoundState(
        phase=GamePhase.FLOP,
        player_cards=current_state.player_cards,
        dealer_cards=current_state.dealer_cards,
        community_cards=(
            *current_state.community_cards,
            *flop,
        ),
        burned_cards=(
            *current_state.burned_cards,
            burned_card,
        ),
    )


def reveal_turn_and_river(
    state: RoundState,
    card_source: CardSource,
) -> RoundState:
    """
    Spala jedną kartę, a następnie odkrywa turn i river.

    W Ultimate Texas Hold'em turn i river są odkrywane razem,
    ponieważ pomiędzy nimi gracz nie wykonuje żadnej akcji.
    """

    current_state = _require_round_state(state)
    source = _require_card_source(card_source)

    _require_phase(
        current_state,
        expected_phase=GamePhase.FLOP,
    )

    burned_card = _draw_card(source)
    turn_and_river = _draw_many(source, count=2)

    return RoundState(
        phase=GamePhase.RIVER,
        player_cards=current_state.player_cards,
        dealer_cards=current_state.dealer_cards,
        community_cards=(
            *current_state.community_cards,
            *turn_and_river,
        ),
        burned_cards=(
            *current_state.burned_cards,
            burned_card,
        ),
    )


def _draw_many(
    card_source: CardSource,
    *,
    count: int,
) -> tuple[Card, ...]:
    """Dobiera określoną liczbę poprawnych kart."""

    return tuple(
        _draw_card(card_source)
        for _ in range(count)
    )


def _draw_card(
    card_source: CardSource,
) -> Card:
    """Dobiera kartę i sprawdza wynik źródła kart."""

    card = card_source.draw()

    if not isinstance(card, Card): # type: ignore
        raise TypeError(
            "card source must return an instance of Card"
        )

    return card


def _require_card_source(
    value: object,
) -> CardSource:
    """Sprawdza, czy obiekt może dostarczać karty."""

    if not isinstance(value, CardSource):
        raise TypeError(
            "card_source must implement CardSource"
        )

    return value


def _require_round_state(
    value: object,
) -> RoundState:
    """Sprawdza typ stanu rozdania."""

    if not isinstance(value, RoundState):
        raise TypeError(
            "state must be an instance of RoundState"
        )

    return value


def _require_phase(
    state: RoundState,
    *,
    expected_phase: GamePhase,
) -> None:
    """Sprawdza fazę wymaganą przez operację rozdawania."""

    if state.phase is not expected_phase:
        raise InvalidPhaseTransitionError(
            f"expected phase {expected_phase.value}, "
            f"but current phase is {state.phase.value}"
        )