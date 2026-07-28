from random import Random
from typing import cast

from expert_poker_player.cards import Deck
from expert_poker_player.hands import evaluate_best_hand
from expert_poker_player.uth.card_source import CardSource
from expert_poker_player.uth.dealing import (
    deal_initial_cards,
    reveal_flop,
    reveal_turn_and_river,
)
from expert_poker_player.uth.enums import (
    Action,
    GamePhase,
    RoundOutcome,
)
from expert_poker_player.uth.errors import (
    IllegalActionError,
    RoundFinishedError,
    RoundNotStartedError,
)
from expert_poker_player.uth.models import (
    RoundState,
    StepResult,
    UTHObservation,
    observation_from_state,
    step_result_from_state,
)
from expert_poker_player.uth.rules import legal_actions_for_phase
from expert_poker_player.uth.settlement import (
    settle_fold,
    settle_showdown,
)
from expert_poker_player.uth.showdown import ShowdownResult
from expert_poker_player.uth.trace import (
    DecisionRecord,
    RoundTrace,
)


class UTHGame:
    """
    Silnik pojedynczego rozdania Ultimate Texas Hold'em.

    Silnik przechowuje pełny stan rozdania, lecz agentowi
    udostępnia wyłącznie bezpieczną obserwację.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        record_history: bool = False,
    ) -> None:
        if seed is not None and type(seed) is not int:
            raise TypeError("seed must be an integer or None")

        if type(record_history) is not bool:
            raise TypeError("record_history must be a boolean")

        self._random = Random(seed)
        self._state: RoundState | None = None
        self._card_source: CardSource | None = None

        self._record_history = record_history
        self._round_id = 0
        self._decisions: list[DecisionRecord] | None = None

    @property
    def state(self) -> RoundState:
        """
        Zwraca pełny stan wewnętrzny rozdania.

        Stan zawiera również ukryte karty krupiera i nie powinien
        być bezpośrednio przekazywany agentowi.
        """

        if self._state is None:
            raise RoundNotStartedError(
                "round has not been started; call reset() first"
            )

        return self._state

    @property
    def observation(self) -> UTHObservation:
        """Zwraca informacje widoczne dla agenta."""

        return observation_from_state(self.state)

    @property
    def is_started(self) -> bool:
        """Informuje, czy rozpoczęto rozdanie."""

        return self._state is not None # type: ignore

    @property
    def history_enabled(self) -> bool:
        """Informuje, czy silnik zapisuje przebieg rozdania."""

        return self._record_history # type: ignore


    @property
    def trace(self) -> RoundTrace | None:
        """
        Zwraca aktualny zapis rozdania.

        Zwraca None, gdy historia jest wyłączona albo rozdanie
        nie zostało jeszcze rozpoczęte.
        """

        if (
            not self._record_history
            or self._state is None
            or self._decisions is None
        ):
            return None

        return RoundTrace(
            round_id=self._round_id,
            decisions=tuple(self._decisions),
            state=self._state,
        )

    def reset(
        self,
        *,
        card_source: CardSource | None = None,
    ) -> UTHObservation:
        """
        Rozpoczyna nowe rozdanie i zwraca obserwację preflop.

        Bez jawnego źródła kart tworzona jest nowa losowa talia.
        W testach można przekazać FixedDeck.
        """

        if card_source is None:
            card_source = self._create_random_deck()

        # Najpierw próbujemy utworzyć stan, a dopiero później
        # zapisujemy go w silniku. Nie zostawiamy połowicznego resetu.
        initial_state = deal_initial_cards(card_source)

        self._card_source = card_source
        self._state = initial_state

        self._round_id += 1
        self._decisions = (
            []
            if self._record_history
            else None
        )

        return self.observation

    def step(
        self,
        action: Action,
    ) -> StepResult:
        """
        Wykonuje akcję gracza i przechodzi do kolejnego stanu.

        Zwraca wynik kroku zawierający obserwację oraz, po zakończeniu
        rozdania, wynik i szczegółowe rozliczenie zakładów.
        """

        current_state = self.state

        if current_state.phase is GamePhase.TERMINAL:
            raise RoundFinishedError(
                "round has already finished; call reset() "
                "before performing another action"
            )

        if not isinstance(action, Action): # type: ignore
            raise TypeError("action must be an instance of Action")

        legal_actions = legal_actions_for_phase(
            current_state.phase
        )

        if action not in legal_actions:
            legal_action_names = ", ".join(
                sorted(
                    legal_action.value
                    for legal_action in legal_actions
                )
            )

            raise IllegalActionError(
                f"{action.value} is illegal during "
                f"{current_state.phase.value}; "
                f"legal actions: {legal_action_names}"
            )
        
        self._record_decision(
            observation=observation_from_state(current_state),
            action=action,
        )

        if current_state.phase is GamePhase.PREFLOP:
            return self._step_preflop(
                state=current_state,
                action=action,
            )

        if current_state.phase is GamePhase.FLOP:
            return self._step_flop(
                state=current_state,
                action=action,
            )

        return self._step_river(
            state=current_state,
            action=action,
        )

    def _step_preflop(
        self,
        *,
        state: RoundState,
        action: Action,
    ) -> StepResult:
        """Obsługuje akcję wykonaną przed flopem."""

        card_source = cast(CardSource, self._card_source)

        flop_state = reveal_flop(
            state=state,
            card_source=card_source,
        )

        if action is Action.CHECK:
            return self._store_state(flop_state)

        river_state = reveal_turn_and_river(
            state=flop_state,
            card_source=card_source,
        )

        play_multiplier = (
            4
            if action is Action.BET_4X
            else 3
        )

        return self._finish_showdown(
            state=river_state,
            play_multiplier=play_multiplier,
        )

    def _step_flop(
        self,
        *,
        state: RoundState,
        action: Action,
    ) -> StepResult:
        """Obsługuje akcję wykonaną po odkryciu flopa."""

        card_source = cast(CardSource, self._card_source)

        river_state = reveal_turn_and_river(
            state=state,
            card_source=card_source,
        )

        if action is Action.CHECK:
            return self._store_state(river_state)

        return self._finish_showdown(
            state=river_state,
            play_multiplier=2,
        )

    def _step_river(
        self,
        *,
        state: RoundState,
        action: Action,
    ) -> StepResult:
        """Obsługuje ostatnią decyzję gracza."""

        if action is Action.FOLD:
            return self._finish_fold(state)

        return self._finish_showdown(
            state=state,
            play_multiplier=1,
        )

    def _finish_showdown(
        self,
        *,
        state: RoundState,
        play_multiplier: int,
    ) -> StepResult:
        """Ocenia obie ręce i kończy rozdanie showdownem."""

        showdown = ShowdownResult(
            player_hand=evaluate_best_hand(
                (
                    *state.player_cards,
                    *state.community_cards,
                )
            ),
            dealer_hand=evaluate_best_hand(
                (
                    *state.dealer_cards,
                    *state.community_cards,
                )
            ),
        )

        outcome = showdown.outcome

        settlement = settle_showdown(
            player_hand=showdown.player_hand.value,
            dealer_hand=showdown.dealer_hand.value,
            play_multiplier=play_multiplier,
        )

        terminal_state = RoundState(
            phase=GamePhase.TERMINAL,
            player_cards=state.player_cards,
            dealer_cards=state.dealer_cards,
            community_cards=state.community_cards,
            burned_cards=state.burned_cards,
            play_multiplier=play_multiplier,
            outcome=outcome,
            settlement=settlement,
            showdown=showdown,
        )

        return self._store_state(terminal_state)

    def _finish_fold(
        self,
        state: RoundState,
    ) -> StepResult:
        """Kończy rozdanie spasowaniem gracza."""

        terminal_state = RoundState(
            phase=GamePhase.TERMINAL,
            player_cards=state.player_cards,
            dealer_cards=state.dealer_cards,
            community_cards=state.community_cards,
            burned_cards=state.burned_cards,
            play_multiplier=None,
            outcome=RoundOutcome.PLAYER_FOLD,
            settlement=settle_fold(),
        )

        return self._store_state(terminal_state)

    def _store_state(
        self,
        state: RoundState,
    ) -> StepResult:
        """Zapisuje nowy stan i buduje wynik kroku."""

        self._state = state

        return step_result_from_state(state)

    def _create_random_deck(self) -> Deck:
        """Tworzy świeżą talię z powtarzalnego strumienia seedów."""

        deck_seed = self._random.randrange(0, 2**63)

        return Deck(seed=deck_seed)

    def _record_decision(
        self,
        *,
        observation: UTHObservation,
        action: Action,
    ) -> None:
        """Zapisuje legalną decyzję, gdy historia jest włączona."""

        if self._decisions is None:
            return

        self._decisions.append(
            DecisionRecord(
                observation=observation,
                action=action,
            )
        )