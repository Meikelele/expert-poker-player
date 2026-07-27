from random import Random

from expert_poker_player.cards import Deck
from expert_poker_player.uth.card_source import CardSource
from expert_poker_player.uth.dealing import deal_initial_cards
from expert_poker_player.uth.errors import RoundNotStartedError
from expert_poker_player.uth.models import (
    RoundState,
    UTHObservation,
    observation_from_state,
)


class UTHGame:
    """
    Silnik przechowuje pełny stan pojedynczego rozdania Ultimate Texas Hold'em.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:
        if seed is not None and type(seed) is not int:
            raise TypeError("seed must be an integer or None")

        self._random = Random(seed)
        self._state: RoundState | None = None
        self._card_source: CardSource | None = None

    @property
    def state(self) -> RoundState:
        """
        Zwraca pełny stan wewnętrzny rozdania.
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

        return self._state is not None

    def reset(
        self,
        *,
        card_source: CardSource | None = None,
    ) -> UTHObservation:
        """
        Rozpoczyna nowe rozdanie i zwraca obserwację preflop.

        Gdy card_source nie zostanie przekazane, silnik tworzy
        nową losową talię. W testach można przekazać FixedDeck.
        """

        if card_source is None:
            card_source = self._create_random_deck()

        self._card_source = card_source
        self._state = deal_initial_cards(card_source)

        return self.observation

    def _create_random_deck(self) -> Deck:
        """
        Tworzy nową talię z ziarnem pochodzącym z głównego RNG.

        Dzięki temu kolejne rozdania są różne, ale cała seria
        pozostaje powtarzalna dla tego samego seed.
        """

        deck_seed = self._random.randrange(0, 2**63)

        return Deck(seed=deck_seed)