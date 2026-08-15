from typing import cast

from expert_poker_player.agents import Agent
from expert_poker_player.cards import Deck
from expert_poker_player.evaluation.models import (
    EpisodeResult,
    SimulationConfig,
    SimulationResult,
)
from expert_poker_player.uth import (
    Action,
    CardSource,
    RoundOutcome,
    Settlement,
    UTHGame,
)


def play_round(
    game: UTHGame,
    agent: Agent,
    *,
    card_source: CardSource | None = None,
) -> EpisodeResult:
    """
    Rozgrywa pojedynczy epizod UTH przy użyciu wskazanego agenta.

    Agent otrzymuje wyłącznie obserwacje udostępniane przez silnik.
    Opcjonalne źródło kart umożliwia deterministyczne odtworzenie rozdania.
    """

    if not isinstance(game, UTHGame): # type: ignore
        raise TypeError(
            "game must be an instance of UTHGame"
        )

    if not isinstance(agent, Agent): # type: ignore
        raise TypeError(
            "agent must implement the Agent protocol"
        )

    observation = game.reset(
        card_source=card_source,
    )
    actions: list[Action] = []

    while True:
        action = agent.select_action(observation)
        step_result = game.step(action)
        actions.append(action)

        if step_result.terminated:
            return EpisodeResult(
                actions=tuple(actions),
                outcome=cast(
                    RoundOutcome,
                    step_result.outcome,
                ),
                settlement=cast(
                    Settlement,
                    step_result.settlement,
                ),
            )

        observation = step_result.observation

def run_simulation(
    agent: Agent,
    config: SimulationConfig,
) -> SimulationResult:
    """
    Uruchamia agenta na deterministycznym zestawie rozdań.

    Każdy seed z konfiguracji tworzy osobną talię i osobny epizod.
    """

    if not isinstance(agent, Agent): # type: ignore
        raise TypeError(
            "agent must implement the Agent protocol"
        )

    if not isinstance(config, SimulationConfig): # type: ignore
        raise TypeError(
            "config must be an instance of SimulationConfig"
        )

    episodes = tuple(
        play_round(
            game=UTHGame(),
            agent=agent,
            card_source=Deck(seed=deck_seed),
        )
        for deck_seed in config.deck_seeds
    )

    return SimulationResult(
        config=config,
        episodes=episodes,
    )