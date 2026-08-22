import pytest

from expert_poker_player.cards import Deck
from expert_poker_player.agents import RandomAgent
from expert_poker_player.evaluation import (
    EpisodeResult,
    SimulationConfig,
    SimulationResult,
    play_round,
    run_simulation,
)
from expert_poker_player.uth import (
    Action,
    GamePhase,
    IllegalActionError,
    UTHGame,
    UTHObservation,
)

class CheckUntilRiverAgent:
    """Agent czekający do rivera, a następnie wykonujący zakład 1x."""

    def __init__(self) -> None:
        self.observations: list[UTHObservation] = []

    def select_action(
        self,
        observation: UTHObservation,
    ) -> Action:
        self.observations.append(observation)

        if observation.phase is GamePhase.RIVER:
            return Action.BET_1X

        return Action.CHECK


class ScriptedAgent:
    """Agent zwracający wcześniej ustaloną sekwencję akcji."""

    def __init__(
        self,
        actions: tuple[Action, ...],
    ) -> None:
        self._actions = iter(actions)
        self.observations: list[UTHObservation] = []

    def select_action(
        self,
        observation: UTHObservation,
    ) -> Action:
        self.observations.append(observation)

        return next(self._actions)


@pytest.mark.parametrize(
    (
        "actions",
        "expected_phases",
        "expected_multiplier",
    ),
    [
        (
            (Action.BET_4X,),
            (GamePhase.PREFLOP,),
            4,
        ),
        (
            (Action.BET_3X,),
            (GamePhase.PREFLOP,),
            3,
        ),
        (
            (
                Action.CHECK,
                Action.BET_2X,
            ),
            (
                GamePhase.PREFLOP,
                GamePhase.FLOP,
            ),
            2,
        ),
        (
            (
                Action.CHECK,
                Action.CHECK,
                Action.BET_1X,
            ),
            (
                GamePhase.PREFLOP,
                GamePhase.FLOP,
                GamePhase.RIVER,
            ),
            1,
        ),
        (
            (
                Action.CHECK,
                Action.CHECK,
                Action.FOLD,
            ),
            (
                GamePhase.PREFLOP,
                GamePhase.FLOP,
                GamePhase.RIVER,
            ),
            None,
        ),
    ],
)
def test_plays_each_complete_round_path(
    actions: tuple[Action, ...],
    expected_phases: tuple[GamePhase, ...],
    expected_multiplier: int | None,
) -> None:
    game = UTHGame(seed=123)
    agent = ScriptedAgent(actions)

    result = play_round(
        game=game,
        agent=agent,
    )

    assert isinstance(result, EpisodeResult)
    assert result.actions == actions
    assert result.decision_count == len(actions)
    assert result.play_multiplier == expected_multiplier
    assert result.folded is (
        actions[-1] is Action.FOLD
    )

    assert tuple(
        observation.phase
        for observation in agent.observations
    ) == expected_phases

    assert all(
        isinstance(observation, UTHObservation)
        for observation in agent.observations
    )

    assert game.state.phase is GamePhase.TERMINAL
    assert game.observation.terminated


def test_rejects_invalid_game() -> None:
    agent = ScriptedAgent((Action.BET_4X,))

    with pytest.raises(
        TypeError,
        match="game must be an instance of UTHGame",
    ):
        play_round(
            game=object(),  # type: ignore[arg-type]
            agent=agent,
        )


def test_rejects_object_without_agent_protocol() -> None:
    with pytest.raises(
        TypeError,
        match="agent must implement the Agent protocol",
    ):
        play_round(
            game=UTHGame(seed=123),
            agent=object(),  # type: ignore[arg-type]
        )


def test_propagates_illegal_agent_action() -> None:
    game = UTHGame(seed=123)
    agent = ScriptedAgent((Action.BET_1X,))

    with pytest.raises(IllegalActionError):
        play_round(
            game=game,
            agent=agent,
        )


def test_explicit_card_source_controls_the_round() -> None:
    first_result = play_round(
        game=UTHGame(seed=1),
        agent=ScriptedAgent((Action.BET_4X,)),
        card_source=Deck(seed=777),
    )
    second_result = play_round(
        game=UTHGame(seed=999),
        agent=ScriptedAgent((Action.BET_4X,)),
        card_source=Deck(seed=777),
    )

    assert first_result.outcome is second_result.outcome
    assert first_result.settlement == second_result.settlement

def test_runs_one_episode_for_each_deck_seed() -> None:
    config = SimulationConfig(
        deck_seeds=(101, 202, 303),
    )
    agent = ScriptedAgent(
        (
            Action.BET_4X,
            Action.BET_4X,
            Action.BET_4X,
        )
    )

    result = run_simulation(
        agent=agent,
        config=config,
    )

    assert isinstance(result, SimulationResult)
    assert result.config is config
    assert result.round_count == 3
    assert len(result.episodes) == 3

    assert all(
        episode.actions == (Action.BET_4X,)
        for episode in result.episodes
    )


def test_same_configuration_and_agent_seed_are_reproducible() -> None:
    config = SimulationConfig(
        deck_seeds=(101, 202, 303, 404, 505),
    )

    first_result = run_simulation(
        agent=RandomAgent(seed=999),
        config=config,
    )
    second_result = run_simulation(
        agent=RandomAgent(seed=999),
        config=config,
    )

    assert first_result == second_result


def test_same_configuration_exposes_same_cards() -> None:
    config = SimulationConfig(
        deck_seeds=(101, 202, 303),
    )
    first_agent = CheckUntilRiverAgent()
    second_agent = CheckUntilRiverAgent()

    first_result = run_simulation(
        agent=first_agent,
        config=config,
    )
    second_result = run_simulation(
        agent=second_agent,
        config=config,
    )

    assert first_result == second_result
    assert first_agent.observations == second_agent.observations


def test_result_matches_individually_played_rounds() -> None:
    config = SimulationConfig(
        deck_seeds=(101, 202),
    )

    simulation = run_simulation(
        agent=CheckUntilRiverAgent(),
        config=config,
    )

    expected_episodes = tuple(
        play_round(
            game=UTHGame(),
            agent=CheckUntilRiverAgent(),
            card_source=Deck(seed=deck_seed),
        )
        for deck_seed in config.deck_seeds
    )

    assert simulation.episodes == expected_episodes


def test_simulation_rejects_object_without_agent_protocol() -> None:
    config = SimulationConfig(
        deck_seeds=(101,),
    )

    with pytest.raises(
        TypeError,
        match="agent must implement the Agent protocol",
    ):
        run_simulation(
            agent=object(),  # type: ignore[arg-type]
            config=config,
        )


def test_simulation_rejects_invalid_config() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "config must be an instance "
            "of SimulationConfig"
        ),
    ):
        run_simulation(
            agent=CheckUntilRiverAgent(),
            config=object(),  # type: ignore[arg-type]
        )