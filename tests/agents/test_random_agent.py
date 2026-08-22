import pytest

from expert_poker_player.agents import (
    Agent,
    RandomAgent,
)
from expert_poker_player.uth import (
    Action,
    GamePhase,
    UTHGame,
    UTHObservation,
)


def make_observation(
    phase: GamePhase,
) -> UTHObservation:
    game = UTHGame(seed=123)
    observation = game.reset()

    if phase is GamePhase.PREFLOP:
        return observation

    observation = game.step(
        Action.CHECK
    ).observation

    if phase is GamePhase.FLOP:
        return observation

    observation = game.step(
        Action.CHECK
    ).observation

    if phase is GamePhase.RIVER:
        return observation

    return game.step(
        Action.FOLD
    ).observation


def test_random_agent_satisfies_agent_protocol() -> None:
    agent = RandomAgent(seed=123)

    assert isinstance(agent, Agent)


@pytest.mark.parametrize(
    "phase",
    [
        GamePhase.PREFLOP,
        GamePhase.FLOP,
        GamePhase.RIVER,
    ],
)
def test_selects_only_legal_actions(
    phase: GamePhase,
) -> None:
    agent = RandomAgent(seed=123)
    observation = make_observation(phase)

    selected_actions = {
        agent.select_action(observation)
        for _ in range(100)
    }

    assert selected_actions
    assert selected_actions <= observation.legal_actions


def test_same_seed_produces_same_action_sequence() -> None:
    first_agent = RandomAgent(seed=123)
    second_agent = RandomAgent(seed=123)

    observations = (
        make_observation(GamePhase.PREFLOP),
        make_observation(GamePhase.FLOP),
        make_observation(GamePhase.RIVER),
    ) * 20

    first_actions = tuple(
        first_agent.select_action(observation)
        for observation in observations
    )
    second_actions = tuple(
        second_agent.select_action(observation)
        for observation in observations
    )

    assert first_actions == second_actions


@pytest.mark.parametrize(
    "seed",
    [
        1.5,
        "123",
        True,
    ],
)
def test_rejects_invalid_seed(
    seed: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="seed must be an integer or None",
    ):
        RandomAgent(
            seed=seed,  # type: ignore[arg-type]
        )


def test_rejects_invalid_observation() -> None:
    agent = RandomAgent(seed=123)

    with pytest.raises(
        TypeError,
        match=(
            "observation must be an instance "
            "of UTHObservation"
        ),
    ):
        agent.select_action(
            object(),  # type: ignore[arg-type]
        )


def test_rejects_terminal_observation() -> None:
    agent = RandomAgent(seed=123)
    observation = make_observation(
        GamePhase.TERMINAL
    )

    with pytest.raises(
        ValueError,
        match=(
            "cannot select an action for "
            "a terminal observation"
        ),
    ):
        agent.select_action(observation)