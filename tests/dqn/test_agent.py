import pytest
import torch

from expert_poker_player.agents import Agent
from expert_poker_player.dqn import (
    DQNAgent,
    QNetwork,
)
from expert_poker_player.state_representation import (
    RawStateEncoder,
    FeatureStateEncoder,
    StateEncoder,
    RAW_STATE_SIZE,
    FEATURE_STATE_SIZE,
)
from expert_poker_player.uth import (
    Action,
    UTHGame,
)

def test_dqn_agent_satisfies_agent_protocol() -> None:
    agent = DQNAgent(
        q_network=QNetwork(
            input_size=RAW_STATE_SIZE
        ),
        state_encoder=RawStateEncoder(),
        epsilon=0.0,
        seed=1,
    )

    assert isinstance(
        agent,
        Agent,
    )

@pytest.mark.parametrize(
    (
        "encoder",
        "input_size",
    ),
    [
        (
            RawStateEncoder(),
            RAW_STATE_SIZE,
        ),
        (
            FeatureStateEncoder(),
            FEATURE_STATE_SIZE,
        ),
    ],
)
def test_supports_state_encoders(
    encoder: StateEncoder,
    input_size: int,
) -> None:
    agent = DQNAgent(
        q_network=QNetwork(
            input_size=input_size
        ),
        state_encoder=encoder,
        epsilon=0.0,
        seed=1,
    )

    observation = UTHGame(
        seed=1
    ).reset()

    action = agent.select_action(
        observation
    )

    assert action in observation.legal_actions

def test_rejects_mismatched_network_input_size() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "q_network input size must match "
            "state encoder output size"
        ),
    ):
        DQNAgent(
            q_network=QNetwork(
                input_size=FEATURE_STATE_SIZE
            ),
            state_encoder=RawStateEncoder(),
        )

def set_output_biases(
    network: QNetwork,
    values: tuple[float, ...],
) -> None:
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()

        last_layer = network.network[-1]

        assert isinstance(
            last_layer,
            torch.nn.Linear,
        )

        last_layer.bias.copy_(
            torch.tensor(values)
        )

def test_epsilon_zero_selects_best_legal_action() -> None:
    network = QNetwork(
        input_size=RAW_STATE_SIZE
    )

    set_output_biases(
        network,
        (
            1000.0,
            900.0,
            800.0,
            700.0,
            2.0,
            1.0,
        ),
    )

    agent = DQNAgent(
        q_network=network,
        state_encoder=RawStateEncoder(),
        epsilon=0.0,
        seed=1,
    )

    game = UTHGame(
        seed=1
    )

    game.reset()
    game.step(Action.CHECK)

    observation = game.step(
        Action.CHECK
    ).observation

    action = agent.select_action(
        observation
    )

    assert action is Action.BET_1X

def test_epsilon_one_selects_only_legal_actions() -> None:
    agent = DQNAgent(
        q_network=QNetwork(
            input_size=RAW_STATE_SIZE
        ),
        state_encoder=RawStateEncoder(),
        epsilon=1.0,
        seed=123,
    )

    game = UTHGame(
        seed=1
    )

    observation = game.reset()

    selected = {
        agent.select_action(
            observation
        )
        for _ in range(100)
    }

    assert selected <= observation.legal_actions
    assert len(selected) > 1

def test_same_seed_produces_same_exploration_sequence() -> None:
    observation = UTHGame(
        seed=1
    ).reset()

    first = DQNAgent(
        q_network=QNetwork(
            input_size=RAW_STATE_SIZE
        ),
        state_encoder=RawStateEncoder(),
        epsilon=1.0,
        seed=42,
    )

    second = DQNAgent(
        q_network=QNetwork(
            input_size=RAW_STATE_SIZE
        ),
        state_encoder=RawStateEncoder(),
        epsilon=1.0,
        seed=42,
    )

    first_actions = [
        first.select_action(
            observation
        )
        for _ in range(20)
    ]

    second_actions = [
        second.select_action(
            observation
        )
        for _ in range(20)
    ]

    assert first_actions == second_actions

@pytest.mark.parametrize(
    "epsilon",
    [
        -0.01,
        1.01,
    ],
)
def test_rejects_invalid_epsilon(
    epsilon: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="epsilon must be between 0 and 1",
    ):
        DQNAgent(
            q_network=QNetwork(
                input_size=RAW_STATE_SIZE
            ),
            state_encoder=RawStateEncoder(),
            epsilon=epsilon,
        )

def test_epsilon_can_be_updated() -> None:
    agent = DQNAgent(
        q_network=QNetwork(
            input_size=RAW_STATE_SIZE
        ),
        state_encoder=RawStateEncoder(),
        epsilon=1.0,
    )

    agent.epsilon = 0.25

    assert agent.epsilon == 0.25

def test_rejects_terminal_observation() -> None:
    game = UTHGame(
        seed=1
    )

    game.reset()

    result = game.step(
        Action.BET_4X
    )

    agent = DQNAgent(
        q_network=QNetwork(
            input_size=RAW_STATE_SIZE
        ),
        state_encoder=RawStateEncoder(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "cannot select an action "
            "for a terminal observation"
        ),
    ):
        agent.select_action(
            result.observation
        )

