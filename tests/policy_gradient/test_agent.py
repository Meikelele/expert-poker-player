import pytest
import torch

from expert_poker_player.agents import (
    Agent,
)
from expert_poker_player.policy_gradient import (
    PolicyGradientAgent,
    PolicyNetwork,
)
from expert_poker_player.state_representation import (
    FEATURE_STATE_SIZE,
    RAW_STATE_SIZE,
    FeatureStateEncoder,
    RawStateEncoder,
    StateEncoder,
)
from expert_poker_player.uth import (
    Action,
    UTHGame,
)

def test_policy_gradient_agent_satisfies_agent_protocol() -> None:
    agent = PolicyGradientAgent(
        policy_network=PolicyNetwork(
            input_size=RAW_STATE_SIZE
        ),
        state_encoder=RawStateEncoder(),
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
    agent = PolicyGradientAgent(
        policy_network=PolicyNetwork(
            input_size=input_size
        ),
        state_encoder=encoder,
    )

    observation = UTHGame(
        seed=1
    ).reset()

    action = agent.select_action(
        observation
    )

    assert action in observation.legal_actions

def set_output_biases(
    network: PolicyNetwork,
    values: tuple[float, ...],
) -> None:
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()

        output_layer = network.network[-1]

        assert isinstance(
            output_layer,
            torch.nn.Linear,
        )

        output_layer.bias.copy_(
            torch.tensor(
                values
            )
        )

def test_deterministic_agent_selects_best_legal_action() -> None:
    network = PolicyNetwork(
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

    game = UTHGame(
        seed=1
    )

    game.reset()
    game.step(
        Action.CHECK
    )

    observation = game.step(
        Action.CHECK
    ).observation

    agent = PolicyGradientAgent(
        policy_network=network,
        state_encoder=RawStateEncoder(),
        deterministic=True,
        seed=1,
    )

    action = agent.select_action(
        observation
    )

    assert action is Action.BET_1X

def test_stochastic_agent_selects_only_legal_actions() -> None:
    agent = PolicyGradientAgent(
        policy_network=PolicyNetwork(
            input_size=RAW_STATE_SIZE
        ),
        state_encoder=RawStateEncoder(),
        deterministic=False,
        seed=42,
    )

    observation = UTHGame(
        seed=1
    ).reset()

    selected = {
        agent.select_action(
            observation
        )
        for _ in range(100)
    }

    assert selected <= observation.legal_actions

def test_same_seed_reproduces_sampling_sequence() -> None:
    torch.manual_seed(1)  # type: ignore

    network = PolicyNetwork(
        input_size=RAW_STATE_SIZE
    )

    observation = UTHGame(
        seed=1
    ).reset()

    first = PolicyGradientAgent(
        policy_network=network,
        state_encoder=RawStateEncoder(),
        deterministic=False,
        seed=123,
    )

    second = PolicyGradientAgent(
        policy_network=network,
        state_encoder=RawStateEncoder(),
        deterministic=False,
        seed=123,
    )

    first_actions = [
        first.select_action(
            observation
        )
        for _ in range(30)
    ]

    second_actions = [
        second.select_action(
            observation
        )
        for _ in range(30)
    ]

    assert first_actions == second_actions

def test_deterministic_mode_can_be_updated() -> None:
    agent = PolicyGradientAgent(
        policy_network=PolicyNetwork(
            input_size=RAW_STATE_SIZE
        ),
        state_encoder=RawStateEncoder(),
        deterministic=False,
    )

    agent.deterministic = True

    assert agent.deterministic

def test_rejects_terminal_observation() -> None:
    game = UTHGame(
        seed=1
    )

    game.reset()

    result = game.step(
        Action.BET_4X
    )

    agent = PolicyGradientAgent(
        policy_network=PolicyNetwork(
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

def test_rejects_mismatched_network_input_size() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "policy_network input size must match "
            "state encoder output size"
        ),
    ):
        PolicyGradientAgent(
            policy_network=PolicyNetwork(
                input_size=FEATURE_STATE_SIZE
            ),
            state_encoder=RawStateEncoder(),
        )

