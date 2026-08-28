import pytest
import torch

from expert_poker_player.policy_gradient import (
    PolicyNetwork,
)
from expert_poker_player.rl.actions import (
    ACTION_COUNT,
)
from expert_poker_player.state_representation import (
    FEATURE_STATE_SIZE,
    RAW_STATE_SIZE,
)


@pytest.mark.parametrize(
    "input_size",
    [
        RAW_STATE_SIZE,
        FEATURE_STATE_SIZE,
    ],
)
def test_policy_network_outputs_logit_for_each_action(
    input_size: int,
) -> None:
    network = PolicyNetwork(
        input_size=input_size
    )

    state = torch.zeros(
        input_size
    )

    logits = network(
        state
    )

    assert logits.shape == (
        ACTION_COUNT,
    )


@pytest.mark.parametrize(
    "input_size",
    [
        RAW_STATE_SIZE,
        FEATURE_STATE_SIZE,
    ],
)
def test_policy_network_supports_batches(
    input_size: int,
) -> None:
    network = PolicyNetwork(
        input_size=input_size
    )

    states = torch.zeros(
        4,
        input_size,
    )

    logits = network(
        states
    )

    assert logits.shape == (
        4,
        ACTION_COUNT,
    )


def test_policy_network_supports_custom_hidden_sizes() -> None:
    network = PolicyNetwork(
        input_size=RAW_STATE_SIZE,
        hidden_sizes=(
            128,
            64,
        ),
    )

    assert network.hidden_sizes == (
        128,
        64,
    )

    assert network.input_size == RAW_STATE_SIZE


@pytest.mark.parametrize(
    "input_size",
    [
        0,
        -1,
    ],
)
def test_rejects_non_positive_input_size(
    input_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="input_size must be positive",
    ):
        PolicyNetwork(
            input_size=input_size
        )


def test_rejects_empty_hidden_sizes() -> None:
    with pytest.raises(
        ValueError,
        match="hidden_sizes cannot be empty",
    ):
        PolicyNetwork(
            input_size=RAW_STATE_SIZE,
            hidden_sizes=(),
        )


@pytest.mark.parametrize(
    "hidden_sizes",
    [
        (0,),
        (-1,),
        (32, 0),
    ],
)
def test_rejects_non_positive_hidden_size(
    hidden_sizes: tuple[int, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "hidden_sizes must contain positive values"
        ),
    ):
        PolicyNetwork(
            input_size=RAW_STATE_SIZE,
            hidden_sizes=hidden_sizes,
        )

def test_policy_network_returns_raw_logits() -> None:
    network = PolicyNetwork(
        input_size=RAW_STATE_SIZE,
        hidden_sizes=(16,),
    )

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
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ]
            )
        )

    logits = network(
        torch.zeros(
            RAW_STATE_SIZE
        )
    )

    assert torch.equal(
        logits,
        torch.tensor(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ]
        ),
    )