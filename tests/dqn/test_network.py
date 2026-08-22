import pytest
import torch
from torch import nn

from expert_poker_player.dqn import QNetwork
from expert_poker_player.state_representation import (
    FEATURE_STATE_SIZE,
    RAW_STATE_SIZE,
)
from expert_poker_player.uth import Action


@pytest.mark.parametrize(
    "input_size",
    [
        RAW_STATE_SIZE,
        FEATURE_STATE_SIZE,
    ],
)
def test_returns_q_value_for_each_action(
    input_size: int,
) -> None:
    network = QNetwork(
        input_size=input_size
    )

    state = torch.zeros(
        1,
        input_size,
    )

    q_values = network(state)

    assert q_values.shape == (
        1,
        len(Action),
    )


def test_supports_batches() -> None:
    network = QNetwork(
        input_size=RAW_STATE_SIZE
    )

    states = torch.zeros(
        32,
        RAW_STATE_SIZE,
    )

    q_values = network(states)

    assert q_values.shape == (
        32,
        len(Action),
    )


def test_uses_configured_hidden_sizes() -> None:
    network = QNetwork(
        input_size=RAW_STATE_SIZE,
        hidden_sizes=(128, 64),
    )

    linear_layers = [
        layer
        for layer in network.network
        if isinstance(layer, nn.Linear)
    ]

    assert linear_layers[0].in_features == RAW_STATE_SIZE
    assert linear_layers[0].out_features == 128

    assert linear_layers[1].in_features == 128
    assert linear_layers[1].out_features == 64

    assert linear_layers[2].in_features == 64
    assert linear_layers[2].out_features == len(Action)


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
        QNetwork(
            input_size=input_size
        )


def test_rejects_non_integer_input_size() -> None:
    with pytest.raises(
        TypeError,
        match="input_size must be an integer",
    ):
        QNetwork(
            input_size=373.0  # type: ignore[arg-type]
        )


def test_rejects_empty_hidden_sizes() -> None:
    with pytest.raises(
        ValueError,
        match="hidden_sizes cannot be empty",
    ):
        QNetwork(
            input_size=RAW_STATE_SIZE,
            hidden_sizes=(),
        )


def test_rejects_non_integer_hidden_size() -> None:
    with pytest.raises(
        TypeError,
        match="hidden_sizes must contain integers",
    ):
        QNetwork(
            input_size=RAW_STATE_SIZE,
            hidden_sizes=(256, 64.0),  # type: ignore[arg-type]
        )


def test_rejects_non_positive_hidden_size() -> None:
    with pytest.raises(
        ValueError,
        match="hidden_sizes must contain positive values",
    ):
        QNetwork(
            input_size=RAW_STATE_SIZE,
            hidden_sizes=(256, 0),
        )

def test_forward_preserves_batch_dimension() -> None:
    network = QNetwork(
        input_size=FEATURE_STATE_SIZE
    )

    states = torch.randn(
        7,
        FEATURE_STATE_SIZE,
    )

    result = network(states)

    assert result.ndim == 2
    assert result.shape[0] == 7
    assert result.shape[1] == len(Action)