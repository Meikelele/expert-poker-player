import pytest
import torch

from expert_poker_player.dqn import (
    QNetwork,
    Transition,
    compute_bellman_targets,
)


INPUT_SIZE = 3


def set_constant_q_values(
    network: QNetwork,
    q_values: tuple[float, ...],
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
                q_values,
                dtype=torch.float32,
            )
        )

def test_terminal_target_equals_reward() -> None:
    network = QNetwork(
        input_size=INPUT_SIZE
    )

    transition = Transition(
        state=(1.0, 2.0, 3.0),
        action_index=0,
        reward=-6.0,
        next_state=None,
        terminated=True,
        next_action_mask=None,
    )

    targets = compute_bellman_targets(
        (
            transition,
        ),
        target_network=network,
        gamma=0.99,
    )

    assert targets.tolist() == pytest.approx( # type: ignore
        [
            -6.0,
        ]
    )

def test_non_terminal_target_bootstraps_from_target_network() -> None:
    network = QNetwork(
        input_size=INPUT_SIZE
    )

    set_constant_q_values(
        network,
        (
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ),
    )

    transition = Transition(
        state=(1.0, 2.0, 3.0),
        action_index=0,
        reward=0.5,
        next_state=(4.0, 5.0, 6.0),
        terminated=False,
        next_action_mask=(
            True,
            True,
            True,
            False,
            False,
            False,
        ),
    )

    targets = compute_bellman_targets(
        (
            transition,
        ),
        target_network=network,
        gamma=0.9,
    )

    assert targets.item() == pytest.approx(
        0.5
        + 0.9
        * 3.0
    )

def test_illegal_action_cannot_define_bellman_target() -> None:
    network = QNetwork(
        input_size=INPUT_SIZE
    )

    set_constant_q_values(
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

    transition = Transition(
        state=(1.0, 2.0, 3.0),
        action_index=0,
        reward=0.0,
        next_state=(4.0, 5.0, 6.0),
        terminated=False,
        next_action_mask=(
            False,
            False,
            False,
            False,
            True,
            True,
        ),
    )

    targets = compute_bellman_targets(
        (
            transition,
        ),
        target_network=network,
        gamma=0.5,
    )

    assert targets.item() == pytest.approx(
        1.0
    )

def test_mixed_batch_handles_terminal_and_non_terminal_transitions() -> None:
    network = QNetwork(
        input_size=INPUT_SIZE
    )

    set_constant_q_values(
        network,
        (
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ),
    )

    terminal = Transition(
        state=(1.0, 1.0, 1.0),
        action_index=0,
        reward=-2.0,
        next_state=None,
        terminated=True,
        next_action_mask=None,
    )

    non_terminal = Transition(
        state=(2.0, 2.0, 2.0),
        action_index=0,
        reward=0.25,
        next_state=(3.0, 3.0, 3.0),
        terminated=False,
        next_action_mask=(
            False,
            False,
            False,
            False,
            True,
            True,
        ),
    )

    targets = compute_bellman_targets(
        (
            terminal,
            non_terminal,
        ),
        target_network=network,
        gamma=0.9,
    )

    assert targets.tolist() == pytest.approx( # type: ignore
        [
            -2.0,
            0.25 + 0.9 * 6.0,
        ]
    )

def test_gamma_zero_removes_bootstrap_term() -> None:
    network = QNetwork(
        input_size=INPUT_SIZE
    )

    set_constant_q_values(
        network,
        (
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ),
    )

    transition = Transition(
        state=(1.0, 2.0, 3.0),
        action_index=0,
        reward=2.5,
        next_state=(4.0, 5.0, 6.0),
        terminated=False,
        next_action_mask=(
            True,
            True,
            True,
            False,
            False,
            False,
        ),
    )

    targets = compute_bellman_targets(
        (
            transition,
        ),
        target_network=network,
        gamma=0.0,
    )

    assert targets.item() == pytest.approx(
        2.5
    )

def test_bellman_targets_do_not_require_gradients() -> None:
    network = QNetwork(
        input_size=INPUT_SIZE
    )

    transition = Transition(
        state=(1.0, 2.0, 3.0),
        action_index=0,
        reward=0.0,
        next_state=(4.0, 5.0, 6.0),
        terminated=False,
        next_action_mask=(
            True,
            True,
            True,
            False,
            False,
            False,
        ),
    )

    targets = compute_bellman_targets(
        (
            transition,
        ),
        target_network=network,
        gamma=0.99,
    )

    assert targets.requires_grad is False

