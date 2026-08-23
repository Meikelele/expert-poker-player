import pytest
import torch

from expert_poker_player.dqn import (
    DQNOptimizer,
    QNetwork,
    Transition,
)


INPUT_SIZE = 3


def build_terminal_transition() -> Transition:
    return Transition(
        state=(1.0, 2.0, 3.0),
        action_index=0,
        reward=10.0,
        next_state=None,
        terminated=True,
        next_action_mask=None,
    )

def test_optimization_updates_policy_network() -> None:
    torch.manual_seed(1) # type: ignore

    policy = QNetwork(
        input_size=INPUT_SIZE
    )

    target = QNetwork(
        input_size=INPUT_SIZE
    )

    optimizer = DQNOptimizer(
        policy_network=policy,
        target_network=target,
        learning_rate=1e-2,
        gamma=0.99,
    )

    before = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]

    optimizer.optimize(
        (
            build_terminal_transition(),
        )
    )

    after = list(
        policy.parameters()
    )

    assert any(
        not torch.equal(
            old,
            new,
        )
        for old, new in zip(
            before,
            after,
        )
    )

def test_optimization_does_not_update_target_network() -> None:
    torch.manual_seed(1) # type: ignore

    policy = QNetwork(
        input_size=INPUT_SIZE
    )

    target = QNetwork(
        input_size=INPUT_SIZE
    )

    optimizer = DQNOptimizer(
        policy_network=policy,
        target_network=target,
        learning_rate=1e-2,
        gamma=0.99,
    )

    before = [
        parameter.detach().clone()
        for parameter in target.parameters()
    ]

    optimizer.optimize(
        (
            build_terminal_transition(),
        )
    )

    after = list(
        target.parameters()
    )

    assert all(
        torch.equal(
            old,
            new,
        )
        for old, new in zip(
            before,
            after,
        )
    )

def test_sync_target_network_copies_policy_parameters() -> None:
    torch.manual_seed(1) # type: ignore

    policy = QNetwork(
        input_size=INPUT_SIZE
    )

    target = QNetwork(
        input_size=INPUT_SIZE
    )

    optimizer = DQNOptimizer(
        policy_network=policy,
        target_network=target,
        learning_rate=1e-3,
        gamma=0.99,
    )

    optimizer.sync_target_network()

    for policy_parameter, target_parameter in zip(
        policy.parameters(),
        target.parameters(),
    ):
        assert torch.equal(
            policy_parameter,
            target_parameter,
        )

def test_target_remains_stable_until_explicit_sync() -> None:
    torch.manual_seed(1) # type: ignore

    policy = QNetwork(
        input_size=INPUT_SIZE
    )

    target = QNetwork(
        input_size=INPUT_SIZE
    )

    optimizer = DQNOptimizer(
        policy_network=policy,
        target_network=target,
        learning_rate=1e-2,
        gamma=0.99,
    )

    optimizer.sync_target_network()

    optimizer.optimize(
        (
            build_terminal_transition(),
        )
    )

    assert any(
        not torch.equal(
            policy_parameter,
            target_parameter,
        )
        for policy_parameter, target_parameter in zip(
            policy.parameters(),
            target.parameters(),
        )
    )

    optimizer.sync_target_network()

    assert all(
        torch.equal(
            policy_parameter,
            target_parameter,
        )
        for policy_parameter, target_parameter in zip(
            policy.parameters(),
            target.parameters(),
        )
    )

def test_optimization_returns_finite_loss() -> None:
    policy = QNetwork(
        input_size=INPUT_SIZE
    )

    target = QNetwork(
        input_size=INPUT_SIZE
    )

    optimizer = DQNOptimizer(
        policy_network=policy,
        target_network=target,
        learning_rate=1e-3,
        gamma=0.99,
    )

    loss = optimizer.optimize(
        (
            build_terminal_transition(),
        )
    )

    assert isinstance(
        loss,
        float,
    )

    assert torch.isfinite(
        torch.tensor(loss)
    )

def test_loss_uses_q_value_for_taken_action() -> None:
    policy = QNetwork(
        input_size=INPUT_SIZE
    )

    target = QNetwork(
        input_size=INPUT_SIZE
    )

    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()

        output_layer = policy.network[-1]

        assert isinstance(
            output_layer,
            torch.nn.Linear,
        )

        output_layer.bias.copy_(
            torch.tensor(
                [
                    1.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                ]
            )
        )

    optimizer = DQNOptimizer(
        policy_network=policy,
        target_network=target,
        learning_rate=1e-3,
        gamma=0.99,
    )

    transition = Transition(
        state=(1.0, 2.0, 3.0),
        action_index=1,
        reward=20.0,
        next_state=None,
        terminated=True,
        next_action_mask=None,
    )

    loss = optimizer.optimize(
        (
            transition,
        )
    )

    assert loss == 0.0

def test_loss_matches_smooth_l1_for_known_nonzero_residual() -> None:
    """Odróżnia SmoothL1Loss od MSE/L1 na znanym, niezerowym residuum."""

    policy = QNetwork(
        input_size=INPUT_SIZE
    )

    target = QNetwork(
        input_size=INPUT_SIZE
    )

    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()

        output_layer = policy.network[-1]

        assert isinstance(
            output_layer,
            torch.nn.Linear,
        )

        output_layer.bias.copy_(
            torch.tensor(
                [
                    1.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                ]
            )
        )

    optimizer = DQNOptimizer(
        policy_network=policy,
        target_network=target,
        learning_rate=1e-3,
        gamma=0.99,
    )

    transition = Transition(
        state=(1.0, 2.0, 3.0),
        action_index=1,
        reward=18.0,
        next_state=None,
        terminated=True,
        next_action_mask=None,
    )

    loss = optimizer.optimize(
        (
            transition,
        )
    )

    # policy Q for action 1 == 20.0, terminal target == reward == 18.0
    # residual = |20.0 - 18.0| = 2.0 >= beta(1.0),
    # so SmoothL1Loss == |residual| - 0.5 * beta == 1.5.
    # MSELoss would give 4.0, L1Loss would give 2.0 -- both distinguishable.
    assert loss == pytest.approx(1.5)


