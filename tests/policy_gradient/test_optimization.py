import math

import pytest
import torch

from expert_poker_player.policy_gradient import (
    PolicyNetwork,
    PolicyStep,
    ReinforceOptimizer,
    Trajectory,
)


INPUT_SIZE = 3


def build_step(
    *,
    action_index: int = 0,
    reward: float = 2.0,
) -> PolicyStep:
    return PolicyStep(
        state=(
            1.0,
            2.0,
            3.0,
        ),
        action_index=action_index,
        action_mask=(
            True,
            True,
            True,
            False,
            False,
            False,
        ),
        reward=reward,
    )


def zero_network(
    network: PolicyNetwork,
) -> None:
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()

def test_loss_matches_single_step_reinforce_objective() -> None:
    network = PolicyNetwork(
        input_size=INPUT_SIZE,
        hidden_sizes=(4,),
    )

    zero_network(
        network
    )

    optimizer = ReinforceOptimizer(
        policy_network=network,
        learning_rate=1e-3,
        gamma=0.99,
    )

    trajectory = Trajectory(
        steps=(
            build_step(
                reward=2.0
            ),
        )
    )

    loss = optimizer.optimize(
        trajectory
    )

    assert loss == pytest.approx(
        2.0 * math.log(3.0),
        rel=1e-5,
    )

def test_loss_uses_probability_of_taken_action() -> None:
    network = PolicyNetwork(
        input_size=INPUT_SIZE,
        hidden_sizes=(4,),
    )

    zero_network(
        network
    )

    with torch.no_grad():
        output_layer = (
            network.network[-1]
        )

        assert isinstance(
            output_layer,
            torch.nn.Linear,
        )

        output_layer.bias.copy_(
            torch.tensor(
                [
                    0.0,
                    math.log(2.0),
                    0.0,
                    1000.0,
                    900.0,
                    800.0,
                ]
            )
        )

    optimizer = ReinforceOptimizer(
        policy_network=network,
        learning_rate=1e-3,
        gamma=1.0,
    )

    trajectory = Trajectory(
        steps=(
            build_step(
                action_index=1,
                reward=2.0,
            ),
        )
    )

    loss = optimizer.optimize(
        trajectory
    )

    assert loss == pytest.approx(
        2.0 * math.log(2.0),
        rel=1e-5,
    )

def test_optimization_updates_policy_network() -> None:
    network = PolicyNetwork(
        input_size=INPUT_SIZE,
        hidden_sizes=(4,),
    )

    optimizer = ReinforceOptimizer(
        policy_network=network,
        learning_rate=1e-2,
        gamma=0.99,
    )

    before = [
        parameter.detach().clone()
        for parameter in network.parameters()
    ]

    optimizer.optimize(
        Trajectory(
            steps=(
                build_step(),
            )
        )
    )

    after = list(
        network.parameters()
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

def test_zero_return_does_not_update_policy_network() -> None:
    network = PolicyNetwork(
        input_size=INPUT_SIZE,
        hidden_sizes=(4,),
    )

    optimizer = ReinforceOptimizer(
        policy_network=network,
        learning_rate=1e-2,
        gamma=0.99,
    )

    before = [
        parameter.detach().clone()
        for parameter in network.parameters()
    ]

    optimizer.optimize(
        Trajectory(
            steps=(
                build_step(
                    reward=0.0
                ),
            )
        )
    )

    after = list(
        network.parameters()
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

def test_loss_uses_discounted_returns() -> None:
    network = PolicyNetwork(
        input_size=INPUT_SIZE,
        hidden_sizes=(4,),
    )

    zero_network(
        network
    )

    optimizer = ReinforceOptimizer(
        policy_network=network,
        learning_rate=1e-3,
        gamma=0.5,
    )

    trajectory = Trajectory(
        steps=(
            build_step(
                reward=0.0
            ),
            build_step(
                reward=2.0
            ),
        )
    )

    loss = optimizer.optimize(
        trajectory
    )

    assert loss == pytest.approx(
        3.0 * math.log(3.0),
        rel=1e-5,
    )

def test_rejects_state_incompatible_with_policy_network() -> None:
    network = PolicyNetwork(
        input_size=4,
        hidden_sizes=(4,),
    )

    optimizer = ReinforceOptimizer(
        policy_network=network,
        learning_rate=1e-3,
        gamma=0.99,
    )

    with pytest.raises(
        ValueError,
        match=(
            "state size must match "
            "policy network input size"
        ),
    ):
        optimizer.optimize(
            Trajectory(
                steps=(
                    build_step(),
                )
            )
        )

