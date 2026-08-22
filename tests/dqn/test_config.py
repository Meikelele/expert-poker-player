import json

import pytest

from expert_poker_player.dqn import (
    DQNConfig,
)


def test_default_config_is_valid() -> None:
    config = DQNConfig()

    assert config.learning_rate > 0.0
    assert 0.0 <= config.gamma <= 1.0
    assert config.batch_size > 0
    assert config.replay_capacity >= config.batch_size
    assert config.epsilon_start >= config.epsilon_end

def test_epsilon_schedule_starts_at_configured_value() -> None:
    config = DQNConfig(
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
    )

    assert config.epsilon_at_step(
        0
    ) == pytest.approx(
        1.0
    )


def test_epsilon_schedule_reaches_end_value() -> None:
    config = DQNConfig(
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
    )

    assert config.epsilon_at_step(
        100
    ) == pytest.approx(
        0.1
    )

    assert config.epsilon_at_step(
        1_000
    ) == pytest.approx(
        0.1
    )

def test_epsilon_schedule_is_linear() -> None:
    config = DQNConfig(
        epsilon_start=1.0,
        epsilon_end=0.0,
        epsilon_decay_steps=100,
    )

    assert config.epsilon_at_step(
        50
    ) == pytest.approx(
        0.5
    )

def test_epsilon_schedule_is_monotonically_decreasing() -> None:
    config = DQNConfig(
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
    )

    values = [
        config.epsilon_at_step(
            step
        )
        for step in range(
            101
        )
    ]

    assert all(
        left >= right
        for left, right in zip(
            values,
            values[1:],
        )
    )

def test_epsilon_schedule_supports_constant_value() -> None:
    config = DQNConfig(
        epsilon_start=0.2,
        epsilon_end=0.2,
        epsilon_decay_steps=100,
    )

    assert config.epsilon_at_step(
        0
    ) == pytest.approx(
        0.2
    )

    assert config.epsilon_at_step(
        100
    ) == pytest.approx(
        0.2
    )

@pytest.mark.parametrize(
    "gamma",
    [
        -0.01,
        1.01,
    ],
)
def test_rejects_invalid_gamma(
    gamma: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="gamma must be between 0 and 1",
    ):
        DQNConfig(
            gamma=gamma
        )


def test_rejects_replay_capacity_smaller_than_batch() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "replay_capacity must be at least "
            "batch_size"
        ),
    ):
        DQNConfig(
            replay_capacity=31,
            batch_size=32,
        )


def test_rejects_epsilon_end_above_start() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "epsilon_end cannot exceed "
            "epsilon_start"
        ),
    ):
        DQNConfig(
            epsilon_start=0.2,
            epsilon_end=0.5,
        )


def test_rejects_negative_schedule_step() -> None:
    config = DQNConfig()

    with pytest.raises(
        ValueError,
        match="step cannot be negative",
    ):
        config.epsilon_at_step(
            -1
        )

def test_config_is_json_serializable() -> None:
    config = DQNConfig(
        seed=123
    )

    serialized = json.dumps(
        config.to_dict()
    )

    restored = json.loads(
        serialized
    )

    assert restored["seed"] == 123
    assert restored["hidden_sizes"] == [
        256,
        256,
    ]

