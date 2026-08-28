import json

import pytest

from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
)


def test_default_configuration() -> None:
    config = PolicyGradientConfig()

    assert config.learning_rate == 1e-3
    assert config.gamma == 0.99

    assert config.hidden_sizes == (
        256,
        256,
    )

    assert config.training_episodes == 10_000
    assert config.seed == 42


def test_supports_custom_configuration() -> None:
    config = PolicyGradientConfig(
        learning_rate=5e-4,
        gamma=1.0,
        hidden_sizes=(
            128,
            64,
        ),
        training_episodes=5_000,
        seed=123,
    )

    assert config.learning_rate == 5e-4
    assert config.gamma == 1.0

    assert config.hidden_sizes == (
        128,
        64,
    )

    assert config.training_episodes == 5_000
    assert config.seed == 123


@pytest.mark.parametrize(
    "learning_rate",
    [
        0.0,
        -1e-3,
        float("nan"),
        float("inf"),
    ],
)
def test_rejects_invalid_learning_rate(
    learning_rate: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "learning_rate must be positive "
            "and finite"
        ),
    ):
        PolicyGradientConfig(
            learning_rate=learning_rate
        )


@pytest.mark.parametrize(
    "gamma",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
    ],
)
def test_rejects_invalid_gamma(
    gamma: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="gamma must be between 0 and 1",
    ):
        PolicyGradientConfig(
            gamma=gamma
        )


def test_rejects_empty_hidden_sizes() -> None:
    with pytest.raises(
        ValueError,
        match="hidden_sizes cannot be empty",
    ):
        PolicyGradientConfig(
            hidden_sizes=()
        )


@pytest.mark.parametrize(
    "hidden_sizes",
    [
        (0,),
        (-1,),
        (32, 0),
    ],
)
def test_rejects_non_positive_hidden_sizes(
    hidden_sizes: tuple[int, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "hidden_sizes must contain "
            "positive values"
        ),
    ):
        PolicyGradientConfig(
            hidden_sizes=hidden_sizes
        )


@pytest.mark.parametrize(
    "training_episodes",
    [
        0,
        -1,
    ],
)
def test_rejects_non_positive_training_episodes(
    training_episodes: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "training_episodes must be positive"
        ),
    ):
        PolicyGradientConfig(
            training_episodes=training_episodes
        )


def test_to_dict_is_json_serializable() -> None:
    config = PolicyGradientConfig(
        hidden_sizes=(
            128,
            64,
        ),
        seed=123,
    )

    values = config.to_dict()

    assert values["hidden_sizes"] == [
        128,
        64,
    ]

    serialized = json.dumps(
        values
    )

    assert isinstance(
        serialized,
        str,
    )