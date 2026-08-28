import pytest

from expert_poker_player.policy_gradient import (
    compute_discounted_returns,
)


def test_computes_discounted_returns() -> None:
    returns = compute_discounted_returns(
        (
            1.0,
            2.0,
            3.0,
        ),
        gamma=0.5,
    )

    assert returns == pytest.approx(
        (
            2.75,
            3.5,
            3.0,
        )
    )

def test_terminal_reward_propagates_backwards() -> None:
    returns = compute_discounted_returns(
        (
            0.0,
            0.0,
            6.0,
        ),
        gamma=0.5,
    )

    assert returns == pytest.approx(
        (
            1.5,
            3.0,
            6.0,
        )
    )

def test_gamma_one_preserves_full_future_return() -> None:
    returns = compute_discounted_returns(
        (
            0.0,
            0.0,
            -4.5,
        ),
        gamma=1.0,
    )

    assert returns == pytest.approx(
        (
            -4.5,
            -4.5,
            -4.5,
        )
    )

def test_gamma_zero_uses_immediate_rewards_only() -> None:
    returns = compute_discounted_returns(
        (
            1.0,
            2.0,
            3.0,
        ),
        gamma=0.0,
    )

    assert returns == pytest.approx(
        (
            1.0,
            2.0,
            3.0,
        )
    )

def test_single_step_return_equals_reward() -> None:
    returns = compute_discounted_returns(
        (
            5.5,
        ),
        gamma=0.99,
    )

    assert returns == pytest.approx(
        (
            5.5,
        )
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
        compute_discounted_returns(
            (
                1.0,
            ),
            gamma=gamma,
        )


def test_rejects_empty_rewards() -> None:
    with pytest.raises(
        ValueError,
        match="rewards cannot be empty",
    ):
        compute_discounted_returns(
            (),
            gamma=0.99,
        )

