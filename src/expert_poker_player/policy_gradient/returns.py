import math


def compute_discounted_returns(
    rewards: tuple[float, ...],
    *,
    gamma: float,
) -> tuple[float, ...]:
    """Oblicza zdyskontowany return dla każdego kroku epizodu."""

    if not isinstance(
        rewards,
        tuple,
    ):  # type: ignore
        raise TypeError(
            "rewards must be a tuple"
        )

    if not rewards:
        raise ValueError(
            "rewards cannot be empty"
        )

    normalized_rewards: list[float] = []

    for reward in rewards:
        if not isinstance(
            reward,
            (int, float),
        ):  # type: ignore
            raise TypeError(
                "rewards must contain numbers"
            )

        normalized_reward = float(
            reward
        )

        if not math.isfinite(
            normalized_reward
        ):
            raise ValueError(
                "rewards must be finite"
            )

        normalized_rewards.append(
            normalized_reward
        )

    if not isinstance(
        gamma,
        (int, float),
    ):  # type: ignore
        raise TypeError(
            "gamma must be a number"
        )

    gamma = float(
        gamma
    )

    if (
        not math.isfinite(gamma)
        or not 0.0 <= gamma <= 1.0
    ):
        raise ValueError(
            "gamma must be between 0 and 1"
        )

    returns = [
        0.0
        for _ in normalized_rewards
    ]

    running_return = 0.0

    for index in range(
        len(normalized_rewards) - 1,
        -1,
        -1,
    ):
        running_return = (
            normalized_rewards[index]
            + gamma * running_return
        )

        returns[index] = (
            running_return
        )

    return tuple(
        returns
    )