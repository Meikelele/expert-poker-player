import pytest

from expert_poker_player.policy_gradient import (
    PolicyStep,
    Trajectory,
)
from expert_poker_player.rl.actions import (
    ACTION_COUNT,
    action_to_index,
)
from expert_poker_player.uth import (
    Action,
)


def build_step(
    *,
    reward: float = 0.0,
) -> PolicyStep:
    return PolicyStep(
        state=(
            1.0,
            2.0,
            3.0,
        ),
        action_index=action_to_index(
            Action.CHECK
        ),
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


def test_policy_step_stores_decision() -> None:
    step = build_step(
        reward=2.5
    )

    assert step.action_index == action_to_index(
        Action.CHECK
    )

    assert step.reward == 2.5


def test_rejects_illegal_selected_action() -> None:
    with pytest.raises(
        ValueError,
        match="selected action must be legal",
    ):
        PolicyStep(
            state=(1.0,),
            action_index=action_to_index(
                Action.BET_1X
            ),
            action_mask=(
                True,
                True,
                True,
                False,
                False,
                False,
            ),
            reward=0.0,
        )


def test_rejects_invalid_action_mask_size() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "action_mask must match "
            "the action count"
        ),
    ):
        PolicyStep(
            state=(1.0,),
            action_index=0,
            action_mask=(
                True,
            ),
            reward=0.0,
        )


def test_rejects_mask_without_legal_action() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "action_mask must contain "
            "at least one legal action"
        ),
    ):
        PolicyStep(
            state=(1.0,),
            action_index=0,
            action_mask=tuple(
                False
                for _ in range(
                    ACTION_COUNT
                )
            ),
            reward=0.0,
        )


def test_trajectory_stores_episode() -> None:
    trajectory = Trajectory(
        steps=(
            build_step(),
            build_step(
                reward=3.5
            ),
        )
    )

    assert len(
        trajectory
    ) == 2

    assert trajectory.total_reward == 3.5


def test_rejects_empty_trajectory() -> None:
    with pytest.raises(
        ValueError,
        match="trajectory cannot be empty",
    ):
        Trajectory(
            steps=()
        )

@pytest.mark.parametrize(
    "reward",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_non_finite_reward(
    reward: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="reward must be finite",
    ):
        PolicyStep(
            state=(1.0,),
            action_index=0,
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

