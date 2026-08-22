import pytest

from expert_poker_player.dqn import (
    ACTION_COUNT,
    ReplayBuffer,
    Transition,
)


def build_transition(
    identifier: float = 0.0,
    *,
    terminated: bool = False,
) -> Transition:
    state = (
        identifier,
        1.0,
        2.0,
    )

    if terminated:
        return Transition(
            state=state,
            action_index=0,
            reward=identifier,
            next_state=None,
            terminated=True,
            next_action_mask=None,
        )

    return Transition(
        state=state,
        action_index=0,
        reward=identifier,
        next_state=(
            identifier + 1.0,
            1.0,
            2.0,
        ),
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

def test_stores_transition() -> None:
    buffer = ReplayBuffer(
        capacity=10,
        seed=1,
    )

    transition = build_transition()

    buffer.add(
        transition
    )

    assert len(buffer) == 1

    sampled = buffer.sample(
        1
    )

    assert sampled == (
        transition,
    )

def test_discards_oldest_transition_when_full() -> None:
    buffer = ReplayBuffer(
        capacity=3,
        seed=1,
    )

    for identifier in range(4):
        buffer.add(
            build_transition(
                float(identifier)
            )
        )

    assert len(buffer) == 3

    sampled = buffer.sample(
        3
    )

    identifiers = {
        transition.state[0]
        for transition in sampled
    }

    assert identifiers == {
        1.0,
        2.0,
        3.0,
    }

def test_same_seed_produces_same_samples() -> None:
    first = ReplayBuffer(
        capacity=10,
        seed=42,
    )

    second = ReplayBuffer(
        capacity=10,
        seed=42,
    )

    transitions = [
        build_transition(
            float(identifier)
        )
        for identifier in range(10)
    ]

    for transition in transitions:
        first.add(
            transition
        )

        second.add(
            transition
        )

    first_samples = [
        first.sample(4)
        for _ in range(5)
    ]

    second_samples = [
        second.sample(4)
        for _ in range(5)
    ]

    assert first_samples == second_samples

def test_supports_terminal_transition() -> None:
    transition = build_transition(
        terminated=True
    )

    assert transition.terminated is True
    assert transition.next_state is None
    assert transition.next_action_mask is None

def test_non_terminal_transition_requires_next_state() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "non-terminal transition requires "
            "next_state"
        ),
    ):
        Transition(
            state=(1.0,),
            action_index=0,
            reward=0.0,
            next_state=None,
            terminated=False,
            next_action_mask=(
                True,
            )
            * ACTION_COUNT,
        )

def test_non_terminal_transition_requires_next_action_mask() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "non-terminal transition requires "
            "next_action_mask"
        ),
    ):
        Transition(
            state=(1.0,),
            action_index=0,
            reward=0.0,
            next_state=(2.0,),
            terminated=False,
            next_action_mask=None,
        )

def test_terminal_transition_rejects_next_state() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "terminal transition cannot "
            "have next_state"
        ),
    ):
        Transition(
            state=(1.0,),
            action_index=0,
            reward=-2.0,
            next_state=(2.0,),
            terminated=True,
            next_action_mask=None,
        )

def test_rejects_invalid_next_action_mask_size() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "next_action_mask must match "
            "the action count"
        ),
    ):
        Transition(
            state=(1.0,),
            action_index=0,
            reward=0.0,
            next_state=(2.0,),
            terminated=False,
            next_action_mask=(
                True,
                False,
            ),
        )

def test_rejects_batch_larger_than_buffer() -> None:
    buffer = ReplayBuffer(
        capacity=10,
        seed=1,
    )

    buffer.add(
        build_transition()
    )

    with pytest.raises(
        ValueError,
        match=(
            "batch_size cannot exceed "
            "the number of stored transitions"
        ),
    ):
        buffer.sample(
            2
        )
