import pytest
import torch

from expert_poker_player.dqn import (
    ACTION_COUNT,
    ACTION_ORDER,
    action_from_index,
    action_to_index,
    legal_action_mask,
    mask_q_values,
)
from expert_poker_player.uth import (
    Action,
    UTHGame,
)


def test_action_order_is_stable() -> None:
    assert ACTION_ORDER == (
        Action.CHECK,
        Action.BET_4X,
        Action.BET_3X,
        Action.BET_2X,
        Action.BET_1X,
        Action.FOLD,
    )

    assert ACTION_COUNT == 6


@pytest.mark.parametrize(
    "action",
    list(Action),
)
def test_action_index_round_trip(
    action: Action,
) -> None:
    index = action_to_index(
        action
    )

    assert action_from_index(
        index
    ) is action

def test_action_to_index_rejects_invalid_action() -> None:
    with pytest.raises(
        TypeError,
        match="action must be an instance of Action",
    ):
        action_to_index(
            "check"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "index",
    [
        -1,
        ACTION_COUNT,
    ],
)
def test_action_from_index_rejects_invalid_index(
    index: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="index must reference an available action",
    ):
        action_from_index(
            index
        )

def test_preflop_action_mask() -> None:
    game = UTHGame(
        seed=1
    )

    observation = game.reset()

    mask = legal_action_mask(
        observation
    )

    assert mask.tolist() == [  # pyright: ignore[reportUnknownMemberType]
        True,
        True,
        True,
        False,
        False,
        False,
    ]

def test_flop_action_mask() -> None:
    game = UTHGame(
        seed=1
    )

    game.reset()

    result = game.step(
        Action.CHECK
    )

    mask = legal_action_mask(
        result.observation
    )

    assert mask.tolist() == [  # pyright: ignore[reportUnknownMemberType]
        True,
        False,
        False,
        True,
        False,
        False,
    ]

def test_river_action_mask() -> None:
    game = UTHGame(
        seed=1
    )

    game.reset()

    game.step(
        Action.CHECK
    )

    result = game.step(
        Action.CHECK
    )

    mask = legal_action_mask(
        result.observation
    )

    assert mask.tolist() == [  # pyright: ignore[reportUnknownMemberType]
        False,
        False,
        False,
        False,
        True,
        True,
    ]

def test_illegal_action_cannot_win_argmax() -> None:
    game = UTHGame(
        seed=1
    )

    game.reset()

    game.step(
        Action.CHECK
    )

    result = game.step(
        Action.CHECK
    )

    q_values = torch.tensor(
        [
            1000.0,
            900.0,
            800.0,
            700.0,
            2.0,
            1.0,
        ]
    )

    action_mask = legal_action_mask(
        result.observation
    )

    masked = mask_q_values(
        q_values,
        action_mask,
    )

    action_index = int(
        torch.argmax(masked).item()
    )

    action = action_from_index(
        action_index
    )

    assert action is Action.BET_1X

def test_mask_q_values_replaces_illegal_values_with_negative_infinity() -> None:
    q_values = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
    )

    action_mask = torch.tensor(
        [
            True,
            False,
            True,
            False,
            False,
            False,
        ]
    )

    masked = mask_q_values(
        q_values,
        action_mask,
    )

    assert masked[0] == 1.0
    assert torch.isneginf(
        masked[1]
    )
    assert masked[2] == 3.0
    assert torch.isneginf(
        masked[3]
    )

def test_mask_q_values_supports_batches() -> None:
    q_values = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            [
                6.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        ]
    )

    action_mask = torch.tensor(
        [
            [
                True,
                True,
                True,
                False,
                False,
                False,
            ],
            [
                False,
                False,
                False,
                False,
                True,
                True,
            ],
        ]
    )

    masked = mask_q_values(
        q_values,
        action_mask,
    )

    assert int(
        torch.argmax(
            masked[0]
        ).item()
    ) == action_to_index(
        Action.BET_3X
    )

    assert int(
        torch.argmax(
            masked[1]
        ).item()
    ) == action_to_index(
        Action.BET_1X
    )

def test_rejects_terminal_observation() -> None:
    game = UTHGame(
        seed=1
    )

    game.reset()

    result = game.step(
        Action.BET_4X
    )

    assert result.observation.terminated

    with pytest.raises(
        ValueError,
        match=(
            "cannot build an action mask "
            "for a terminal observation"
        ),
    ):
        legal_action_mask(
            result.observation
        )


