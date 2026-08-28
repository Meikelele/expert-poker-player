import pytest
import torch

from expert_poker_player.rl.actions import (
    ACTION_COUNT,
    ACTION_ORDER,
    action_from_index,
    action_to_index,
    legal_action_mask,
    mask_action_values,
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
    assert action_from_index(
        action_to_index(action)
    ) is action


def test_preflop_action_mask() -> None:
    observation = UTHGame(
        seed=1
    ).reset()

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
    game = UTHGame(seed=1)

    game.reset()

    observation = game.step(
        Action.CHECK
    ).observation

    mask = legal_action_mask(
        observation
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
    game = UTHGame(seed=1)

    game.reset()
    game.step(Action.CHECK)

    observation = game.step(
        Action.CHECK
    ).observation

    mask = legal_action_mask(
        observation
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
    values = torch.tensor(
        [
            1000.0,
            900.0,
            800.0,
            700.0,
            2.0,
            1.0,
        ]
    )

    mask = torch.tensor(
        [
            False,
            False,
            False,
            False,
            True,
            True,
        ]
    )

    masked = mask_action_values(
        values,
        mask,
    )

    assert int(
        torch.argmax(masked).item()
    ) == action_to_index(
        Action.BET_1X
    )


def test_rejects_mask_without_legal_action() -> None:
    values = torch.zeros(
        ACTION_COUNT
    )

    mask = torch.zeros(
        ACTION_COUNT,
        dtype=torch.bool,
    )

    with pytest.raises(
        ValueError,
        match=(
            "action_mask must contain at least "
            "one legal action"
        ),
    ):
        mask_action_values(
            values,
            mask,
        )