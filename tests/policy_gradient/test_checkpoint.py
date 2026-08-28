from pathlib import Path

import pytest
import torch

from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
    PolicyNetwork,
    load_policy_gradient_checkpoint,
    save_policy_gradient_checkpoint,
)
from expert_poker_player.rewards import (
    RewardType,
)
from expert_poker_player.state_representation import (
    FEATURE_STATE_SIZE,
    RAW_STATE_SIZE,
    StateRepresentation,
)


def test_checkpoint_restores_model_metadata(
    tmp_path: Path,
) -> None:
    config = PolicyGradientConfig(
        hidden_sizes=(
            32,
            16,
        ),
        seed=123,
    )

    network = PolicyNetwork(
        input_size=RAW_STATE_SIZE,
        hidden_sizes=config.hidden_sizes,
    )

    path = tmp_path / "policy.pt"

    save_policy_gradient_checkpoint(
        path,
        policy_network=network,
        state_representation=(
            StateRepresentation.RAW
        ),
        reward_type=(
            RewardType.NET_PROFIT
        ),
        config=config,
    )

    loaded = (
        load_policy_gradient_checkpoint(
            path
        )
    )

    assert (
        loaded.state_representation
        is StateRepresentation.RAW
    )

    assert (
        loaded.reward_type
        is RewardType.NET_PROFIT
    )

    assert loaded.training_seed == 123

    assert (
        loaded.config.hidden_sizes
        == (
            32,
            16,
        )
    )

    assert (
        loaded.policy_network.input_size
        == RAW_STATE_SIZE
    )

def test_restored_model_produces_same_logits(
    tmp_path: Path,
) -> None:
    torch.manual_seed(  # pyright: ignore[reportUnknownMemberType]
        1
    )

    config = PolicyGradientConfig(
        hidden_sizes=(16,),
    )

    network = PolicyNetwork(
        input_size=RAW_STATE_SIZE,
        hidden_sizes=config.hidden_sizes,
    )

    state = torch.randn(
        1,
        RAW_STATE_SIZE,
    )

    with torch.no_grad():
        expected = network(
            state
        ).clone()

    path = tmp_path / "policy.pt"

    save_policy_gradient_checkpoint(
        path,
        policy_network=network,
        state_representation=(
            StateRepresentation.RAW
        ),
        reward_type=(
            RewardType.STAKE_SCALED_NET_PROFIT
        ),
        config=config,
    )

    loaded = (
        load_policy_gradient_checkpoint(
            path
        )
    )

    with torch.no_grad():
        actual = (
            loaded.policy_network(
                state
            )
        )

    assert torch.equal(
        expected,
        actual,
    )

def test_checkpoint_supports_feature_state(
    tmp_path: Path,
) -> None:
    config = PolicyGradientConfig(
        hidden_sizes=(16,),
    )

    network = PolicyNetwork(
        input_size=FEATURE_STATE_SIZE,
        hidden_sizes=config.hidden_sizes,
    )

    path = tmp_path / "features.pt"

    save_policy_gradient_checkpoint(
        path,
        policy_network=network,
        state_representation=(
            StateRepresentation.FEATURES
        ),
        reward_type=(
            RewardType.NET_PROFIT
        ),
        config=config,
    )

    loaded = (
        load_policy_gradient_checkpoint(
            path
        )
    )

    assert (
        loaded.policy_network.input_size
        == FEATURE_STATE_SIZE
    )

def test_rejects_network_incompatible_with_state_representation(
    tmp_path: Path,
) -> None:
    config = PolicyGradientConfig()

    network = PolicyNetwork(
        input_size=FEATURE_STATE_SIZE,
        hidden_sizes=config.hidden_sizes,
    )

    with pytest.raises(
        ValueError,
        match=(
            "policy network input size does not match "
            "the state representation"
        ),
    ):
        save_policy_gradient_checkpoint(
            tmp_path / "invalid.pt",
            policy_network=network,
            state_representation=(
                StateRepresentation.RAW
            ),
            reward_type=(
                RewardType.NET_PROFIT
            ),
            config=config,
        )


def test_rejects_network_architecture_incompatible_with_config(
    tmp_path: Path,
) -> None:
    config = PolicyGradientConfig(
        hidden_sizes=(
            256,
            256,
        )
    )

    network = PolicyNetwork(
        input_size=RAW_STATE_SIZE,
        hidden_sizes=(
            32,
            16,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "policy network architecture does not match "
            "the training configuration"
        ),
    ):
        save_policy_gradient_checkpoint(
            tmp_path / "invalid.pt",
            policy_network=network,
            state_representation=(
                StateRepresentation.RAW
            ),
            reward_type=(
                RewardType.NET_PROFIT
            ),
            config=config,
        )


