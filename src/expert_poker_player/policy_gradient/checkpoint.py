from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Final,
    cast,
)

import torch

from expert_poker_player.policy_gradient.config import (
    PolicyGradientConfig,
)
from expert_poker_player.policy_gradient.network import (
    PolicyNetwork,
)
from expert_poker_player.rewards import (
    RewardType,
)
from expert_poker_player.state_representation import (
    StateRepresentation,
    build_state_encoder,
)


CHECKPOINT_VERSION: Final[int] = 1


@dataclass(
    frozen=True,
    slots=True,
)
class LoadedPolicyGradientCheckpoint:
    """Wczytany checkpoint modelu Policy Gradient."""

    policy_network: PolicyNetwork
    state_representation: StateRepresentation
    reward_type: RewardType
    config: PolicyGradientConfig
    training_seed: int


def save_policy_gradient_checkpoint(
    path: str | Path,
    *,
    policy_network: PolicyNetwork,
    state_representation: StateRepresentation,
    reward_type: RewardType,
    config: PolicyGradientConfig,
) -> None:
    """Zapisuje model Policy Gradient wraz z metadanymi."""

    if not isinstance(
        policy_network,
        PolicyNetwork,
    ):  # type: ignore
        raise TypeError(
            "policy_network must be an instance "
            "of PolicyNetwork"
        )

    if not isinstance(
        state_representation,
        StateRepresentation,
    ):  # type: ignore
        raise TypeError(
            "state_representation must be an instance "
            "of StateRepresentation"
        )

    if not isinstance(
        reward_type,
        RewardType,
    ):  # type: ignore
        raise TypeError(
            "reward_type must be an instance "
            "of RewardType"
        )

    if not isinstance(
        config,
        PolicyGradientConfig,
    ):  # type: ignore
        raise TypeError(
            "config must be an instance "
            "of PolicyGradientConfig"
        )

    encoder = build_state_encoder(
        state_representation
    )

    if (
        policy_network.input_size
        != encoder.output_size
    ):
        raise ValueError(
            "policy network input size does not match "
            "the state representation"
        )

    if (
        policy_network.hidden_sizes
        != config.hidden_sizes
    ):
        raise ValueError(
            "policy network architecture does not match "
            "the training configuration"
        )

    checkpoint: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "policy_state_dict": (
            policy_network.state_dict()
        ),
        "input_size": (
            policy_network.input_size
        ),
        "hidden_sizes": list(
            policy_network.hidden_sizes
        ),
        "state_representation": (
            state_representation.value
        ),
        "reward_type": reward_type.value,
        "training_seed": config.seed,
        "config": config.to_dict(),
    }

    torch.save(
        checkpoint,
        Path(path),
    )


def load_policy_gradient_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedPolicyGradientCheckpoint:
    """Wczytuje model Policy Gradient wraz z metadanymi."""

    loaded = torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )

    if not isinstance(
        loaded,
        dict,
    ):
        raise ValueError(
            "invalid Policy Gradient checkpoint"
        )

    checkpoint = cast(
        "dict[str, Any]",
        loaded,
    )

    if (
        checkpoint.get("version")
        != CHECKPOINT_VERSION
    ):
        raise ValueError(
            "unsupported Policy Gradient "
            "checkpoint version"
        )

    state_representation = (
        StateRepresentation(
            checkpoint[
                "state_representation"
            ]
        )
    )

    reward_type = RewardType(
        checkpoint[
            "reward_type"
        ]
    )

    if not isinstance(
        checkpoint["config"],
        dict,
    ):
        raise ValueError(
            "checkpoint config must be a mapping"
        )

    config_values = dict(
        cast(
            "dict[str, Any]",
            checkpoint["config"],
        )
    )

    if not isinstance(
        config_values.get(
            "hidden_sizes"
        ),
        list,
    ):
        raise ValueError(
            "checkpoint hidden_sizes "
            "must be a list"
        )

    config_values[
        "hidden_sizes"
    ] = tuple(
        config_values[
            "hidden_sizes"
        ]
    )

    config = PolicyGradientConfig(
        **config_values
    )

    training_seed = checkpoint[
        "training_seed"
    ]

    if not isinstance(
        training_seed,
        int,
    ):
        raise ValueError(
            "checkpoint training seed "
            "must be an integer"
        )

    if training_seed != config.seed:
        raise ValueError(
            "checkpoint training seed does not match "
            "the stored configuration"
        )

    input_size = checkpoint[
        "input_size"
    ]

    if not isinstance(
        input_size,
        int,
    ):
        raise ValueError(
            "checkpoint input size "
            "must be an integer"
        )

    stored_hidden_sizes_raw = (
        checkpoint[
            "hidden_sizes"
        ]
    )

    if not isinstance(
        stored_hidden_sizes_raw,
        list,
    ):
        raise ValueError(
            "checkpoint architecture "
            "must be a list"
        )

    stored_hidden_sizes = tuple(
        cast(
            "list[int]",
            stored_hidden_sizes_raw,
        )
    )

    if (
        stored_hidden_sizes
        != config.hidden_sizes
    ):
        raise ValueError(
            "checkpoint architecture does not match "
            "the stored configuration"
        )

    encoder = build_state_encoder(
        state_representation
    )

    if (
        input_size
        != encoder.output_size
    ):
        raise ValueError(
            "checkpoint input size does not match "
            "the state representation"
        )

    network = PolicyNetwork(
        input_size=input_size,
        hidden_sizes=stored_hidden_sizes,
    )

    network.load_state_dict(
        checkpoint[
            "policy_state_dict"
        ]
    )

    network.to(
        device
    )

    return LoadedPolicyGradientCheckpoint(
        policy_network=network,
        state_representation=state_representation,
        reward_type=reward_type,
        config=config,
        training_seed=training_seed,
    )