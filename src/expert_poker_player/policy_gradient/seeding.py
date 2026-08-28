from dataclasses import dataclass
import random

import torch

from expert_poker_player.policy_gradient.config import (
    PolicyGradientConfig,
)
from expert_poker_player.policy_gradient.network import (
    PolicyNetwork,
)
from expert_poker_player.state_representation import (
    StateEncoder,
)


@dataclass(
    frozen=True,
    slots=True,
)
class PolicyGradientSeeds:
    """Ziarna wyprowadzone z jednego ziarna konfiguracji."""

    torch_seed: int
    environment_seed: int
    agent_seed: int

    def __post_init__(self) -> None:
        for name, value in (
            ("torch_seed", self.torch_seed),
            ("environment_seed", self.environment_seed),
            ("agent_seed", self.agent_seed),
        ):
            if type(value) is not int:
                raise TypeError(
                    f"{name} must be an integer"
                )


def derive_policy_gradient_seeds(
    seed: int,
) -> PolicyGradientSeeds:
    """
    Wyprowadza ziarna torcha, środowiska i agenta z jednego ziarna.

    Kolejność losowania musi pozostać niezmienna, ponieważ od niej
    zależy odtwarzalność treningu i kontroli nieuczonej sieci.
    """

    if type(seed) is not int:
        raise TypeError(
            "seed must be an integer"
        )

    seed_rng = random.Random(
        seed
    )

    torch_seed = seed_rng.randrange(
        0,
        2**63,
    )

    environment_seed = seed_rng.randrange(
        0,
        2**63,
    )

    agent_seed = seed_rng.randrange(
        0,
        2**63,
    )

    return PolicyGradientSeeds(
        torch_seed=torch_seed,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
    )


def build_initial_policy_network(
    *,
    state_encoder: StateEncoder,
    config: PolicyGradientConfig,
) -> tuple[
    PolicyNetwork,
    PolicyGradientSeeds,
]:
    """
    Tworzy nowo zainicjalizowaną sieć polityki dla danej konfiguracji.

    Używana zarówno przez pełny trening, jak i kontrolę nieuczonej
    sieci, aby obie ścieżki korzystały z tej samej procedury seedowania.
    """

    if not isinstance(
        state_encoder,
        StateEncoder,
    ):  # type: ignore
        raise TypeError(
            "state_encoder must satisfy StateEncoder"
        )

    if not isinstance(
        config,
        PolicyGradientConfig,
    ):  # type: ignore
        raise TypeError(
            "config must be an instance "
            "of PolicyGradientConfig"
        )

    seeds = derive_policy_gradient_seeds(
        config.seed
    )

    torch.manual_seed(  # pyright: ignore[reportUnknownMemberType]
        seeds.torch_seed
    )

    policy_network = PolicyNetwork(
        input_size=state_encoder.output_size,
        hidden_sizes=config.hidden_sizes,
    )

    return policy_network, seeds
