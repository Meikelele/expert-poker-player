from dataclasses import dataclass
import random

import torch

from expert_poker_player.policy_gradient.agent import (
    PolicyGradientAgent,
)
from expert_poker_player.policy_gradient.config import (
    PolicyGradientConfig,
)
from expert_poker_player.policy_gradient.network import (
    PolicyNetwork,
)
from expert_poker_player.policy_gradient.optimization import (
    ReinforceOptimizer,
)
from expert_poker_player.policy_gradient.trajectory import (
    PolicyStep,
    Trajectory,
)
from expert_poker_player.rewards import (
    RewardFunction,
)
from expert_poker_player.rl.actions import (
    ACTION_ORDER,
    action_to_index,
)
from expert_poker_player.state_representation import (
    StateEncoder,
)
from expert_poker_player.uth import (
    UTHGame,
    UTHObservation,
)


@dataclass(
    frozen=True,
    slots=True,
)
class PolicyGradientEpisodeStats:
    """Statystyki pojedynczego epizodu treningowego."""

    episode: int
    total_reward: float
    steps: int

@dataclass(
    frozen=True,
    slots=True,
)
class PolicyGradientUpdateStats:
    """Statystyki pojedynczego batch update'u REINFORCE."""

    update: int
    first_episode: int
    last_episode: int
    batch_size: int
    loss: float


@dataclass(
    frozen=True,
    slots=True,
)
class PolicyGradientTrainingResult:
    """Wynik kompletnego treningu Policy Gradient."""

    agent: PolicyGradientAgent
    policy_network: PolicyNetwork
    episode_stats: tuple[
        PolicyGradientEpisodeStats,
        ...
    ]
    update_stats: tuple[
        PolicyGradientUpdateStats,
        ...
    ]
    total_steps: int
    optimizer_updates: int


def train_policy_gradient(
    *,
    state_encoder: StateEncoder,
    reward_function: RewardFunction,
    config: PolicyGradientConfig,
) -> PolicyGradientTrainingResult:
    """Trenuje politykę REINFORCE w środowisku UTH."""

    if not isinstance(
        state_encoder,
        StateEncoder,
    ):  # type: ignore
        raise TypeError(
            "state_encoder must satisfy StateEncoder"
        )

    if not isinstance(
        reward_function,
        RewardFunction,
    ):  # type: ignore
        raise TypeError(
            "reward_function must satisfy RewardFunction"
        )

    if not isinstance(
        config,
        PolicyGradientConfig,
    ):  # type: ignore
        raise TypeError(
            "config must be an instance "
            "of PolicyGradientConfig"
        )

    seed_rng = random.Random(
        config.seed
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

    torch.manual_seed(  # pyright: ignore[reportUnknownMemberType]
        torch_seed
    )

    policy_network = PolicyNetwork(
        input_size=state_encoder.output_size,
        hidden_sizes=config.hidden_sizes,
    )

    policy_network.train()

    optimizer = ReinforceOptimizer(
        policy_network=policy_network,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
    )

    agent = PolicyGradientAgent(
        policy_network=policy_network,
        state_encoder=state_encoder,
        deterministic=False,
        seed=agent_seed,
    )

    game = UTHGame(
        seed=environment_seed
    )

    total_steps = 0
    optimizer_updates = 0

    episode_stats: list[
        PolicyGradientEpisodeStats
    ] = []

    update_stats: list[
        PolicyGradientUpdateStats
    ] = []

    pending_trajectories: list[
        Trajectory
    ] = []

    for episode in range(
        config.training_episodes
    ):
        observation = game.reset()

        steps: list[
            PolicyStep
        ] = []

        episode_reward = 0.0
        episode_steps = 0

        while not observation.terminated:
            state = state_encoder.encode(
                observation
            )

            action_mask = _build_action_mask(
                observation
            )

            action = agent.select_action(
                observation
            )

            step_result = game.step(
                action
            )

            reward = reward_function.calculate_reward(
                step_result
            )

            steps.append(
                PolicyStep(
                    state=state,
                    action_index=action_to_index(
                        action
                    ),
                    action_mask=action_mask,
                    reward=reward,
                )
            )

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            observation = (
                step_result.observation
            )

        trajectory = Trajectory(
            steps=tuple(
                steps
            )
        )

        pending_trajectories.append(
            trajectory
        )

        episode_stats.append(
            PolicyGradientEpisodeStats(
                episode=episode,
                total_reward=episode_reward,
                steps=episode_steps,
            )
        )

        if (
            len(pending_trajectories)
            == config.batch_size
        ):
            loss = optimizer.optimize_batch(
                tuple(
                    pending_trajectories
                )
            )

            update_stats.append(
                PolicyGradientUpdateStats(
                    update=optimizer_updates,
                    first_episode=(
                        episode
                        - len(
                            pending_trajectories
                        )
                        + 1
                    ),
                    last_episode=episode,
                    batch_size=len(
                        pending_trajectories
                    ),
                    loss=loss,
                )
            )

            optimizer_updates += 1

            pending_trajectories.clear()

    if pending_trajectories:
        loss = optimizer.optimize_batch(
            tuple(
                pending_trajectories
            )
        )

        batch_size = len(
            pending_trajectories
        )

        update_stats.append(
            PolicyGradientUpdateStats(
                update=optimizer_updates,
                first_episode=(
                    config.training_episodes
                    - batch_size
                ),
                last_episode=(
                    config.training_episodes
                    - 1
                ),
                batch_size=batch_size,
                loss=loss,
            )
        )

        optimizer_updates += 1

    return PolicyGradientTrainingResult(
        agent=agent,
        policy_network=policy_network,
        episode_stats=tuple(
            episode_stats
        ),
        update_stats=tuple(
            update_stats
        ),
        total_steps=total_steps,
        optimizer_updates=optimizer_updates,
    )


def _build_action_mask(
    observation: UTHObservation,
) -> tuple[bool, ...]:
    return tuple(
        action in observation.legal_actions
        for action in ACTION_ORDER
    )