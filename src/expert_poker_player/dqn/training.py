from dataclasses import dataclass
import random
from collections.abc import Callable
import torch

from expert_poker_player.dqn.actions import (
    ACTION_ORDER,
    action_to_index,
)
from expert_poker_player.dqn.agent import (
    DQNAgent,
)
from expert_poker_player.dqn.config import (
    DQNConfig,
)
from expert_poker_player.dqn.network import (
    QNetwork,
)
from expert_poker_player.dqn.optimization import (
    DQNOptimizer,
)
from expert_poker_player.dqn.replay import (
    ActionMask,
    ReplayBuffer,
    Transition,
)
from expert_poker_player.rewards import (
    RewardFunction,
)
from expert_poker_player.state_representation import (
    StateEncoder,
)
from expert_poker_player.uth import (
    UTHGame,
    UTHObservation,
)

DQNTrainingObserver = Callable[
    [int, QNetwork],
    None,
]


@dataclass(
    frozen=True,
    slots=True,
)
class DQNEpisodeStats:
    """Statystyki pojedynczego epizodu treningowego."""

    episode: int
    total_reward: float
    steps: int
    optimizer_updates: int
    mean_loss: float | None
    mean_gradient_norm: float | None
    epsilon: float


@dataclass(
    frozen=True,
    slots=True,
)
class DQNTrainingResult:
    """Wynik kompletnego treningu DQN."""

    agent: DQNAgent
    policy_network: QNetwork
    target_network: QNetwork
    episode_stats: tuple[DQNEpisodeStats, ...]
    total_steps: int
    optimizer_updates: int


def train_dqn(
    *,
    state_encoder: StateEncoder,
    reward_function: RewardFunction,
    config: DQNConfig,
    on_episode_completed: DQNTrainingObserver | None = None,
) -> DQNTrainingResult:
    """Trenuje agenta DQN w środowisku Ultimate Texas Hold'em."""

    if not isinstance(
        state_encoder,
        StateEncoder,
    ): # type: ignore
        raise TypeError(
            "state_encoder must satisfy StateEncoder"
        )

    if not isinstance(
        reward_function,
        RewardFunction,
    ): # type: ignore
        raise TypeError(
            "reward_function must satisfy RewardFunction"
        )

    if not isinstance(
        config,
        DQNConfig,
    ): # type: ignore
        raise TypeError(
            "config must be an instance of DQNConfig"
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

    replay_seed = seed_rng.randrange(
        0,
        2**63,
    )

    torch.manual_seed(  # pyright: ignore[reportUnknownMemberType]
        torch_seed
    )

    policy_network = QNetwork(
        input_size=state_encoder.output_size,
        hidden_sizes=config.hidden_sizes,
    )

    target_network = QNetwork(
        input_size=state_encoder.output_size,
        hidden_sizes=config.hidden_sizes,
    )

    optimizer = DQNOptimizer(
        policy_network=policy_network,
        target_network=target_network,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
    )

    optimizer.sync_target_network()

    agent = DQNAgent(
        q_network=policy_network,
        state_encoder=state_encoder,
        epsilon=config.epsilon_start,
        seed=agent_seed,
    )

    replay_buffer = ReplayBuffer(
        capacity=config.replay_capacity,
        seed=replay_seed,
    )

    game = UTHGame(
        seed=environment_seed
    )

    total_steps = 0
    optimizer_updates = 0

    episode_stats: list[
        DQNEpisodeStats
    ] = []

    for episode in range(
        config.training_episodes
    ):
        observation = game.reset()

        episode_reward = 0.0
        episode_steps = 0
        episode_updates = 0

        losses: list[float] = []
        gradient_norms: list[float] = []

        while not observation.terminated:
            epsilon = config.epsilon_at_step(
                total_steps
            )

            agent.epsilon = epsilon

            state = state_encoder.encode(
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

            if step_result.terminated:
                next_state = None
                next_action_mask = None
            else:
                next_observation = (
                    step_result.observation
                )

                next_state = state_encoder.encode(
                    next_observation
                )

                next_action_mask = (
                    _build_action_mask(
                        next_observation
                    )
                )

            replay_buffer.add(
                Transition(
                    state=state,
                    action_index=action_to_index(
                        action
                    ),
                    reward=reward,
                    next_state=next_state,
                    terminated=step_result.terminated,
                    next_action_mask=next_action_mask,
                )
            )

            total_steps += 1
            episode_steps += 1
            episode_reward += reward

            if (
                total_steps
                >= config.warmup_steps
                and len(replay_buffer)
                >= config.batch_size
            ):
                transitions = replay_buffer.sample(
                    config.batch_size
                )

                result = optimizer.optimize(
                    transitions
                )

                losses.append(
                    result.loss
                )

                gradient_norms.append(
                    result.gradient_norm
                )

                optimizer_updates += 1
                episode_updates += 1

                if (
                    optimizer_updates
                    % config.target_sync_interval
                    == 0
                ):
                    optimizer.sync_target_network()

            observation = step_result.observation

        mean_loss = (
            sum(losses) / len(losses)
            if losses
            else None
        )

        mean_gradient_norm = (
            sum(gradient_norms) / len(gradient_norms)
            if gradient_norms
            else None
        )

        episode_stats.append(
            DQNEpisodeStats(
                episode=episode,
                total_reward=episode_reward,
                steps=episode_steps,
                optimizer_updates=episode_updates,
                mean_loss=mean_loss,
                mean_gradient_norm=mean_gradient_norm,
                epsilon=agent.epsilon,
            )
        )

        if on_episode_completed is not None:
            on_episode_completed(
                episode + 1,
                policy_network,
            )

    return DQNTrainingResult(
        agent=agent,
        policy_network=policy_network,
        target_network=target_network,
        episode_stats=tuple(
            episode_stats
        ),
        total_steps=total_steps,
        optimizer_updates=optimizer_updates,
    )


def _build_action_mask(
    observation: UTHObservation,
) -> ActionMask:
    return tuple(
        action in observation.legal_actions
        for action in ACTION_ORDER
    )