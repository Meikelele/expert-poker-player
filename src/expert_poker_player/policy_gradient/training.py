from dataclasses import dataclass

from expert_poker_player.policy_gradient.agent import (
    PolicyGradientAgent,
)
from expert_poker_player.policy_gradient.config import (
    PolicyGradientConfig,
)
from expert_poker_player.policy_gradient.diagnostics import (
    PROBE_CHECKPOINTS,
    ProbeSnapshot,
    ProbeState,
    compute_probe_snapshots,
)
from expert_poker_player.policy_gradient.network import (
    PolicyNetwork,
)
from expert_poker_player.policy_gradient.optimization import (
    ReinforceOptimizer,
)
from expert_poker_player.policy_gradient.seeding import (
    build_initial_policy_network,
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
    gradient_norm: float
    cumulative_steps: int
    mean_episode_length: float
    mean_abs_return: float
    max_abs_return: float


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
    probe_snapshots: tuple[
        ProbeSnapshot,
        ...
    ]
    total_steps: int
    optimizer_updates: int


def train_policy_gradient(
    *,
    state_encoder: StateEncoder,
    reward_function: RewardFunction,
    config: PolicyGradientConfig,
    probe_states: tuple[ProbeState, ...] | None = None,
    probe_checkpoints: tuple[int, ...] = PROBE_CHECKPOINTS,
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

    policy_network, seeds = build_initial_policy_network(
        state_encoder=state_encoder,
        config=config,
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
        seed=seeds.agent_seed,
    )

    game = UTHGame(
        seed=seeds.environment_seed
    )

    total_steps = 0
    optimizer_updates = 0

    episode_stats: list[
        PolicyGradientEpisodeStats
    ] = []

    update_stats: list[
        PolicyGradientUpdateStats
    ] = []

    probe_snapshots: list[
        ProbeSnapshot
    ] = []

    requested_checkpoints = (
        frozenset(probe_checkpoints)
        if probe_states
        else frozenset()
    )

    _maybe_capture_probe_snapshot(
        agent=agent,
        probe_states=probe_states,
        checkpoints=requested_checkpoints,
        update=0,
        snapshots=probe_snapshots,
    )

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
            stats, optimizer_updates = _finalize_batch(
                optimizer=optimizer,
                pending_trajectories=pending_trajectories,
                optimizer_updates=optimizer_updates,
                first_episode=(
                    episode
                    - len(
                        pending_trajectories
                    )
                    + 1
                ),
                last_episode=episode,
                cumulative_steps=total_steps,
            )

            update_stats.append(
                stats
            )

            _maybe_capture_probe_snapshot(
                agent=agent,
                probe_states=probe_states,
                checkpoints=requested_checkpoints,
                update=optimizer_updates,
                snapshots=probe_snapshots,
            )

            pending_trajectories.clear()

    if pending_trajectories:
        batch_size = len(
            pending_trajectories
        )

        stats, optimizer_updates = _finalize_batch(
            optimizer=optimizer,
            pending_trajectories=pending_trajectories,
            optimizer_updates=optimizer_updates,
            first_episode=(
                config.training_episodes
                - batch_size
            ),
            last_episode=(
                config.training_episodes
                - 1
            ),
            cumulative_steps=total_steps,
        )

        update_stats.append(
            stats
        )

        _maybe_capture_probe_snapshot(
            agent=agent,
            probe_states=probe_states,
            checkpoints=requested_checkpoints,
            update=optimizer_updates,
            snapshots=probe_snapshots,
        )

    return PolicyGradientTrainingResult(
        agent=agent,
        policy_network=policy_network,
        episode_stats=tuple(
            episode_stats
        ),
        update_stats=tuple(
            update_stats
        ),
        probe_snapshots=tuple(
            probe_snapshots
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


def _finalize_batch(
    *,
    optimizer: ReinforceOptimizer,
    pending_trajectories: list[Trajectory],
    optimizer_updates: int,
    first_episode: int,
    last_episode: int,
    cumulative_steps: int,
) -> tuple[PolicyGradientUpdateStats, int]:
    """
    Wykonuje jeden update REINFORCE i buduje jego statystyki.

    Współdzielona przez pełny batch i końcowy niepełny batch, żeby
    nie duplikować logiki liczenia statystyk update'u.
    """

    trajectories = tuple(
        pending_trajectories
    )

    result = optimizer.optimize_batch(
        trajectories
    )

    episode_lengths = [
        len(trajectory)
        for trajectory in trajectories
    ]

    absolute_returns = [
        abs(trajectory.total_reward)
        for trajectory in trajectories
    ]

    completed_updates = (
        optimizer_updates + 1
    )

    stats = PolicyGradientUpdateStats(
        update=completed_updates,
        first_episode=first_episode,
        last_episode=last_episode,
        batch_size=len(
            trajectories
        ),
        loss=result.loss,
        gradient_norm=result.gradient_norm,
        cumulative_steps=cumulative_steps,
        mean_episode_length=(
            sum(episode_lengths)
            / len(episode_lengths)
        ),
        mean_abs_return=(
            sum(absolute_returns)
            / len(absolute_returns)
        ),
        max_abs_return=max(
            absolute_returns
        ),
    )

    return stats, completed_updates


def _maybe_capture_probe_snapshot(
    *,
    agent: PolicyGradientAgent,
    probe_states: tuple[ProbeState, ...] | None,
    checkpoints: frozenset[int],
    update: int,
    snapshots: list[ProbeSnapshot],
) -> None:
    """Dopisuje zrzut sond, jeśli ten update znajduje się w harmonogramie."""

    if not probe_states or update not in checkpoints:
        return

    snapshots.extend(
        compute_probe_snapshots(
            agent=agent,
            probe_states=probe_states,
            update=update,
        )
    )