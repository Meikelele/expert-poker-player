from dataclasses import dataclass

from expert_poker_player.dqn.agent import DQNAgent
from expert_poker_player.dqn.network import QNetwork
from expert_poker_player.evaluation import (
    AgentMetrics,
    SimulationConfig,
    calculate_metrics,
    run_simulation,
)
from expert_poker_player.policy_gradient.agent import (
    PolicyGradientAgent,
)
from expert_poker_player.policy_gradient.network import (
    PolicyNetwork,
)
from expert_poker_player.state_representation import (
    StateEncoder,
)
from expert_poker_player.agents import Agent


@dataclass(frozen=True, slots=True)
class PolicyEvaluationSnapshot:
    completed_episodes: int
    metrics: AgentMetrics


class DQNPeriodicEvaluator:
    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        schedule: SimulationConfig,
        checkpoints: tuple[int, ...],
    ) -> None:
        self._state_encoder = state_encoder
        self._schedule = schedule
        self._checkpoints = _validate_checkpoints(
            checkpoints
        )
        self._snapshots: list[
            PolicyEvaluationSnapshot
        ] = []

    @property
    def snapshots(
        self,
    ) -> tuple[PolicyEvaluationSnapshot, ...]:
        return tuple(
            self._snapshots
        )

    def __call__(
        self,
        completed_episodes: int,
        policy_network: QNetwork,
    ) -> None:
        if (
            completed_episodes
            not in self._checkpoints
        ):
            return

        agent = DQNAgent(
            q_network=policy_network,
            state_encoder=self._state_encoder,
            epsilon=0.0,
            seed=0,
        )

        self._snapshots.append(
            _evaluate(
                completed_episodes=completed_episodes,
                agent=agent,
                schedule=self._schedule,
            )
        )


class PolicyGradientPeriodicEvaluator:
    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        schedule: SimulationConfig,
        checkpoints: tuple[int, ...],
    ) -> None:
        self._state_encoder = state_encoder
        self._schedule = schedule
        self._checkpoints = _validate_checkpoints(
            checkpoints
        )
        self._snapshots: list[
            PolicyEvaluationSnapshot
        ] = []

    @property
    def snapshots(
        self,
    ) -> tuple[PolicyEvaluationSnapshot, ...]:
        return tuple(
            self._snapshots
        )

    def __call__(
        self,
        completed_episodes: int,
        policy_network: PolicyNetwork,
    ) -> None:
        if (
            completed_episodes
            not in self._checkpoints
        ):
            return

        agent = PolicyGradientAgent(
            policy_network=policy_network,
            state_encoder=self._state_encoder,
            deterministic=True,
            seed=0,
        )

        self._snapshots.append(
            _evaluate(
                completed_episodes=completed_episodes,
                agent=agent,
                schedule=self._schedule,
            )
        )


def _evaluate(
    *,
    completed_episodes: int,
    agent: Agent,
    schedule: SimulationConfig,
) -> PolicyEvaluationSnapshot:
    simulation = run_simulation(
        agent=agent,  # type: ignore[arg-type]
        config=schedule,
    )

    return PolicyEvaluationSnapshot(
        completed_episodes=completed_episodes,
        metrics=calculate_metrics(
            simulation
        ),
    )


def _validate_checkpoints(
    checkpoints: tuple[int, ...],
) -> frozenset[int]:
    if not isinstance(
        checkpoints,
        tuple,
    ):  # type: ignore
        raise TypeError(
            "checkpoints must be a tuple"
        )

    if not checkpoints:
        raise ValueError(
            "checkpoints cannot be empty"
        )

    if not all(
        type(checkpoint) is int
        for checkpoint in checkpoints
    ):
        raise TypeError(
            "checkpoints must contain only integers"
        )

    if any(
        checkpoint <= 0
        for checkpoint in checkpoints
    ):
        raise ValueError(
            "checkpoints must be positive"
        )

    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError(
            "checkpoints must be unique"
        )

    return frozenset(
        checkpoints
    )