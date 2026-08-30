from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from expert_poker_player.evaluation import AgentMetrics
from expert_poker_player.experiments.periodic_evaluation import (
    PolicyEvaluationSnapshot,
)
from expert_poker_player.experiments.protocol import (
    ExperimentVariant,
)
from expert_poker_player.uth import (
    Action,
    RoundOutcome,
)


RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    round_count: int
    total_net_profit: float
    estimated_ev: float
    total_staked: float
    mean_staked: float
    standard_deviation: float
    standard_error: float
    outcome_counts: Mapping[str, int]
    action_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if type(self.round_count) is not int:
            raise TypeError(
                "round_count must be an integer"
            )

        if self.round_count <= 0:
            raise ValueError(
                "round_count must be positive"
            )

        numeric_values = (
            self.total_net_profit,
            self.estimated_ev,
            self.total_staked,
            self.mean_staked,
            self.standard_deviation,
            self.standard_error,
        )

        if not all(
            isinstance(value, (int, float)) # type: ignore
            for value in numeric_values
        ):
            raise TypeError(
                "evaluation metrics must be numeric"
            )

        if not all(
            isfinite(float(value))
            for value in numeric_values
        ):
            raise ValueError(
                "evaluation metrics must be finite"
            )

    @classmethod
    def from_metrics(
        cls,
        metrics: AgentMetrics,
    ) -> "EvaluationRecord":
        if not isinstance(
            metrics,
            AgentMetrics,
        ):  # type: ignore
            raise TypeError(
                "metrics must be an instance of AgentMetrics"
            )

        return cls(
            round_count=metrics.round_count,
            total_net_profit=float(
                metrics.total_net_profit
            ),
            estimated_ev=float(
                metrics.estimated_ev
            ),
            total_staked=float(
                metrics.total_staked
            ),
            mean_staked=float(
                metrics.mean_staked
            ),
            standard_deviation=(
                metrics.standard_deviation
            ),
            standard_error=(
                metrics.standard_error
            ),
            outcome_counts={
                outcome.name: metrics.outcome_counts[
                    outcome
                ]
                for outcome in RoundOutcome
            },
            action_counts={
                action.name: metrics.action_counts[
                    action
                ]
                for action in Action
            },
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "round_count": self.round_count,
            "total_net_profit": self.total_net_profit,
            "estimated_ev": self.estimated_ev,
            "total_staked": self.total_staked,
            "mean_staked": self.mean_staked,
            "standard_deviation": (
                self.standard_deviation
            ),
            "standard_error": self.standard_error,
            "outcome_counts": dict(
                self.outcome_counts
            ),
            "action_counts": dict(
                self.action_counts
            ),
        }


@dataclass(frozen=True, slots=True)
class LearningCurvePoint:
    completed_episodes: int
    evaluation: EvaluationRecord

    def __post_init__(self) -> None:
        if type(self.completed_episodes) is not int:
            raise TypeError(
                "completed_episodes must be an integer"
            )

        if self.completed_episodes <= 0:
            raise ValueError(
                "completed_episodes must be positive"
            )

        if not isinstance(
            self.evaluation,
            EvaluationRecord,
        ):  # type: ignore
            raise TypeError(
                "evaluation must be an EvaluationRecord"
            )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PolicyEvaluationSnapshot,
    ) -> "LearningCurvePoint":
        if not isinstance(
            snapshot,
            PolicyEvaluationSnapshot,
        ):  # type: ignore
            raise TypeError(
                "snapshot must be a PolicyEvaluationSnapshot"
            )

        return cls(
            completed_episodes=(
                snapshot.completed_episodes
            ),
            evaluation=EvaluationRecord.from_metrics(
                snapshot.metrics
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "completed_episodes": (
                self.completed_episodes
            ),
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExperimentRunSummary:
    variant: ExperimentVariant
    training_seed: int
    training_episodes: int
    total_steps: int
    optimizer_updates: int
    training_config: Mapping[str, object]
    validation_curve: tuple[
        LearningCurvePoint,
        ...
    ]
    final_evaluation: EvaluationRecord | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.variant,
            ExperimentVariant,
        ):  # type: ignore
            raise TypeError(
                "variant must be an ExperimentVariant"
            )

        if type(self.training_seed) is not int:
            raise TypeError(
                "training_seed must be an integer"
            )

        if (
            type(self.training_episodes) is not int
            or self.training_episodes <= 0
        ):
            raise ValueError(
                "training_episodes must be positive"
            )

        if (
            type(self.total_steps) is not int
            or self.total_steps <= 0
        ):
            raise ValueError(
                "total_steps must be positive"
            )

        if (
            type(self.optimizer_updates) is not int
            or self.optimizer_updates < 0
        ):
            raise ValueError(
                "optimizer_updates cannot be negative"
            )

        if not isinstance(
            self.validation_curve,
            tuple,
        ):  # type: ignore
            raise TypeError(
                "validation_curve must be a tuple"
            )

        completed_episodes = tuple(
            point.completed_episodes
            for point in self.validation_curve
        )

        if completed_episodes != tuple(
            sorted(
                set(completed_episodes)
            )
        ):
            raise ValueError(
                "validation checkpoints must be "
                "strictly increasing"
            )

        if (
            completed_episodes
            and completed_episodes[-1]
            > self.training_episodes
        ):
            raise ValueError(
                "validation checkpoint cannot exceed "
                "training_episodes"
            )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "variant": {
                "name": self.variant.name,
                "algorithm": self.variant.algorithm.value,
                "state_representation": (
                    self.variant.state_representation.value
                ),
                "reward_type": (
                    self.variant.reward_type.value
                ),
            },
            "training_seed": self.training_seed,
            "training_episodes": self.training_episodes,
            "total_steps": self.total_steps,
            "optimizer_updates": self.optimizer_updates,
            "training_config": dict(
                self.training_config
            ),
            "validation_curve": [
                point.to_dict()
                for point in self.validation_curve
            ],
            "final_evaluation": (
                self.final_evaluation.to_dict()
                if self.final_evaluation is not None
                else None
            ),
        }


def build_learning_curve(
    snapshots: tuple[
        PolicyEvaluationSnapshot,
        ...
    ],
) -> tuple[LearningCurvePoint, ...]:
    return tuple(
        LearningCurvePoint.from_snapshot(
            snapshot
        )
        for snapshot in snapshots
    )