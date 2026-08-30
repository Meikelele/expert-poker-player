from dataclasses import dataclass

from expert_poker_player.dqn import (
    DQNConfig,
    DQNTrainingResult,
    QNetwork,
    train_dqn,
)
from expert_poker_player.evaluation import (
    SimulationConfig,
)
from expert_poker_player.experiments.periodic_evaluation import (
    DQNPeriodicEvaluator,
    PolicyGradientPeriodicEvaluator,
)
from expert_poker_player.experiments.protocol import (
    ExperimentVariant,
    RLAlgorithm,
)
from expert_poker_player.experiments.results import (
    EvaluationRecord,
    ExperimentRunSummary,
    build_learning_curve,
)
from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
    PolicyGradientTrainingResult,
    PolicyNetwork,
    train_policy_gradient,
)
from expert_poker_player.rewards import (
    build_reward_function,
)
from expert_poker_player.state_representation import (
    StateEncoder,
    build_state_encoder,
)


TrainingConfig = (
    DQNConfig
    | PolicyGradientConfig
)

TrainedNetwork = (
    QNetwork
    | PolicyNetwork
)


@dataclass(frozen=True, slots=True)
class ExperimentRunSpec:
    variant: ExperimentVariant
    training_seed: int
    training_episodes: int
    validation_schedule: SimulationConfig
    validation_checkpoints: tuple[int, ...]
    final_evaluation_schedule: (
        SimulationConfig | None
    ) = None

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

        if type(self.training_episodes) is not int:
            raise TypeError(
                "training_episodes must be an integer"
            )

        if self.training_episodes <= 0:
            raise ValueError(
                "training_episodes must be positive"
            )

        if not isinstance(
            self.validation_schedule,
            SimulationConfig,
        ):  # type: ignore
            raise TypeError(
                "validation_schedule must be "
                "a SimulationConfig"
            )

        if not isinstance(
            self.validation_checkpoints,
            tuple,
        ):  # type: ignore
            raise TypeError(
                "validation_checkpoints must be a tuple"
            )

        if not self.validation_checkpoints:
            raise ValueError(
                "validation_checkpoints cannot be empty"
            )

        if not all(
            type(checkpoint) is int
            for checkpoint
            in self.validation_checkpoints
        ):
            raise TypeError(
                "validation_checkpoints must contain "
                "only integers"
            )

        if any(
            checkpoint <= 0
            for checkpoint
            in self.validation_checkpoints
        ):
            raise ValueError(
                "validation_checkpoints must be positive"
            )

        if (
            tuple(
                sorted(
                    set(
                        self.validation_checkpoints
                    )
                )
            )
            != self.validation_checkpoints
        ):
            raise ValueError(
                "validation_checkpoints must be "
                "strictly increasing"
            )

        if (
            self.validation_checkpoints[-1]
            > self.training_episodes
        ):
            raise ValueError(
                "validation checkpoint cannot exceed "
                "training_episodes"
            )

        if (
            self.final_evaluation_schedule is not None
            and not isinstance(
                self.final_evaluation_schedule,
                SimulationConfig,
            ) # type: ignore
        ):
            raise TypeError(
                "final_evaluation_schedule must be "
                "a SimulationConfig or None"
            )


@dataclass(frozen=True, slots=True)
class ExperimentExecutionResult:
    summary: ExperimentRunSummary
    policy_network: TrainedNetwork
    training_config: TrainingConfig


def run_experiment(
    spec: ExperimentRunSpec,
) -> ExperimentExecutionResult:
    if not isinstance(
        spec,
        ExperimentRunSpec,
    ):  # type: ignore
        raise TypeError(
            "spec must be an ExperimentRunSpec"
        )

    if spec.variant.algorithm is RLAlgorithm.DQN:
        return _run_dqn_experiment(
            spec
        )

    return _run_policy_gradient_experiment(
        spec
    )


def _run_dqn_experiment(
    spec: ExperimentRunSpec,
) -> ExperimentExecutionResult:
    encoder = build_state_encoder(
        spec.variant.state_representation
    )

    reward_function = build_reward_function(
        spec.variant.reward_type
    )

    config = DQNConfig(
        training_episodes=spec.training_episodes,
        seed=spec.training_seed,
    )

    evaluator = DQNPeriodicEvaluator(
        state_encoder=encoder,
        schedule=spec.validation_schedule,
        checkpoints=spec.validation_checkpoints,
    )

    training_result = train_dqn(
        state_encoder=encoder,
        reward_function=reward_function,
        config=config,
        on_episode_completed=evaluator,
    )

    final_evaluation = (
        _evaluate_dqn_final_policy(
            network=training_result.policy_network,
            state_encoder=encoder,
            completed_episodes=(
                spec.training_episodes
            ),
            schedule=spec.final_evaluation_schedule,
        )
        if spec.final_evaluation_schedule is not None
        else None
    )

    summary = _build_dqn_summary(
        spec=spec,
        config=config,
        training_result=training_result,
        evaluator=evaluator,
        final_evaluation=final_evaluation,
    )

    return ExperimentExecutionResult(
        summary=summary,
        policy_network=training_result.policy_network,
        training_config=config,
    )


def _run_policy_gradient_experiment(
    spec: ExperimentRunSpec,
) -> ExperimentExecutionResult:
    encoder = build_state_encoder(
        spec.variant.state_representation
    )

    reward_function = build_reward_function(
        spec.variant.reward_type
    )

    config = PolicyGradientConfig(
        training_episodes=spec.training_episodes,
        seed=spec.training_seed,
    )

    evaluator = PolicyGradientPeriodicEvaluator(
        state_encoder=encoder,
        schedule=spec.validation_schedule,
        checkpoints=spec.validation_checkpoints,
    )

    training_result = train_policy_gradient(
        state_encoder=encoder,
        reward_function=reward_function,
        config=config,
        probe_states=None,
        on_episode_completed=evaluator,
    )

    final_evaluation = (
        _evaluate_pg_final_policy(
            network=training_result.policy_network,
            state_encoder=encoder,
            completed_episodes=(
                spec.training_episodes
            ),
            schedule=spec.final_evaluation_schedule,
        )
        if spec.final_evaluation_schedule is not None
        else None
    )

    summary = _build_pg_summary(
        spec=spec,
        config=config,
        training_result=training_result,
        evaluator=evaluator,
        final_evaluation=final_evaluation,
    )

    return ExperimentExecutionResult(
        summary=summary,
        policy_network=training_result.policy_network,
        training_config=config,
    )

def _evaluate_dqn_final_policy(
    *,
    network: QNetwork,
    state_encoder: StateEncoder,
    completed_episodes: int,
    schedule: SimulationConfig,
) -> EvaluationRecord:
    evaluator = DQNPeriodicEvaluator(
        state_encoder=state_encoder,
        schedule=schedule,
        checkpoints=(
            completed_episodes,
        ),
    )

    evaluator(
        completed_episodes,
        network,
    )

    return EvaluationRecord.from_metrics(
        evaluator.snapshots[0].metrics
    )


def _evaluate_pg_final_policy(
    *,
    network: PolicyNetwork,
    state_encoder: StateEncoder,
    completed_episodes: int,
    schedule: SimulationConfig,
) -> EvaluationRecord:
    evaluator = PolicyGradientPeriodicEvaluator(
        state_encoder=state_encoder,
        schedule=schedule,
        checkpoints=(
            completed_episodes,
        ),
    )

    evaluator(
        completed_episodes,
        network,
    )

    return EvaluationRecord.from_metrics(
        evaluator.snapshots[0].metrics
    )

def _build_dqn_summary(
    *,
    spec: ExperimentRunSpec,
    config: DQNConfig,
    training_result: DQNTrainingResult,
    evaluator: DQNPeriodicEvaluator,
    final_evaluation: EvaluationRecord | None,
) -> ExperimentRunSummary:
    return ExperimentRunSummary(
        variant=spec.variant,
        training_seed=spec.training_seed,
        training_episodes=spec.training_episodes,
        total_steps=training_result.total_steps,
        optimizer_updates=(
            training_result.optimizer_updates
        ),
        training_config=config.to_dict(),
        validation_curve=build_learning_curve(
            evaluator.snapshots
        ),
        final_evaluation=final_evaluation,
    )


def _build_pg_summary(
    *,
    spec: ExperimentRunSpec,
    config: PolicyGradientConfig,
    training_result: PolicyGradientTrainingResult,
    evaluator: PolicyGradientPeriodicEvaluator,
    final_evaluation: EvaluationRecord | None,
) -> ExperimentRunSummary:
    return ExperimentRunSummary(
        variant=spec.variant,
        training_seed=spec.training_seed,
        training_episodes=spec.training_episodes,
        total_steps=training_result.total_steps,
        optimizer_updates=(
            training_result.optimizer_updates
        ),
        training_config=config.to_dict(),
        validation_curve=build_learning_curve(
            evaluator.snapshots
        ),
        final_evaluation=final_evaluation,
    )