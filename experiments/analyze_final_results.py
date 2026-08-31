import csv
import json

from dataclasses import dataclass
from pathlib import Path
from typing import cast
from fractions import Fraction
from math import isfinite, sqrt
from statistics import fmean, stdev


FINAL_TRAINING_RUN_COUNT = 40
EXTENDED_TRAINING_RUN_COUNT = 10
FINAL_EVALUATION_TARGET_COUNT = 52
FINAL_EVALUATION_ROUNDS = 100_000

DEFAULT_FINAL_TRAINING_DIR = Path(
    "experiments/runs/final_training"
)

DEFAULT_EXTENDED_TRAINING_DIR = Path(
    "experiments/runs/extended_training"
)

DEFAULT_FINAL_EVALUATION_DIR = Path(
    "experiments/runs/final_evaluation"
)
FINAL_TRAINING_SEED_COUNT = 5

NORMAL_95_CRITICAL_VALUE = (
    1.959963984540054
)

STUDENT_T_95_DF4_CRITICAL_VALUE = (
    2.7764451051977987
)


@dataclass(frozen=True, slots=True)
class EvaluationTargetData:
    identifier: str
    summary_path: Path
    rounds_path: Path
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class FinalResultsData:
    final_training_manifest: dict[str, object]
    extended_training_manifest: dict[str, object]
    final_evaluation_manifest: dict[str, object]
    evaluation_targets: tuple[
        EvaluationTargetData,
        ...
    ]


def load_json(
    path: Path,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    loaded = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        loaded,
        dict,
    ):
        raise ValueError(
            f"{path} must contain a JSON object"
        )

    return cast(
        dict[str, object],
        loaded,
    )


def load_final_results(
    *,
    final_training_dir: Path = (
        DEFAULT_FINAL_TRAINING_DIR
    ),
    extended_training_dir: Path = (
        DEFAULT_EXTENDED_TRAINING_DIR
    ),
    final_evaluation_dir: Path = (
        DEFAULT_FINAL_EVALUATION_DIR
    ),
) -> FinalResultsData:
    final_training_manifest = load_json(
        final_training_dir
        / "manifest.json"
    )

    extended_training_manifest = load_json(
        extended_training_dir
        / "manifest.json"
    )

    final_evaluation_manifest = load_json(
        final_evaluation_dir
        / "manifest.json"
    )

    _validate_training_manifest(
        final_training_manifest,
        expected_runs=(
            FINAL_TRAINING_RUN_COUNT
        ),
        name="final training",
    )

    _validate_training_manifest(
        extended_training_manifest,
        expected_runs=(
            EXTENDED_TRAINING_RUN_COUNT
        ),
        name="extended training",
    )

    _validate_final_evaluation_manifest(
        final_evaluation_manifest
    )

    targets = _load_evaluation_targets(
        final_evaluation_dir,
        final_evaluation_manifest,
    )

    _validate_common_round_schedule(
        targets
    )

    return FinalResultsData(
        final_training_manifest=(
            final_training_manifest
        ),
        extended_training_manifest=(
            extended_training_manifest
        ),
        final_evaluation_manifest=(
            final_evaluation_manifest
        ),
        evaluation_targets=targets,
    )


def _validate_training_manifest(
    manifest: dict[str, object],
    *,
    expected_runs: int,
    name: str,
) -> None:
    if (
        manifest.get("run_count")
        != expected_runs
    ):
        raise ValueError(
            f"{name} must contain "
            f"{expected_runs} runs"
        )

    if (
        manifest.get("completed_runs")
        != expected_runs
    ):
        raise ValueError(
            f"{name} is not complete"
        )

    if (
        manifest.get("partial_runs")
        != 0
    ):
        raise ValueError(
            f"{name} contains partial runs"
        )

    if (
        manifest.get("pending_runs")
        != 0
    ):
        raise ValueError(
            f"{name} contains pending runs"
        )


def _validate_final_evaluation_manifest(
    manifest: dict[str, object],
) -> None:
    if (
        manifest.get("target_count")
        != FINAL_EVALUATION_TARGET_COUNT
    ):
        raise ValueError(
            "final evaluation must contain "
            "52 targets"
        )

    if (
        manifest.get("completed_targets")
        != FINAL_EVALUATION_TARGET_COUNT
    ):
        raise ValueError(
            "final evaluation is not complete"
        )

    if (
        manifest.get("round_count")
        != FINAL_EVALUATION_ROUNDS
    ):
        raise ValueError(
            "final evaluation must use "
            "100000 rounds"
        )

    for key in (
        "partial_targets",
        "pending_targets",
        "missing_checkpoint_targets",
    ):
        if manifest.get(key) != 0:
            raise ValueError(
                "final evaluation contains "
                f"non-completed targets: {key}"
            )


def _load_evaluation_targets(
    evaluation_dir: Path,
    manifest: dict[str, object],
) -> tuple[
    EvaluationTargetData,
    ...
]:
    raw_targets = manifest.get(
        "targets"
    )

    if not isinstance(
        raw_targets,
        list,
    ):
        raise ValueError(
            "evaluation manifest targets "
            "must be a list"
        )

    targets: list[
        EvaluationTargetData
    ] = []

    for raw_target in raw_targets: # type: ignore
        if not isinstance(
            raw_target,
            dict,
        ):
            raise ValueError(
                "evaluation target "
                "must be an object"
            )

        target = cast(
            dict[str, object],
            raw_target,
        )

        identifier = target.get(
            "identifier"
        )

        output_dir_raw = target.get(
            "output_dir"
        )

        if not isinstance(
            identifier,
            str,
        ):
            raise ValueError(
                "target identifier "
                "must be a string"
            )

        if not isinstance(
            output_dir_raw,
            str,
        ):
            raise ValueError(
                "target output_dir "
                "must be a string"
            )

        output_dir = Path(
            output_dir_raw
        )

        if not output_dir.is_absolute():
            output_dir = (
                Path.cwd()
                / output_dir
            )

        summary_path = (
            output_dir / "summary.json"
        )

        rounds_path = (
            output_dir
            / "evaluation_rounds.csv"
        )

        targets.append(
            EvaluationTargetData(
                identifier=identifier,
                summary_path=summary_path,
                rounds_path=rounds_path,
                summary=load_json(
                    summary_path
                ),
            )
        )

    if (
        len(targets)
        != FINAL_EVALUATION_TARGET_COUNT
    ):
        raise ValueError(
            "expected exactly "
            "52 evaluation targets"
        )

    return tuple(
        targets
    )


def _load_round_keys(
    path: Path,
) -> tuple[
    tuple[int, int],
    ...
]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    keys: list[
        tuple[int, int]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            keys.append(
                (
                    int(
                        row["round_index"]
                    ),
                    int(
                        row["deck_seed"]
                    ),
                )
            )

    if (
        len(keys)
        != FINAL_EVALUATION_ROUNDS
    ):
        raise ValueError(
            f"{path} must contain "
            "100000 rounds"
        )

    return tuple(
        keys
    )


def _validate_common_round_schedule(
    targets: tuple[
        EvaluationTargetData,
        ...
    ],
) -> None:
    reference = _load_round_keys(
        targets[0].rounds_path
    )

    for target in targets[1:]:
        candidate = _load_round_keys(
            target.rounds_path
        )

        if candidate != reference:
            raise ValueError(
                "evaluation targets do not "
                "share the same round schedule"
            )

@dataclass(frozen=True, slots=True)
class MeanEstimate:
    sample_count: int
    mean: float
    standard_deviation: float
    standard_error: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True, slots=True)
class ModelTargetMetadata:
    algorithm: str
    state_representation: str
    reward_type: str
    training_seed: int
    training_episodes: int

def summarize_values(
    values: tuple[float, ...],
    *,
    critical_value: float,
) -> MeanEstimate:
    if len(values) < 2:
        raise ValueError(
            "at least two values are required"
        )

    if (
        not isinstance(
            critical_value,
            (int, float),
        ) # type: ignore
        or not isfinite(
            float(critical_value)
        )
        or critical_value <= 0
    ):
        raise ValueError(
            "critical_value must be "
            "a positive finite number"
        )

    if not all(
        isfinite(value)
        for value in values
    ):
        raise ValueError(
            "values must be finite"
        )

    value_mean = fmean(
        values
    )

    standard_deviation = stdev(
        values
    )

    standard_error = (
        standard_deviation
        / sqrt(
            len(values)
        )
    )

    margin = (
        float(critical_value)
        * standard_error
    )

    return MeanEstimate(
        sample_count=len(values),
        mean=value_mean,
        standard_deviation=(
            standard_deviation
        ),
        standard_error=(
            standard_error
        ),
        ci95_low=value_mean - margin,
        ci95_high=value_mean + margin,
    )


def get_evaluation_ev(
    target: EvaluationTargetData,
) -> float:
    raw_evaluation = target.summary.get(
        "evaluation"
    )

    if not isinstance(
        raw_evaluation,
        dict,
    ):
        raise ValueError(
            f"{target.identifier} "
            "does not contain evaluation data"
        )

    evaluation = cast(
        dict[str, object],
        raw_evaluation,
    )

    estimated_ev = evaluation.get(
        "estimated_ev"
    )

    if type(estimated_ev) not in (
        int,
        float,
    ):
        raise ValueError(
            f"{target.identifier} "
            "has invalid estimated_ev"
        )

    estimated_ev = cast(
        int | float,
        estimated_ev,
    )

    result = float(
        estimated_ev
    )

    if not isfinite(
        result
    ):
        raise ValueError(
            f"{target.identifier} "
            "has non-finite estimated_ev"
        )

    return result


def get_model_metadata(
    target: EvaluationTargetData,
) -> ModelTargetMetadata | None:
    raw_target = target.summary.get(
        "target"
    )

    if not isinstance(
        raw_target,
        dict,
    ):
        raise ValueError(
            f"{target.identifier} "
            "does not contain target metadata"
        )

    target_metadata = cast(
        dict[str, object],
        raw_target,
    )

    raw_variant = target_metadata.get(
        "variant"
    )

    if raw_variant is None:
        return None

    if not isinstance(
        raw_variant,
        dict,
    ):
        raise ValueError(
            f"{target.identifier} "
            "has invalid variant metadata"
        )

    variant = cast(
        dict[str, object],
        raw_variant,
    )

    algorithm = variant.get(
        "algorithm"
    )

    state_representation = variant.get(
        "state_representation"
    )

    reward_type = variant.get(
        "reward_type"
    )

    training_seed = target_metadata.get(
        "training_seed"
    )

    training_episodes = target_metadata.get(
        "training_episodes"
    )

    if not isinstance(
        algorithm,
        str,
    ):
        raise ValueError(
            "algorithm must be a string"
        )

    if not isinstance(
        state_representation,
        str,
    ):
        raise ValueError(
            "state representation "
            "must be a string"
        )

    if not isinstance(
        reward_type,
        str,
    ):
        raise ValueError(
            "reward type must be a string"
        )

    if type(training_seed) is not int:
        raise ValueError(
            "training seed must be an integer"
        )

    if type(training_episodes) is not int:
        raise ValueError(
            "training episodes "
            "must be an integer"
        )

    return ModelTargetMetadata(
        algorithm=algorithm,
        state_representation=(
            state_representation
        ),
        reward_type=reward_type,
        training_seed=training_seed,
        training_episodes=(
            training_episodes
        ),
    )


def find_model_targets(
    results: FinalResultsData,
    *,
    algorithm: str,
    state_representation: str,
    reward_type: str,
    training_episodes: int,
) -> tuple[
    EvaluationTargetData,
    ...
]:
    matches: list[
        EvaluationTargetData
    ] = []

    for target in results.evaluation_targets:
        metadata = get_model_metadata(
            target
        )

        if metadata is None:
            continue

        if (
            metadata.algorithm
            == algorithm
            and metadata.state_representation
            == state_representation
            and metadata.reward_type
            == reward_type
            and metadata.training_episodes
            == training_episodes
        ):
            matches.append(
                target
            )

    if (
        len(matches)
        != FINAL_TRAINING_SEED_COUNT
    ):
        raise ValueError(
            "expected exactly five "
            "matching training seeds"
        )

    seeds = {
        cast(
            ModelTargetMetadata,
            get_model_metadata(target),
        ).training_seed
        for target in matches
    }

    if (
        len(seeds)
        != FINAL_TRAINING_SEED_COUNT
    ):
        raise ValueError(
            "model targets must contain "
            "five unique training seeds"
        )

    return tuple(
        matches
    )


def aggregate_model_ev(
    targets: tuple[
        EvaluationTargetData,
        ...
    ],
) -> MeanEstimate:
    if (
        len(targets)
        != FINAL_TRAINING_SEED_COUNT
    ):
        raise ValueError(
            "model aggregation requires "
            "five training seeds"
        )

    values = tuple(
        get_evaluation_ev(
            target
        )
        for target in targets
    )

    return summarize_values(
        values,
        critical_value=(
            STUDENT_T_95_DF4_CRITICAL_VALUE
        ),
    )


def paired_seed_comparison(
    left_targets: tuple[
        EvaluationTargetData,
        ...
    ],
    right_targets: tuple[
        EvaluationTargetData,
        ...
    ],
) -> MeanEstimate:
    left_by_seed = (
        _evaluation_ev_by_seed(
            left_targets
        )
    )

    right_by_seed = (
        _evaluation_ev_by_seed(
            right_targets
        )
    )

    if (
        left_by_seed.keys()
        != right_by_seed.keys()
    ):
        raise ValueError(
            "paired model groups must use "
            "the same training seeds"
        )

    differences = tuple(
        left_by_seed[seed]
        - right_by_seed[seed]
        for seed in sorted(
            left_by_seed
        )
    )

    return summarize_values(
        differences,
        critical_value=(
            STUDENT_T_95_DF4_CRITICAL_VALUE
        ),
    )


def _evaluation_ev_by_seed(
    targets: tuple[
        EvaluationTargetData,
        ...
    ],
) -> dict[int, float]:
    if (
        len(targets)
        != FINAL_TRAINING_SEED_COUNT
    ):
        raise ValueError(
            "paired seed comparison "
            "requires five targets"
        )

    result: dict[
        int,
        float,
    ] = {}

    for target in targets:
        metadata = get_model_metadata(
            target
        )

        if metadata is None:
            raise ValueError(
                "baseline target cannot be "
                "used in seed comparison"
            )

        if (
            metadata.training_seed
            in result
        ):
            raise ValueError(
                "duplicate training seed"
            )

        result[
            metadata.training_seed
        ] = get_evaluation_ev(
            target
        )

    return result


def find_target(
    results: FinalResultsData,
    identifier: str,
) -> EvaluationTargetData:
    matches = tuple(
        target
        for target
        in results.evaluation_targets
        if target.identifier == identifier
    )

    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one target "
            f"with identifier {identifier}"
        )

    return matches[0]


def _load_round_net_profits(
    path: Path,
) -> tuple[
    Fraction,
    ...
]:
    values: list[
        Fraction
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            values.append(
                Fraction(
                    row["net_profit"]
                )
            )

    if (
        len(values)
        != FINAL_EVALUATION_ROUNDS
    ):
        raise ValueError(
            f"{path} must contain "
            "100000 net profit values"
        )

    return tuple(
        values
    )


def paired_round_comparison(
    left: EvaluationTargetData,
    right: EvaluationTargetData,
) -> MeanEstimate:
    left_values = (
        _load_round_net_profits(
            left.rounds_path
        )
    )

    right_values = (
        _load_round_net_profits(
            right.rounds_path
        )
    )

    if (
        len(left_values)
        != len(right_values)
    ):
        raise ValueError(
            "paired evaluations must have "
            "the same round count"
        )

    differences = tuple(
        float(
            left_value
            - right_value
        )
        for (
            left_value,
            right_value,
        ) in zip(
            left_values,
            right_values,
        )
    )

    return summarize_values(
        differences,
        critical_value=(
            NORMAL_95_CRITICAL_VALUE
        ),
    )




def main() -> None:
    results = load_final_results()

    print(
        "Final results validated"
    )

    print(
        f"Evaluation targets: "
        f"{len(results.evaluation_targets)}"
    )

    print(
        "Rounds per target: "
        f"{FINAL_EVALUATION_ROUNDS}"
    )

    random_target = find_target(
        results,
        "baseline_random",
    )

    rule_based_target = find_target(
        results,
        "baseline_rule_based",
    )

    baseline_comparison = (
        paired_round_comparison(
            rule_based_target,
            random_target,
        )
    )

    print()
    print(
        "RuleBased - Random"
    )

    print(
        f"Mean difference: "
        f"{baseline_comparison.mean:.6f}"
    )

    print(
        "95% CI: "
        f"["
        f"{baseline_comparison.ci95_low:.6f}, "
        f"{baseline_comparison.ci95_high:.6f}"
        f"]"
    )

    dqn_features = find_model_targets(
        results,
        algorithm="dqn",
        state_representation="features",
        reward_type="net_profit",
        training_episodes=50_000,
    )

    dqn_raw = find_model_targets(
        results,
        algorithm="dqn",
        state_representation="raw",
        reward_type="net_profit",
        training_episodes=50_000,
    )

    state_comparison = (
        paired_seed_comparison(
            dqn_features,
            dqn_raw,
        )
    )

    print()
    print(
        "DQN FEATURES - RAW, NET_PROFIT"
    )

    print(
        f"Mean difference: "
        f"{state_comparison.mean:.6f}"
    )

    print(
        "95% CI: "
        f"["
        f"{state_comparison.ci95_low:.6f}, "
        f"{state_comparison.ci95_high:.6f}"
        f"]"
    )


if __name__ == "__main__":
    main()