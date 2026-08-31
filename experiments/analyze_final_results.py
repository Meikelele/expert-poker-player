import csv
import json

from dataclasses import dataclass
from pathlib import Path
from typing import cast


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


if __name__ == "__main__":
    main()