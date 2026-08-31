import csv

from collections import Counter
from dataclasses import dataclass
from math import pi
from pathlib import Path
from statistics import fmean, stdev
from typing import cast

import matplotlib

matplotlib.use(
    "Agg"
)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from expert_poker_player.cards import (
    Rank,
)
from expert_poker_player.hands.hand_rank import (
    HandRank,
)
from experiments.analyze_final_results import (
    STUDENT_T_95_DF4_CRITICAL_VALUE,
    EvaluationTargetData,
    FinalResultsData,
    MeanEstimate,
    ModelTargetMetadata,
    aggregate_model_ev,
    find_model_targets,
    find_target,
    get_evaluation_ev,
    get_model_metadata,
    load_deck_seeds,
    load_final_results,
    load_round_actions,
    load_round_net_profits,
    paired_seed_comparison,
    reconstruct_postflop_hand_ranks,
    reconstruct_preflop_hole_cards,
    summarize_values,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/results/final_analysis"
)

FINAL_TRAINING_EPISODES = 50_000
EXTENDED_TRAINING_EPISODES = 100_000

NET_PROFIT = "net_profit"
SCALED_NET_PROFIT = (
    "stake_scaled_net_profit"
)

RAW = "raw"
FEATURES = "features"

DQN = "dqn"
REINFORCE = "reinforce"

# Wizard of Odds, przewaga kasyna w Ultimate Texas Hold'em (por.
# tabela~\ref{tab:house_edge_examples} w rozdziale 2 pracy).
UTH_HOUSE_EDGE = 0.0219

PREFLOP_ACTIONS = (
    "CHECK",
    "BET_3X",
    "BET_4X",
)

PREFLOP_ACTION_COLORS = {
    "CHECK": "#4C72B0",
    "BET_3X": "#DD8452",
    "BET_4X": "#C44E52",
}

RANK_ORDER = (
    Rank.ACE,
    Rank.KING,
    Rank.QUEEN,
    Rank.JACK,
    Rank.TEN,
    Rank.NINE,
    Rank.EIGHT,
    Rank.SEVEN,
    Rank.SIX,
    Rank.FIVE,
    Rank.FOUR,
    Rank.THREE,
    Rank.TWO,
)

RANK_LABELS = tuple(
    rank.symbol
    for rank in RANK_ORDER
)

ACTION_COLORS = {
    "CHECK": "#4C72B0",
    "BET_4X": "#C44E52",
    "BET_3X": "#DD8452",
    "BET_2X": "#55A868",
    "BET_1X": "#8172B2",
    "FOLD": "#8C8C8C",
}

STREETS = (
    (
        "Preflop",
        0,
        (
            "CHECK",
            "BET_3X",
            "BET_4X",
        ),
    ),
    (
        "Flop",
        1,
        (
            "CHECK",
            "BET_2X",
        ),
    ),
    (
        "River",
        2,
        (
            "BET_1X",
            "FOLD",
        ),
    ),
)

HAND_RANK_ORDER = (
    HandRank.HIGH_CARD,
    HandRank.ONE_PAIR,
    HandRank.TWO_PAIR,
    HandRank.THREE_OF_A_KIND,
    HandRank.STRAIGHT,
    HandRank.FLUSH,
    HandRank.FULL_HOUSE,
    HandRank.FOUR_OF_A_KIND,
    HandRank.STRAIGHT_FLUSH,
)

DECISION_PATHS = (
    (
        "BET_4X",
    ),
    (
        "BET_3X",
    ),
    (
        "CHECK",
        "BET_2X",
    ),
    (
        "CHECK",
        "CHECK",
        "BET_1X",
    ),
    (
        "CHECK",
        "CHECK",
        "FOLD",
    ),
)

DECISION_PATH_LABELS = {
    (
        "BET_4X",
    ): "Bet 4x\n(preflop)",
    (
        "BET_3X",
    ): "Bet 3x\n(preflop)",
    (
        "CHECK",
        "BET_2X",
    ): "Check, bet 2x\n(flop)",
    (
        "CHECK",
        "CHECK",
        "BET_1X",
    ): "Check, check,\nbet 1x (river)",
    (
        "CHECK",
        "CHECK",
        "FOLD",
    ): "Check, check,\nfold (river)",
}

HAND_RANK_LABELS = {
    HandRank.HIGH_CARD: "Wysoka\nkarta",
    HandRank.ONE_PAIR: "Para",
    HandRank.TWO_PAIR: "Dwie\npary",
    HandRank.THREE_OF_A_KIND: "Trójka",
    HandRank.STRAIGHT: "Strit",
    HandRank.FLUSH: "Kolor",
    HandRank.FULL_HOUSE: "Full",
    HandRank.FOUR_OF_A_KIND: "Kareta",
    HandRank.STRAIGHT_FLUSH: "Poker",
}


BEST_VARIANTS = (
    (
        DQN,
        FEATURES,
        NET_PROFIT,
    ),
    (
        REINFORCE,
        FEATURES,
        SCALED_NET_PROFIT,
    ),
)


VARIANT_SPECS_50K = (
    (
        DQN,
        RAW,
        NET_PROFIT,
    ),
    (
        DQN,
        RAW,
        SCALED_NET_PROFIT,
    ),
    (
        DQN,
        FEATURES,
        NET_PROFIT,
    ),
    (
        DQN,
        FEATURES,
        SCALED_NET_PROFIT,
    ),
    (
        REINFORCE,
        RAW,
        NET_PROFIT,
    ),
    (
        REINFORCE,
        RAW,
        SCALED_NET_PROFIT,
    ),
    (
        REINFORCE,
        FEATURES,
        NET_PROFIT,
    ),
    (
        REINFORCE,
        FEATURES,
        SCALED_NET_PROFIT,
    ),
)


@dataclass(frozen=True, slots=True)
class ValidationPoint:
    completed_episodes: int
    estimated_ev: float


@dataclass(frozen=True, slots=True)
class TrainingRun:
    algorithm: str
    state_representation: str
    reward_type: str
    training_seed: int
    training_episodes: int
    points: tuple[
        ValidationPoint,
        ...
    ]
    training_diagnostics_path: str | None


@dataclass(frozen=True, slots=True)
class AggregateCurvePoint:
    completed_episodes: int
    mean_ev: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class DiagnosticsPoint:
    episode: int
    gradient_norm: float


@dataclass(frozen=True, slots=True)
class AggregateDiagnosticsPoint:
    episode: int
    mean_gradient_norm: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class VariantSummary:
    algorithm: str
    state_representation: str
    reward_type: str
    estimate: MeanEstimate
    minimum_ev: float
    maximum_ev: float


@dataclass(frozen=True, slots=True)
class EffectSummary:
    algorithm: str
    context: str
    estimate: MeanEstimate


@dataclass(frozen=True, slots=True)
class TrainingBudgetSummary:
    algorithm: str
    state_representation: str
    reward_type: str
    ev_50k: MeanEstimate
    ev_100k: MeanEstimate
    improvement: MeanEstimate


def _resolve_path(
    raw_path: str,
) -> Path:
    path = Path(
        raw_path
    )

    if path.is_absolute():
        return path

    return (
        Path.cwd()
        / path
    )


def load_learning_curve(
    path: Path,
) -> tuple[
    ValidationPoint,
    ...
]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    points: list[
        ValidationPoint
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
            points.append(
                ValidationPoint(
                    completed_episodes=int(
                        row[
                            "completed_episodes"
                        ]
                    ),
                    estimated_ev=float(
                        row[
                            "estimated_ev"
                        ]
                    ),
                )
            )

    if not points:
        raise ValueError(
            f"{path} contains no "
            "learning curve points"
        )

    checkpoints = tuple(
        point.completed_episodes
        for point in points
    )

    if (
        len(set(checkpoints))
        != len(checkpoints)
    ):
        raise ValueError(
            f"{path} contains duplicate "
            "checkpoints"
        )

    return tuple(
        points
    )


def load_training_runs(
    manifest: dict[str, object],
) -> tuple[
    TrainingRun,
    ...
]:
    raw_runs = manifest.get(
        "runs"
    )

    training_episodes = manifest.get(
        "training_episodes"
    )

    if not isinstance(
        raw_runs,
        list,
    ):
        raise ValueError(
            "manifest runs must be a list"
        )

    if type(training_episodes) is not int:
        raise ValueError(
            "training_episodes "
            "must be an integer"
        )

    result: list[
        TrainingRun
    ] = []

    for raw_run in raw_runs: # type: ignore
        if not isinstance(
            raw_run,
            dict,
        ):
            raise ValueError(
                "run must be an object"
            )

        run = cast(
            dict[str, object],
            raw_run,
        )

        raw_variant = run.get(
            "variant"
        )

        training_seed = run.get(
            "training_seed"
        )

        learning_curve_path = run.get(
            "learning_curve_path"
        )

        training_diagnostics_path = run.get(
            "training_diagnostics_path"
        )

        if not isinstance(
            raw_variant,
            dict,
        ):
            raise ValueError(
                "run variant must be an object"
            )

        variant = cast(
            dict[str, object],
            raw_variant,
        )

        algorithm = variant.get(
            "algorithm"
        )

        state_representation = (
            variant.get(
                "state_representation"
            )
        )

        reward_type = variant.get(
            "reward_type"
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
                "training seed "
                "must be an integer"
            )

        if not isinstance(
            learning_curve_path,
            str,
        ):
            raise ValueError(
                "learning curve path "
                "must be a string"
            )

        if (
            training_diagnostics_path is not None
            and not isinstance(
                training_diagnostics_path,
                str,
            )
        ):
            raise ValueError(
                "training diagnostics path "
                "must be a string or None"
            )

        result.append(
            TrainingRun(
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=reward_type,
                training_seed=training_seed,
                training_episodes=(
                    training_episodes
                ),
                points=load_learning_curve(
                    _resolve_path(
                        learning_curve_path
                    )
                ),
                training_diagnostics_path=(
                    training_diagnostics_path
                ),
            )
        )

    return tuple(
        result
    )


def aggregate_learning_curves(
    runs: tuple[
        TrainingRun,
        ...
    ],
) -> tuple[
    AggregateCurvePoint,
    ...
]:
    if len(runs) != 5:
        raise ValueError(
            "learning curve aggregation "
            "requires five runs"
        )

    reference_checkpoints = tuple(
        point.completed_episodes
        for point in runs[0].points
    )

    for run in runs[1:]:
        checkpoints = tuple(
            point.completed_episodes
            for point in run.points
        )

        if (
            checkpoints
            != reference_checkpoints
        ):
            raise ValueError(
                "learning curves must use "
                "the same checkpoints"
            )

    aggregates: list[
        AggregateCurvePoint
    ] = []

    for index, completed_episodes in enumerate(
        reference_checkpoints
    ):
        values = tuple(
            run.points[
                index
            ].estimated_ev
            for run in runs
        )

        aggregates.append(
            AggregateCurvePoint(
                completed_episodes=(
                    completed_episodes
                ),
                mean_ev=fmean(
                    values
                ),
                standard_deviation=stdev(
                    values
                ),
            )
        )

    return tuple(
        aggregates
    )


def load_training_diagnostics(
    path: Path,
) -> tuple[
    DiagnosticsPoint,
    ...
]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    points: list[
        DiagnosticsPoint
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
            points.append(
                DiagnosticsPoint(
                    episode=int(
                        row[
                            "episode"
                        ]
                    ),
                    gradient_norm=float(
                        row[
                            "gradient_norm"
                        ]
                    ),
                )
            )

    if not points:
        raise ValueError(
            f"{path} contains no "
            "training diagnostics points"
        )

    episodes = tuple(
        point.episode
        for point in points
    )

    if (
        len(set(episodes))
        != len(episodes)
    ):
        raise ValueError(
            f"{path} contains duplicate "
            "episodes"
        )

    return tuple(
        points
    )


def aggregate_training_diagnostics(
    runs: tuple[
        TrainingRun,
        ...
    ],
) -> tuple[
    AggregateDiagnosticsPoint,
    ...
]:
    if len(runs) != 5:
        raise ValueError(
            "training diagnostics aggregation "
            "requires five runs"
        )

    if any(
        run.training_diagnostics_path is None
        for run in runs
    ):
        raise ValueError(
            "training diagnostics aggregation requires "
            "runs with a training_diagnostics_path"
        )

    run_points = tuple(
        load_training_diagnostics(
            _resolve_path(
                run.training_diagnostics_path
            )
        )
        for run in runs
    )

    # DQN rejestruje punkt tylko dla epizodów z co najmniej jednym
    # update'em, a moment przekroczenia `warmup_steps` przypada na inny
    # numer epizodu w każdym seedzie (epizody mają losową długość), więc
    # zbiory epizodów seedów mogą się nieznacznie różnić tuż po starcie.
    # Agreguję więc po przecięciu zbiorów epizodów, nie po ich pełnej
    # zgodności.
    common_episodes = tuple(
        sorted(
            set.intersection(
                *(
                    {
                        point.episode
                        for point in points
                    }
                    for points in run_points
                )
            )
        )
    )

    if not common_episodes:
        raise ValueError(
            "training diagnostics runs share "
            "no common episodes"
        )

    values_by_episode = tuple(
        {
            point.episode: point.gradient_norm
            for point in points
        }
        for points in run_points
    )

    aggregates: list[
        AggregateDiagnosticsPoint
    ] = []

    for episode in common_episodes:
        values = tuple(
            lookup[episode]
            for lookup in values_by_episode
        )

        aggregates.append(
            AggregateDiagnosticsPoint(
                episode=episode,
                mean_gradient_norm=fmean(
                    values
                ),
                standard_deviation=stdev(
                    values
                ),
            )
        )

    return tuple(
        aggregates
    )


def _find_training_runs(
    runs: tuple[
        TrainingRun,
        ...
    ],
    *,
    algorithm: str,
    state_representation: str,
    reward_type: str,
) -> tuple[
    TrainingRun,
    ...
]:
    matches = tuple(
        run
        for run in runs
        if (
            run.algorithm == algorithm
            and run.state_representation
            == state_representation
            and run.reward_type
            == reward_type
        )
    )

    if len(matches) != 5:
        raise ValueError(
            "expected five matching "
            "training runs"
        )

    return matches


def build_variant_summaries(
    results: FinalResultsData,
) -> tuple[
    VariantSummary,
    ...
]:
    summaries: list[
        VariantSummary
    ] = []

    for (
        algorithm,
        state_representation,
        reward_type,
    ) in VARIANT_SPECS_50K:
        targets = find_model_targets(
            results,
            algorithm=algorithm,
            state_representation=(
                state_representation
            ),
            reward_type=reward_type,
            training_episodes=(
                FINAL_TRAINING_EPISODES
            ),
        )

        values = tuple(
            get_evaluation_ev(
                target
            )
            for target in targets
        )

        summaries.append(
            VariantSummary(
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=reward_type,
                estimate=aggregate_model_ev(
                    targets
                ),
                minimum_ev=min(
                    values
                ),
                maximum_ev=max(
                    values
                ),
            )
        )

    return tuple(
        summaries
    )


def build_state_effects(
    results: FinalResultsData,
) -> tuple[
    EffectSummary,
    ...
]:
    effects: list[
        EffectSummary
    ] = []

    for algorithm in (
        DQN,
        REINFORCE,
    ):
        for reward_type in (
            NET_PROFIT,
            SCALED_NET_PROFIT,
        ):
            features_targets = (
                find_model_targets(
                    results,
                    algorithm=algorithm,
                    state_representation=(
                        FEATURES
                    ),
                    reward_type=reward_type,
                    training_episodes=(
                        FINAL_TRAINING_EPISODES
                    ),
                )
            )

            raw_targets = (
                find_model_targets(
                    results,
                    algorithm=algorithm,
                    state_representation=RAW,
                    reward_type=reward_type,
                    training_episodes=(
                        FINAL_TRAINING_EPISODES
                    ),
                )
            )

            effects.append(
                EffectSummary(
                    algorithm=algorithm,
                    context=reward_type,
                    estimate=(
                        paired_seed_comparison(
                            features_targets,
                            raw_targets,
                        )
                    ),
                )
            )

    return tuple(
        effects
    )


def build_reward_effects(
    results: FinalResultsData,
) -> tuple[
    EffectSummary,
    ...
]:
    effects: list[
        EffectSummary
    ] = []

    for algorithm in (
        DQN,
        REINFORCE,
    ):
        for state_representation in (
            RAW,
            FEATURES,
        ):
            net_targets = find_model_targets(
                results,
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=NET_PROFIT,
                training_episodes=(
                    FINAL_TRAINING_EPISODES
                ),
            )

            scaled_targets = (
                find_model_targets(
                    results,
                    algorithm=algorithm,
                    state_representation=(
                        state_representation
                    ),
                    reward_type=(
                        SCALED_NET_PROFIT
                    ),
                    training_episodes=(
                        FINAL_TRAINING_EPISODES
                    ),
                )
            )

            effects.append(
                EffectSummary(
                    algorithm=algorithm,
                    context=(
                        state_representation
                    ),
                    estimate=(
                        paired_seed_comparison(
                            net_targets,
                            scaled_targets,
                        )
                    ),
                )
            )

    return tuple(
        effects
    )


def build_training_budget_summaries(
    results: FinalResultsData,
) -> tuple[
    TrainingBudgetSummary,
    ...
]:
    specs = BEST_VARIANTS

    summaries: list[
        TrainingBudgetSummary
    ] = []

    for (
        algorithm,
        state_representation,
        reward_type,
    ) in specs:
        targets_50k = find_model_targets(
            results,
            algorithm=algorithm,
            state_representation=(
                state_representation
            ),
            reward_type=reward_type,
            training_episodes=(
                FINAL_TRAINING_EPISODES
            ),
        )

        targets_100k = find_model_targets(
            results,
            algorithm=algorithm,
            state_representation=(
                state_representation
            ),
            reward_type=reward_type,
            training_episodes=(
                EXTENDED_TRAINING_EPISODES
            ),
        )

        summaries.append(
            TrainingBudgetSummary(
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=reward_type,
                ev_50k=aggregate_model_ev(
                    targets_50k
                ),
                ev_100k=aggregate_model_ev(
                    targets_100k
                ),
                improvement=(
                    paired_seed_comparison(
                        targets_100k,
                        targets_50k,
                    )
                ),
            )
        )

    return tuple(
        summaries
    )


def _algorithm_label(
    algorithm: str,
) -> str:
    if algorithm == DQN:
        return "DQN"

    if algorithm == REINFORCE:
        return "REINFORCE"

    return algorithm


def _state_label(
    state_representation: str,
) -> str:
    if state_representation == RAW:
        return "RAW"

    if state_representation == FEATURES:
        return "FEATURES"

    return state_representation


def _reward_label(
    reward_type: str,
) -> str:
    if reward_type == NET_PROFIT:
        return "zysk netto"

    if reward_type == SCALED_NET_PROFIT:
        return "skalowana"

    return reward_type


def _variant_label(
    algorithm: str,
    state_representation: str,
    reward_type: str,
) -> str:
    return (
        f"{_algorithm_label(algorithm)} | "
        f"{_state_label(state_representation)} | "
        f"{_reward_label(reward_type)}"
    )


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _save_figure(
    figure: Figure,
    *,
    output_dir: Path,
    filename: str,
) -> None:
    figure.tight_layout()

    figure.savefig( # type: ignore
        output_dir
        / f"{filename}.png",
        bbox_inches="tight",
    )

    figure.savefig( # type: ignore
        output_dir
        / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_final_ev_50k(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    summaries = build_variant_summaries(
        results
    )

    random_ev = get_evaluation_ev(
        find_target(
            results,
            "baseline_random",
        )
    )

    rule_based_ev = get_evaluation_ev(
        find_target(
            results,
            "baseline_rule_based",
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            6.5,
        )
    )

    positions = list(
        range(
            len(summaries)
        )
    )

    seed_offsets = (
        -0.12,
        -0.06,
        0.0,
        0.06,
        0.12,
    )

    labels: list[str] = []

    for position, summary in zip(
        positions,
        summaries,
    ):
        targets = find_model_targets(
            results,
            algorithm=summary.algorithm,
            state_representation=(
                summary.state_representation
            ),
            reward_type=(
                summary.reward_type
            ),
            training_episodes=(
                FINAL_TRAINING_EPISODES
            ),
        )

        seed_values = tuple(
            get_evaluation_ev(
                target
            )
            for target in targets
        )

        color = (
            "C0"
            if summary.algorithm == DQN
            else "C1"
        )

        axis.scatter( # type: ignore
            seed_values,
            [
                position + offset
                for offset in seed_offsets
            ],
            s=24,
            alpha=0.55,
            color=color,
        )

        estimate = summary.estimate

        axis.errorbar( # type: ignore
            estimate.mean,
            position,
            xerr=[
                [
                    estimate.mean
                    - estimate.ci95_low
                ],
                [
                    estimate.ci95_high
                    - estimate.mean
                ],
            ],
            fmt="D",
            markersize=6,
            capsize=4,
            color=color,
        )

        labels.append(
            _variant_label(
                summary.algorithm,
                summary.state_representation,
                summary.reward_type,
            )
        )

    axis.axvline( # type: ignore
        random_ev,
        linestyle=":",
        linewidth=1.4,
        color="0.45",
        label="Random",
    )

    axis.axvline( # type: ignore
        rule_based_ev,
        linestyle="-.",
        linewidth=1.4,
        color="0.15",
        label="RuleBased",
    )

    axis.set_yticks( # type: ignore
        positions
    )

    axis.set_yticklabels( # type: ignore
        labels
    )

    axis.invert_yaxis()

    axis.set_xlabel( # type: ignore
        "EV [jednostki Ante / rozdanie]"
    )

    axis.set_title( # type: ignore
        "Końcowe EV modeli po 50 tys. epizodów"
    )

    axis.grid( # type: ignore
        axis="x",
        alpha=0.25,
    )

    axis.legend( # type: ignore
        loc="lower right"
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="01_final_ev_50k",
    )


def _plot_effects(
    effects: tuple[
        EffectSummary,
        ...
    ],
    *,
    output_dir: Path,
    filename: str,
    title: str,
    xlabel: str,
    context_is_reward: bool,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            8.5,
            4.8,
        )
    )

    positions = list(
        range(
            len(effects)
        )
    )

    labels: list[str] = []

    for position, effect in zip(
        positions,
        effects,
    ):
        color = (
            "C0"
            if effect.algorithm == DQN
            else "C1"
        )

        estimate = effect.estimate

        axis.errorbar( # type: ignore
            estimate.mean,
            position,
            xerr=[
                [
                    estimate.mean
                    - estimate.ci95_low
                ],
                [
                    estimate.ci95_high
                    - estimate.mean
                ],
            ],
            fmt="o",
            markersize=7,
            capsize=5,
            color=color,
        )

        context_label = (
            _reward_label(
                effect.context
            )
            if context_is_reward
            else _state_label(
                effect.context
            )
        )

        labels.append(
            f"{_algorithm_label(effect.algorithm)} | "
            f"{context_label}"
        )

    axis.axvline( # type: ignore
        0.0,
        color="black",
        linewidth=1.0,
    )

    axis.set_yticks( # type: ignore
        positions
    )

    axis.set_yticklabels( # type: ignore
        labels
    )

    axis.invert_yaxis()

    axis.set_xlabel( # type: ignore
        xlabel
    )

    axis.set_title( # type: ignore
        title
    )

    axis.grid( # type: ignore
        axis="x",
        alpha=0.25,
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename=filename,
    )


def plot_state_effects(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    _plot_effects(
        build_state_effects(
            results
        ),
        output_dir=output_dir,
        filename=(
            "02_state_representation_effect"
        ),
        title=(
            "Wpływ reprezentacji stanu "
            "na końcowe EV"
        ),
        xlabel=(
            "Różnica EV: FEATURES − RAW"
        ),
        context_is_reward=True,
    )


def plot_reward_effects(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    _plot_effects(
        build_reward_effects(
            results
        ),
        output_dir=output_dir,
        filename="03_reward_effect",
        title=(
            "Wpływ funkcji nagrody "
            "na końcowe EV"
        ),
        xlabel=(
            "Różnica EV: zysk netto − skalowana"
        ),
        context_is_reward=False,
    )


def plot_learning_curves(
    runs: tuple[
        TrainingRun,
        ...
    ],
    *,
    algorithm: str,
    output_dir: Path,
    filename: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            9.5,
            5.4,
        )
    )

    for state_representation in (
        RAW,
        FEATURES,
    ):
        for reward_type in (
            NET_PROFIT,
            SCALED_NET_PROFIT,
        ):
            matching_runs = (
                _find_training_runs(
                    runs,
                    algorithm=algorithm,
                    state_representation=(
                        state_representation
                    ),
                    reward_type=reward_type,
                )
            )

            aggregate = (
                aggregate_learning_curves(
                    matching_runs
                )
            )

            episodes = [
                point.completed_episodes
                for point in aggregate
            ]

            means = [
                point.mean_ev
                for point in aggregate
            ]

            lower = [
                point.mean_ev
                - point.standard_deviation
                for point in aggregate
            ]

            upper = [
                point.mean_ev
                + point.standard_deviation
                for point in aggregate
            ]

            label = (
                f"{_state_label(state_representation)} "
                f"+ {_reward_label(reward_type)}"
            )

            line = axis.plot( # type: ignore
                episodes,
                means,
                label=label,
                linewidth=1.8,
            )[0]

            axis.fill_between( # type: ignore
                episodes,
                lower,
                upper,
                alpha=0.12,
                color=line.get_color(),
            )

    axis.set_xlabel( # type: ignore
        "Liczba epizodów treningowych"
    )

    axis.set_ylabel( # type: ignore
        "Walidacyjne EV"
    )

    axis.set_title( # type: ignore
        f"{_algorithm_label(algorithm)} "
        "— przebieg uczenia"
    )

    axis.grid( # type: ignore
        alpha=0.25
    )

    axis.legend( # type: ignore
        ncols=2
    )

    figure.text( # type: ignore
        0.5,
        0.01,
        "Linia: średnia z 5 seedów. "
        "Pasmo: ±1 odchylenie standardowe.",
        ha="center",
        fontsize=9,
    )

    figure.subplots_adjust(
        bottom=0.14
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename=filename,
    )


def plot_extended_learning_curves(
    runs: tuple[
        TrainingRun,
        ...
    ],
    *,
    output_dir: Path,
) -> None:
    specs = BEST_VARIANTS

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            11,
            4.6,
        ),
    )

    for axis, spec in zip(
        axes,
        specs,
    ):
        (
            algorithm,
            state_representation,
            reward_type,
        ) = spec

        matching_runs = (
            _find_training_runs(
                runs,
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=reward_type,
            )
        )

        aggregate = (
            aggregate_learning_curves(
                matching_runs
            )
        )

        episodes = [
            point.completed_episodes
            for point in aggregate
        ]

        means = [
            point.mean_ev
            for point in aggregate
        ]

        lower = [
            point.mean_ev
            - point.standard_deviation
            for point in aggregate
        ]

        upper = [
            point.mean_ev
            + point.standard_deviation
            for point in aggregate
        ]

        line = axis.plot(
            episodes,
            means,
            linewidth=2.0,
        )[0]

        axis.fill_between(
            episodes,
            lower,
            upper,
            alpha=0.15,
            color=line.get_color(),
        )

        axis.axvline(
            50_000,
            linestyle="--",
            linewidth=1.0,
            color="0.35",
        )

        axis.set_title(
            _variant_label(
                algorithm,
                state_representation,
                reward_type,
            )
        )

        axis.set_xlabel(
            "Epizody"
        )

        axis.set_ylabel(
            "Walidacyjne EV"
        )

        axis.grid(
            alpha=0.25
        )

    figure.suptitle( # type: ignore
        "Trening rozszerzony do 100 tys. epizodów"
    )

    figure.text( # type: ignore
        0.5,
        0.01,
        "Linia: średnia z 5 seedów. "
        "Pasmo: ±1 odchylenie standardowe.",
        ha="center",
        fontsize=9,
    )

    figure.subplots_adjust(
        bottom=0.15
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename=(
            "06_extended_learning_curves"
        ),
    )


def _ev_by_seed(
    targets: tuple[
        EvaluationTargetData,
        ...
    ],
) -> dict[int, float]:
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
                "model metadata required"
            )

        result[
            metadata.training_seed
        ] = get_evaluation_ev(
            target
        )

    return result


def plot_training_budget_seed_pairs(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    specs = BEST_VARIANTS

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            10.5,
            4.8,
        ),
    )

    for axis, spec in zip(
        axes,
        specs,
    ):
        (
            algorithm,
            state_representation,
            reward_type,
        ) = spec

        targets_50k = (
            find_model_targets(
                results,
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=reward_type,
                training_episodes=(
                    FINAL_TRAINING_EPISODES
                ),
            )
        )

        targets_100k = (
            find_model_targets(
                results,
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=reward_type,
                training_episodes=(
                    EXTENDED_TRAINING_EPISODES
                ),
            )
        )

        values_50k = _ev_by_seed(
            targets_50k
        )

        values_100k = _ev_by_seed(
            targets_100k
        )

        seeds = sorted(
            values_50k
        )

        for index, seed in enumerate(
            seeds
        ):
            axis.plot(
                (
                    0,
                    1,
                ),
                (
                    values_50k[seed],
                    values_100k[seed],
                ),
                marker="o",
                linewidth=1.3,
                alpha=0.7,
                label=(
                    f"seed {index + 1}"
                ),
            )

        mean_50k = fmean(
            values_50k.values()
        )

        mean_100k = fmean(
            values_100k.values()
        )

        axis.plot(
            (
                0,
                1,
            ),
            (
                mean_50k,
                mean_100k,
            ),
            marker="D",
            linewidth=3.0,
            color="black",
            label="średnia",
        )

        axis.set_xticks(
            (
                0,
                1,
            )
        )

        axis.set_xticklabels(
            (
                "50 tys.",
                "100 tys.",
            )
        )

        axis.set_ylabel(
            "Końcowe EV"
        )

        axis.set_title(
            _algorithm_label(
                algorithm
            )
        )

        axis.grid(
            axis="y",
            alpha=0.25,
        )

    axes[1].legend(
        loc="best",
        fontsize=8,
    )

    figure.suptitle( # type: ignore
        "Wpływ zwiększenia budżetu treningowego"
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename=(
            "07_training_budget_seed_pairs"
        ),
    )


def plot_extended_training_per_seed(
    runs: tuple[
        TrainingRun,
        ...
    ],
    *,
    output_dir: Path,
) -> None:
    specs = BEST_VARIANTS

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            12,
            5,
        ),
        sharey=True,
    )

    for axis, spec in zip(
        axes,
        specs,
    ):
        (
            algorithm,
            state_representation,
            reward_type,
        ) = spec

        matching_runs = sorted(
            _find_training_runs(
                runs,
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=reward_type,
            ),
            key=lambda run: (
                run.training_seed
            ),
        )

        for index, run in enumerate(
            matching_runs
        ):
            axis.plot(
                [
                    point.completed_episodes
                    for point in run.points
                ],
                [
                    point.estimated_ev
                    for point in run.points
                ],
                linewidth=1.1,
                alpha=0.55,
                label=f"seed {index + 1}",
            )

        aggregate = (
            aggregate_learning_curves(
                tuple(
                    matching_runs
                )
            )
        )

        axis.plot(
            [
                point.completed_episodes
                for point in aggregate
            ],
            [
                point.mean_ev
                for point in aggregate
            ],
            linewidth=3.0,
            color="black",
            label="średnia",
        )

        axis.axvline(
            50_000,
            linestyle="--",
            linewidth=1.2,
            color="0.35",
        )

        axis.text(
            51_000,
            axis.get_ylim()[1],
            "50 tys.",
            va="top",
            fontsize=8,
        )

        axis.set_title(
            _variant_label(
                algorithm,
                state_representation,
                reward_type,
            )
        )

        axis.set_xlabel(
            "Liczba epizodów treningowych"
        )

        axis.grid(
            alpha=0.25
        )

    axes[0].set_ylabel(
        "Walidacyjne EV"
    )

    axes[1].legend(
        fontsize=8,
        loc="best",
    )

    figure.suptitle( # type: ignore
        "Trajektorie uczenia dla poszczególnych seedów"
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename=(
            "10_extended_training_per_seed"
        ),
    )


def plot_seed_variability(
    runs: tuple[
        TrainingRun,
        ...
    ],
    *,
    output_dir: Path,
) -> None:
    specs = (
        (
            DQN,
            FEATURES,
            NET_PROFIT,
            "DQN",
        ),
        (
            REINFORCE,
            FEATURES,
            SCALED_NET_PROFIT,
            "REINFORCE",
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for (
        algorithm,
        state_representation,
        reward_type,
        label,
    ) in specs:
        matching_runs = (
            _find_training_runs(
                runs,
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=reward_type,
            )
        )

        aggregate = (
            aggregate_learning_curves(
                matching_runs
            )
        )

        axis.plot( # type: ignore
            [
                point.completed_episodes
                for point in aggregate
            ],
            [
                point.standard_deviation
                for point in aggregate
            ],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=label,
        )

    axis.axvline( # type: ignore
        50_000,
        linestyle="--",
        linewidth=1.0,
        color="0.35",
    )

    axis.set_xlabel( # type: ignore
        "Liczba epizodów treningowych"
    )

    axis.set_ylabel( # type: ignore
        "SD walidacyjnego EV między seedami"
    )

    axis.set_title( # type: ignore
        "Zmienność wyników między niezależnymi treningami"
    )

    axis.grid( # type: ignore
        alpha=0.25
    )

    axis.legend() # type: ignore

    _save_figure(
        figure,
        output_dir=output_dir,
        filename=(
            "08_seed_variability_over_training"
        ),
    )


def _get_counts(
    target: EvaluationTargetData,
    *,
    field_name: str,
) -> dict[str, int]:
    raw_evaluation = target.summary.get(
        "evaluation"
    )

    if not isinstance(
        raw_evaluation,
        dict,
    ):
        raise ValueError(
            "evaluation data missing"
        )

    evaluation = cast(
        dict[str, object],
        raw_evaluation,
    )

    raw_counts = evaluation.get(
        field_name
    )

    if not isinstance(
        raw_counts,
        dict,
    ):
        raise ValueError(
            f"{field_name} missing"
        )

    counts = cast(
        dict[str, object],
        raw_counts,
    )

    return {
        str(key): int(count)  # type: ignore
        for key, count
        in counts.items()
    }


def _get_action_counts(
    target: EvaluationTargetData,
) -> dict[str, int]:
    return _get_counts(
        target,
        field_name="action_counts",
    )


def _get_outcome_counts(
    target: EvaluationTargetData,
) -> dict[str, int]:
    return _get_counts(
        target,
        field_name="outcome_counts",
    )


def _aggregate_outcome_counts(
    targets: tuple[
        EvaluationTargetData,
        ...
    ],
) -> dict[str, int]:
    totals: dict[
        str,
        int,
    ] = {}

    for target in targets:
        for outcome, count in _get_outcome_counts(
            target
        ).items():
            totals[outcome] = (
                totals.get(
                    outcome,
                    0,
                )
                + count
            )

    return totals


def plot_headline_comparison(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    random_target = find_target(
        results,
        "baseline_random",
    )

    rule_based_target = find_target(
        results,
        "baseline_rule_based",
    )

    (
        dqn_algorithm,
        dqn_state,
        dqn_reward,
    ) = BEST_VARIANTS[0]

    (
        reinforce_algorithm,
        reinforce_state,
        reinforce_reward,
    ) = BEST_VARIANTS[1]

    dqn_targets = find_model_targets(
        results,
        algorithm=dqn_algorithm,
        state_representation=dqn_state,
        reward_type=dqn_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    reinforce_targets = find_model_targets(
        results,
        algorithm=reinforce_algorithm,
        state_representation=(
            reinforce_state
        ),
        reward_type=reinforce_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            8.5,
            4.8,
        )
    )

    labels = (
        "Random",
        "RuleBased",
        f"DQN wybrany walidacyjnie ("
        f"{_state_label(dqn_state)} | "
        f"{_reward_label(dqn_reward)})",
        f"REINFORCE wybrany walidacyjnie ("
        f"{_state_label(reinforce_state)} | "
        f"{_reward_label(reinforce_reward)})",
    )

    positions = (
        0,
        1,
        2,
        3,
    )

    for position, target in zip(
        (0, 1),
        (
            random_target,
            rule_based_target,
        ),
    ):
        raw_evaluation = target.summary.get(
            "evaluation"
        )

        evaluation = cast(
            dict[str, object],
            raw_evaluation,
        )

        standard_error = cast(
            float,
            evaluation.get(
                "standard_error"
            ),
        )

        axis.errorbar( # type: ignore
            get_evaluation_ev(
                target
            ),
            position,
            xerr=(
                1.96
                * standard_error
            ),
            fmt="s",
            markersize=7,
            capsize=5,
            color="0.35",
        )

    for position, targets, color in (
        (
            2,
            dqn_targets,
            "C0",
        ),
        (
            3,
            reinforce_targets,
            "C1",
        ),
    ):
        seed_values = tuple(
            get_evaluation_ev(
                target
            )
            for target in targets
        )

        axis.scatter( # type: ignore
            seed_values,
            [
                position
                for _ in seed_values
            ],
            s=20,
            alpha=0.5,
            color=color,
        )

        estimate = aggregate_model_ev(
            targets
        )

        axis.errorbar( # type: ignore
            estimate.mean,
            position,
            xerr=[
                [
                    estimate.mean
                    - estimate.ci95_low
                ],
                [
                    estimate.ci95_high
                    - estimate.mean
                ],
            ],
            fmt="D",
            markersize=7,
            capsize=5,
            color=color,
        )

    axis.axvline( # type: ignore
        0.0,
        color="black",
        linewidth=1.0,
        linestyle=":",
    )

    axis.set_yticks( # type: ignore
        positions
    )

    axis.set_yticklabels( # type: ignore
        labels
    )

    axis.invert_yaxis()

    axis.set_xlabel( # type: ignore
        "EV [jednostki Ante / rozdanie]"
    )

    axis.set_title( # type: ignore
        "Konfiguracje wybrane na "
        "podstawie walidacji"
    )

    axis.grid( # type: ignore
        axis="x",
        alpha=0.25,
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="11_headline_comparison",
    )


def plot_payout_distribution(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    random_target = find_target(
        results,
        "baseline_random",
    )

    rule_based_target = find_target(
        results,
        "baseline_rule_based",
    )

    (
        dqn_algorithm,
        dqn_state,
        dqn_reward,
    ) = BEST_VARIANTS[0]

    (
        reinforce_algorithm,
        reinforce_state,
        reinforce_reward,
    ) = BEST_VARIANTS[1]

    dqn_targets = find_model_targets(
        results,
        algorithm=dqn_algorithm,
        state_representation=dqn_state,
        reward_type=dqn_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    reinforce_targets = find_model_targets(
        results,
        algorithm=reinforce_algorithm,
        state_representation=(
            reinforce_state
        ),
        reward_type=(
            reinforce_reward
        ),
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    series = (
        (
            "Random",
            (
                random_target,
            ),
            "0.35",
        ),
        (
            "RuleBased",
            (
                rule_based_target,
            ),
            "0.1",
        ),
        (
            "DQN",
            dqn_targets,
            "C0",
        ),
        (
            "REINFORCE",
            reinforce_targets,
            "C1",
        ),
    )

    # Wynik netto rozdania jest wielkością dyskretną (skończony zbiór
    # możliwych wypłat wynikających z tabel rozliczeń Ante/Blind/Play),
    # nie ciągłą, więc rysuję dokładne prawdopodobieństwa poszczególnych
    # wartości, a nie histogram z binami, który tworzyłby sztuczne,
    # puste przerwy między nimi.
    display_min = -8.0
    display_max = 16.0

    offsets = (
        -0.18,
        -0.06,
        0.06,
        0.18,
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for (label, targets, color), offset in zip(
        series,
        offsets,
    ):
        values = [
            value
            for target in targets
            for value in load_round_net_profits(
                target.rounds_path
            )
        ]

        total = len(
            values
        )

        counts: dict[
            float,
            int,
        ] = {}

        for value in values:
            numeric_value = float(
                value
            )

            counts[numeric_value] = (
                counts.get(
                    numeric_value,
                    0,
                )
                + 1
            )

        outside_display_range = sum(
            count
            for value, count in counts.items()
            if value < display_min
            or value > display_max
        )

        displayed_values = sorted(
            value
            for value in counts
            if display_min
            <= value
            <= display_max
        )

        xs = [
            value + offset
            for value in displayed_values
        ]

        probabilities = [
            counts[value] / total
            for value in displayed_values
        ]

        axis.vlines( # type: ignore
            xs,
            0,
            probabilities,
            color=color,
            linewidth=2.2,
        )

        axis.scatter( # type: ignore
            xs,
            probabilities,
            s=16,
            color=color,
            zorder=3,
            label=(
                f"{label} "
                f"({outside_display_range} rozdań "
                "poza zakresem osi, np. bonus "
                "Blind za pokera lub pokera "
                "królewskiego)"
            ),
        )

    axis.set_yscale( # type: ignore
        "log"
    )

    axis.set_xlabel( # type: ignore
        "Wynik netto pojedynczego "
        "rozdania [jednostki Ante]"
    )

    axis.set_ylabel( # type: ignore
        "Prawdopodobieństwo "
        "(skala logarytmiczna)"
    )

    axis.set_title( # type: ignore
        "Rozkład wyniku netto per rozdanie"
    )

    axis.grid( # type: ignore
        alpha=0.25
    )

    axis.legend( # type: ignore
        fontsize=7.5,
        framealpha=1.0,
    )

    figure.text( # type: ignore
        0.5,
        0.005,
        "DQN i REINFORCE: rozdania zebrane ze wszystkich pięciu "
        "seedów konfiguracji (500 tys. rozdań), nie z jednego "
        "wybranego przebiegu.",
        ha="center",
        fontsize=8,
    )

    figure.subplots_adjust(
        bottom=0.14
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="12_payout_distribution",
    )


def plot_bankroll_curve(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    random_target = find_target(
        results,
        "baseline_random",
    )

    rule_based_target = find_target(
        results,
        "baseline_rule_based",
    )

    (
        dqn_algorithm,
        dqn_state,
        dqn_reward,
    ) = BEST_VARIANTS[0]

    (
        reinforce_algorithm,
        reinforce_state,
        reinforce_reward,
    ) = BEST_VARIANTS[1]

    dqn_targets = find_model_targets(
        results,
        algorithm=dqn_algorithm,
        state_representation=dqn_state,
        reward_type=dqn_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    reinforce_targets = find_model_targets(
        results,
        algorithm=reinforce_algorithm,
        state_representation=(
            reinforce_state
        ),
        reward_type=(
            reinforce_reward
        ),
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    series = (
        (
            "Random",
            (
                random_target,
            ),
            "0.35",
        ),
        (
            "RuleBased",
            (
                rule_based_target,
            ),
            "0.1",
        ),
        (
            "DQN",
            dqn_targets,
            "C0",
        ),
        (
            "REINFORCE",
            reinforce_targets,
            "C1",
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            9.5,
            5.2,
        )
    )

    for label, targets, color in series:
        # Wszystkie targety danej serii grają na identycznym
        # harmonogramie talii, więc uśredniam wynik po seedach
        # w każdej pozycji rozdania, zamiast pokazywać jeden
        # wybrany przebieg.
        per_target_values = [
            [
                float(value)
                for value in load_round_net_profits(
                    target.rounds_path
                )
            ]
            for target in targets
        ]

        values = [
            fmean(
                round_values
            )
            for round_values in zip(
                *per_target_values,
                strict=True,
            )
        ]

        running_mean: list[
            float
        ] = []

        running_total = 0.0

        for index, value in enumerate(
            values,
            start=1,
        ):
            running_total += value

            running_mean.append(
                running_total
                / index
            )

        axis.plot( # type: ignore
            range(
                1,
                len(running_mean)
                + 1,
            ),
            running_mean,
            label=label,
            linewidth=1.6,
            color=color,
        )

    axis.set_xscale( # type: ignore
        "log"
    )

    axis.axhline( # type: ignore
        0.0,
        color="black",
        linewidth=0.8,
        linestyle=":",
    )

    axis.set_xlabel( # type: ignore
        "Liczba rozegranych rozdań "
        "(skala logarytmiczna)"
    )

    axis.set_ylabel( # type: ignore
        "Średni wynik narastająco "
        "[jednostki Ante]"
    )

    axis.set_title( # type: ignore
        "Zbieganie wyniku do EV na tym "
        "samym rozkładzie rozdań"
    )

    axis.grid( # type: ignore
        alpha=0.25,
        which="both",
    )

    axis.legend() # type: ignore

    figure.text( # type: ignore
        0.5,
        0.005,
        "DQN i REINFORCE: wynik uśredniony po pięciu seedach "
        "konfiguracji w każdej pozycji rozdania, nie z jednego "
        "wybranego przebiegu.",
        ha="center",
        fontsize=8,
    )

    figure.subplots_adjust(
        bottom=0.14
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="13_bankroll_curve",
    )


def plot_outcome_profile(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    random_target = find_target(
        results,
        "baseline_random",
    )

    rule_based_target = find_target(
        results,
        "baseline_rule_based",
    )

    (
        dqn_algorithm,
        dqn_state,
        dqn_reward,
    ) = BEST_VARIANTS[0]

    (
        reinforce_algorithm,
        reinforce_state,
        reinforce_reward,
    ) = BEST_VARIANTS[1]

    dqn_targets = find_model_targets(
        results,
        algorithm=dqn_algorithm,
        state_representation=dqn_state,
        reward_type=dqn_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    reinforce_targets = find_model_targets(
        results,
        algorithm=reinforce_algorithm,
        state_representation=(
            reinforce_state
        ),
        reward_type=reinforce_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    rows = (
        (
            "Random",
            (random_target,),
        ),
        (
            "RuleBased",
            (rule_based_target,),
        ),
        (
            "DQN",
            dqn_targets,
        ),
        (
            "REINFORCE",
            reinforce_targets,
        ),
    )

    outcome_labels = (
        (
            "PLAYER_WIN",
            "gracz wygrywa",
        ),
        (
            "DEALER_WIN",
            "krupier wygrywa",
        ),
        (
            "PUSH",
            "remis",
        ),
        (
            "PLAYER_FOLD",
            "fold",
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            4.5,
        )
    )

    bottoms = [
        0.0
        for _ in rows
    ]

    for outcome, outcome_label in outcome_labels:
        shares: list[
            float
        ] = []

        for _, targets in rows:
            counts = _aggregate_outcome_counts(
                targets
            )

            total = sum(
                counts.values()
            )

            shares.append(
                counts.get(
                    outcome,
                    0,
                )
                / total
                * 100.0
            )

        axis.barh( # type: ignore
            range(
                len(rows)
            ),
            shares,
            left=bottoms,
            label=outcome_label,
        )

        bottoms = [
            bottom + share
            for bottom, share in zip(
                bottoms,
                shares,
            )
        ]

    axis.set_yticks( # type: ignore
        range(
            len(rows)
        )
    )

    axis.set_yticklabels( # type: ignore
        [
            label
            for label, _ in rows
        ]
    )

    axis.invert_yaxis()

    axis.set_xlim(
        0,
        100,
    )

    axis.set_xlabel( # type: ignore
        "Udział wyniku rozdania [%]"
    )

    axis.set_title( # type: ignore
        "Profil wyników rozdania"
    )

    axis.legend( # type: ignore
        ncols=2,
        fontsize=8,
        framealpha=1.0,
    )

    axis.grid( # type: ignore
        axis="x",
        alpha=0.2,
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="14_outcome_profile",
    )


def plot_action_profiles(
    results: FinalResultsData,
    *,
    algorithm: str,
    state_representation: str,
    reward_type: str,
    output_dir: Path,
    filename: str,
) -> None:
    targets = sorted(
        find_model_targets(
            results,
            algorithm=algorithm,
            state_representation=(
                state_representation
            ),
            reward_type=reward_type,
            training_episodes=(
                EXTENDED_TRAINING_EPISODES
            ),
        ),
        key=lambda target: cast(
            ModelTargetMetadata,
            get_model_metadata(
                target
            ),
        ).training_seed,
    )

    actions = (
        "BET_4X",
        "BET_3X",
        "BET_2X",
        "BET_1X",
        "CHECK",
        "FOLD",
    )

    figure, axis = plt.subplots(
        figsize=(
            9.5,
            5,
        )
    )

    bottoms = [
        0.0
        for _ in targets
    ]

    for action in actions:
        shares: list[
            float
        ] = []

        for target in targets:
            counts = _get_action_counts(
                target
            )

            total = sum(
                counts.values()
            )

            shares.append(
                (
                    counts.get(
                        action,
                        0,
                    )
                    / total
                    * 100.0
                )
            )

        axis.barh( # type: ignore
            range(
                len(targets)
            ),
            shares,
            left=bottoms,
            label=action,
        )

        bottoms = [
            bottom + share
            for bottom, share
            in zip(
                bottoms,
                shares,
            )
        ]

    axis.set_yticks( # type: ignore
        range(
            len(targets)
        )
    )

    axis.set_yticklabels( # type: ignore
        [
            f"seed {index + 1}"
            for index in range(
                len(targets)
            )
        ]
    )

    axis.set_xlim(
        0,
        100,
    )

    axis.set_xlabel( # type: ignore
        "Udział akcji [%]"
    )

    axis.set_title( # type: ignore
        f"Profile akcji "
        f"{_algorithm_label(algorithm)} "
        "po 100 tys. epizodów"
    )

    axis.legend( # type: ignore
        ncols=3,
        fontsize=8,
        framealpha=1.0,
    )

    axis.grid( # type: ignore
        axis="x",
        alpha=0.2,
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename=filename,
    )


def plot_preflop_agreement(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    rule_based_target = find_target(
        results,
        "baseline_rule_based",
    )

    (
        dqn_algorithm,
        dqn_state,
        dqn_reward,
    ) = BEST_VARIANTS[0]

    (
        reinforce_algorithm,
        reinforce_state,
        reinforce_reward,
    ) = BEST_VARIANTS[1]

    dqn_targets = find_model_targets(
        results,
        algorithm=dqn_algorithm,
        state_representation=dqn_state,
        reward_type=dqn_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    reinforce_targets = find_model_targets(
        results,
        algorithm=reinforce_algorithm,
        state_representation=(
            reinforce_state
        ),
        reward_type=(
            reinforce_reward
        ),
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    rule_based_actions = load_round_actions(
        rule_based_target.rounds_path
    )

    comparisons = (
        (
            "DQN",
            dqn_targets,
        ),
        (
            "REINFORCE",
            reinforce_targets,
        ),
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            10,
            4.8,
        ),
    )

    for axis, (label, targets) in zip(
        axes,
        comparisons,
    ):
        matrix = [
            [
                0
                for _ in PREFLOP_ACTIONS
            ]
            for _ in PREFLOP_ACTIONS
        ]

        for target in targets:
            comparison_actions = load_round_actions(
                target.rounds_path
            )

            for rule_sequence, comparison_sequence in zip(
                rule_based_actions,
                comparison_actions,
                strict=True,
            ):
                row = PREFLOP_ACTIONS.index(
                    rule_sequence[0]
                )

                column = PREFLOP_ACTIONS.index(
                    comparison_sequence[0]
                )

                matrix[row][column] += 1

        total = sum(
            sum(row)
            for row in matrix
        )

        agreement = (
            sum(
                matrix[index][index]
                for index in range(
                    len(PREFLOP_ACTIONS)
                )
            )
            / total
            * 100.0
        )

        axis.imshow( # type: ignore
            matrix,
            cmap="Blues",
        )

        threshold = (
            max(
                max(row)
                for row in matrix
            )
            / 2
        )

        for row in range(
            len(PREFLOP_ACTIONS)
        ):
            for column in range(
                len(PREFLOP_ACTIONS)
            ):
                axis.text( # type: ignore
                    column,
                    row,
                    str(
                        matrix[row][
                            column
                        ]
                    ),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix[row][
                            column
                        ]
                        > threshold
                        else "black"
                    ),
                )

        axis.set_xticks( # type: ignore
            range(
                len(PREFLOP_ACTIONS)
            )
        )

        axis.set_xticklabels( # type: ignore
            PREFLOP_ACTIONS
        )

        axis.set_yticks( # type: ignore
            range(
                len(PREFLOP_ACTIONS)
            )
        )

        axis.set_yticklabels( # type: ignore
            PREFLOP_ACTIONS
        )

        axis.set_xlabel( # type: ignore
            f"{label}: wybrana akcja"
        )

        axis.set_ylabel( # type: ignore
            "RuleBased: wybrana akcja"
        )

        axis.set_title( # type: ignore
            f"{label} vs RuleBased "
            f"(zgodność {agreement:.1f}%)"
        )

    figure.suptitle( # type: ignore
        "Zgodność decyzji preflop "
        "z agentem regułowym"
    )

    figure.text( # type: ignore
        0.5,
        0.01,
        "Macierze zsumowane po pięciu seedach konfiguracji "
        "(500 tys. rozdań każda), nie z jednego wybranego "
        "przebiegu.",
        ha="center",
        fontsize=8,
    )

    figure.subplots_adjust(
        bottom=0.15
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename=(
            "16_preflop_agreement_"
            "with_rulebased"
        ),
    )


VARIANT_MARKERS = {
    (
        RAW,
        NET_PROFIT,
    ): "o",
    (
        RAW,
        SCALED_NET_PROFIT,
    ): "s",
    (
        FEATURES,
        NET_PROFIT,
    ): "^",
    (
        FEATURES,
        SCALED_NET_PROFIT,
    ): "D",
}


def plot_ev_per_risk(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    baseline_points: list[
        tuple[
            str,
            float,
            float,
        ]
    ] = []

    for identifier, label in (
        (
            "baseline_random",
            "Random",
        ),
        (
            "baseline_rule_based",
            "RuleBased",
        ),
    ):
        target = find_target(
            results,
            identifier,
        )

        evaluation = cast(
            dict[str, object],
            target.summary.get(
                "evaluation"
            ),
        )

        mean_staked = cast(
            float,
            evaluation.get(
                "mean_staked"
            ),
        )

        baseline_points.append(
            (
                label,
                mean_staked,
                get_evaluation_ev(
                    target
                ),
            )
        )

    variant_points: list[
        tuple[
            str,
            float,
            float,
            str,
            str,
        ]
    ] = []

    for algorithm, color in (
        (
            DQN,
            "C0",
        ),
        (
            REINFORCE,
            "C1",
        ),
    ):
        for state_representation, reward_type in (
            (
                RAW,
                NET_PROFIT,
            ),
            (
                RAW,
                SCALED_NET_PROFIT,
            ),
            (
                FEATURES,
                NET_PROFIT,
            ),
            (
                FEATURES,
                SCALED_NET_PROFIT,
            ),
        ):
            targets = find_model_targets(
                results,
                algorithm=algorithm,
                state_representation=(
                    state_representation
                ),
                reward_type=reward_type,
                training_episodes=(
                    FINAL_TRAINING_EPISODES
                ),
            )

            estimate = aggregate_model_ev(
                targets
            )

            mean_staked = fmean(
                cast(
                    float,
                    cast(
                        dict[str, object],
                        target.summary.get(
                            "evaluation"
                        ),
                    ).get(
                        "mean_staked"
                    ),
                )
                for target in targets
            )

            variant_points.append(
                (
                    _variant_label(
                        algorithm,
                        state_representation,
                        reward_type,
                    ),
                    mean_staked,
                    estimate.mean,
                    color,
                    VARIANT_MARKERS[
                        (
                            state_representation,
                            reward_type,
                        )
                    ],
                )
            )

    figure, axis = plt.subplots(
        figsize=(
            11,
            6.5,
        )
    )

    for label, mean_staked, ev in baseline_points:
        axis.scatter( # type: ignore
            mean_staked,
            ev,
            color="0.25",
            marker="s",
            s=50,
            zorder=3,
            label=label,
        )

    for label, mean_staked, ev, color, marker in variant_points:
        axis.scatter( # type: ignore
            mean_staked,
            ev,
            color=color,
            marker=marker,
            s=60,
            zorder=3,
            label=label,
        )

    x_max = (
        max(
            [
                mean_staked
                for _, mean_staked, _ in baseline_points
            ]
            + [
                mean_staked
                for _, mean_staked, _, _, _ in variant_points
            ]
        )
        * 1.1
    )

    axis.plot( # type: ignore
        (0, x_max),
        (0, 0),
        linestyle=":",
        color="black",
        linewidth=1.0,
        label="edge = 0%",
    )

    axis.plot( # type: ignore
        (0, x_max),
        (
            0,
            -UTH_HOUSE_EDGE
            * x_max,
        ),
        linestyle="--",
        color="0.5",
        linewidth=1.0,
        label=(
            f"edge = "
            f"-{UTH_HOUSE_EDGE * 100:.2f}% "
            "(typowa przewaga kasyna w UTH)"
        ),
    )

    axis.set_xlim(
        0,
        x_max,
    )

    axis.set_xlabel( # type: ignore
        "Średnia stawka na rozdanie "
        "[jednostki Ante]"
    )

    axis.set_ylabel( # type: ignore
        "EV [jednostki Ante / rozdanie]"
    )

    axis.set_title( # type: ignore
        "EV względem podjętego ryzyka"
    )

    axis.grid( # type: ignore
        alpha=0.25
    )

    axis.legend( # type: ignore
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        framealpha=1.0,
    )

    figure.subplots_adjust( # type: ignore
        right=0.68
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="17_ev_per_risk",
    )


def _rank_index(
    rank: Rank,
) -> int:
    return (
        Rank.ACE.value
        - rank.value
    )


def _canonical_hand_cell(
    high_rank: Rank,
    low_rank: Rank,
    *,
    suited: bool,
) -> tuple[
    int,
    int,
]:
    """
    Zwraca (wiersz, kolumna) na siatce 13x13 układów startowych.

    Konwencja jak w typowych "range chartach" pokerowych: przekątna to
    pary, górny trójkąt to układy suited, dolny trójkąt to układy offsuit.
    """

    high_index = _rank_index(
        high_rank
    )

    low_index = _rank_index(
        low_rank
    )

    if high_rank == low_rank:
        return high_index, high_index

    if suited:
        return high_index, low_index

    return low_index, high_index


def plot_preflop_range_chart(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    rule_based_target = find_target(
        results,
        "baseline_rule_based",
    )

    (
        dqn_algorithm,
        dqn_state,
        dqn_reward,
    ) = BEST_VARIANTS[0]

    (
        reinforce_algorithm,
        reinforce_state,
        reinforce_reward,
    ) = BEST_VARIANTS[1]

    dqn_targets = find_model_targets(
        results,
        algorithm=dqn_algorithm,
        state_representation=dqn_state,
        reward_type=dqn_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    reinforce_targets = find_model_targets(
        results,
        algorithm=reinforce_algorithm,
        state_representation=(
            reinforce_state
        ),
        reward_type=(
            reinforce_reward
        ),
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    deck_seeds = load_deck_seeds(
        rule_based_target.rounds_path
    )

    hole_cards = reconstruct_preflop_hole_cards(
        deck_seeds
    )

    series = (
        (
            "RuleBased",
            (
                rule_based_target,
            ),
        ),
        (
            "DQN",
            dqn_targets,
        ),
        (
            "REINFORCE",
            reinforce_targets,
        ),
    )

    colormap = ListedColormap(
        list(
            PREFLOP_ACTION_COLORS.values()
        )
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            5.6,
        ),
    )

    for axis, (label, targets) in zip(
        axes,
        series,
    ):
        cell_actions: dict[
            tuple[int, int],
            list[str],
        ] = {}

        for target in targets:
            actions = load_round_actions(
                target.rounds_path
            )

            for (
                player_card_a,
                player_card_b,
            ), action_sequence in zip(
                hole_cards,
                actions,
                strict=True,
            ):
                high_card, low_card = sorted(
                    (
                        player_card_a,
                        player_card_b,
                    ),
                    key=lambda card: card.rank.value,
                    reverse=True,
                )

                cell = _canonical_hand_cell(
                    high_card.rank,
                    low_card.rank,
                    suited=(
                        high_card.suit
                        == low_card.suit
                    ),
                )

                cell_actions.setdefault(
                    cell,
                    [],
                ).append(
                    action_sequence[0]
                )

        grid = [
            [
                0
                for _ in range(13)
            ]
            for _ in range(13)
        ]

        for (
            row,
            column,
        ), cell_action_list in cell_actions.items():
            dominant_action = Counter(
                cell_action_list
            ).most_common(
                1
            )[0][0]

            grid[row][column] = list(
                PREFLOP_ACTION_COLORS
            ).index(
                dominant_action
            )

        axis.imshow( # type: ignore
            grid,
            cmap=colormap,
            vmin=0,
            vmax=(
                len(
                    PREFLOP_ACTION_COLORS
                )
                - 1
            ),
        )

        axis.set_xticks( # type: ignore
            range(13)
        )

        axis.set_xticklabels( # type: ignore
            RANK_LABELS,
            fontsize=8,
        )

        axis.set_yticks( # type: ignore
            range(13)
        )

        axis.set_yticklabels( # type: ignore
            RANK_LABELS,
            fontsize=8,
        )

        axis.set_title( # type: ignore
            label
        )

    legend_handles = [
        Patch(
            color=color,
            label=action,
        )
        for action, color in PREFLOP_ACTION_COLORS.items()
    ]

    figure.legend( # type: ignore
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.04,
        ),
        ncols=3,
        fontsize=9,
    )

    figure.suptitle( # type: ignore
        "Decyzja preflop w zależności od układu "
        "startowego (dominująca akcja)"
    )

    figure.text( # type: ignore
        0.5,
        -0.12,
        "DQN i REINFORCE: decyzje zebrane ze wszystkich pięciu "
        "seedów konfiguracji, nie z jednego wybranego przebiegu.",
        ha="center",
        fontsize=8,
    )

    figure.subplots_adjust(
        bottom=0.32
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="18_preflop_range_chart",
    )


def _smooth_series(
    values: tuple[
        float,
        ...
    ],
    *,
    window: int,
) -> tuple[
    float,
    ...
]:
    if window <= 1:
        return values

    smoothed: list[
        float
    ] = []

    for index in range(
        len(values)
    ):
        start = max(
            0,
            index - window + 1,
        )

        smoothed.append(
            fmean(
                values[
                    start : index + 1
                ]
            )
        )

    return tuple(
        smoothed
    )


def plot_training_stability_by_reward_scale(
    training_runs: tuple[
        TrainingRun,
        ...
    ],
    *,
    output_dir: Path,
) -> None:
    # Poziom wygładzenia dopasowany do granulacji aktualizacji: DQN
    # optymalizuje co krok środowiska (bardzo szumiące, per-krokowe
    # normy gradientu), REINFORCE co batch 32 epizodów (już z natury
    # dużo gładszy sygnał, więc bez dodatkowego wygładzania).
    specs = (
        (
            DQN,
            500,
        ),
        (
            REINFORCE,
            1,
        ),
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            11,
            4.6,
        ),
    )

    for axis, (algorithm, smoothing_window) in zip(
        axes,
        specs,
    ):
        for reward_type, color in (
            (
                NET_PROFIT,
                "C0",
            ),
            (
                SCALED_NET_PROFIT,
                "C1",
            ),
        ):
            matching_runs = (
                _find_training_runs(
                    training_runs,
                    algorithm=algorithm,
                    state_representation=FEATURES,
                    reward_type=reward_type,
                )
            )

            aggregate = (
                aggregate_training_diagnostics(
                    matching_runs
                )
            )

            episodes = tuple(
                point.episode
                for point in aggregate
            )

            means = _smooth_series(
                tuple(
                    point.mean_gradient_norm
                    for point in aggregate
                ),
                window=smoothing_window,
            )

            axis.plot( # type: ignore
                episodes,
                means,
                label=(
                    "zysk netto"
                    if reward_type == NET_PROFIT
                    else "skalowana"
                ),
                linewidth=1.4,
                color=color,
            )

        axis.set_yscale( # type: ignore
            "log"
        )

        axis.set_xlabel( # type: ignore
            "Epizody"
        )

        axis.set_ylabel( # type: ignore
            "Norma gradientu (log)"
        )

        axis.set_title( # type: ignore
            _algorithm_label(
                algorithm
            )
        )

        axis.grid( # type: ignore
            alpha=0.25,
            which="both",
        )

        axis.legend( # type: ignore
            fontsize=8,
            framealpha=1.0,
        )

    figure.suptitle( # type: ignore
        "Norma gradientu w zależności "
        "od skali funkcji nagrody"
    )

    figure.text( # type: ignore
        0.5,
        0.01,
        "DQN: średnia z 5 seedów, wygładzona oknem "
        "500 epizodów. REINFORCE: średnia z 5 seedów, "
        "bez wygładzania (co batch 32 epizodów).",
        ha="center",
        fontsize=9,
    )

    figure.subplots_adjust(
        bottom=0.18
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="19_training_stability_by_reward_scale",
    )


def _reference_series(
    results: FinalResultsData,
) -> tuple[
    tuple[
        str,
        tuple[
            EvaluationTargetData,
            ...,
        ],
    ],
    ...,
]:
    """
    Zwraca cztery serie odniesienia: Random, RuleBased oraz najlepszy
    wariant DQN/REINFORCE, każdy jako pełny zestaw pięciu seedów.

    Baseline'y mają tylko jeden target (nie trenuje się ich), więc
    dostają jednoelementowy tuple, żeby wszystkie serie miały ten
    sam kształt danych dla wywołującego kodu.
    """

    (
        dqn_algorithm,
        dqn_state,
        dqn_reward,
    ) = BEST_VARIANTS[0]

    (
        reinforce_algorithm,
        reinforce_state,
        reinforce_reward,
    ) = BEST_VARIANTS[1]

    dqn_targets = find_model_targets(
        results,
        algorithm=dqn_algorithm,
        state_representation=dqn_state,
        reward_type=dqn_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    reinforce_targets = find_model_targets(
        results,
        algorithm=reinforce_algorithm,
        state_representation=(
            reinforce_state
        ),
        reward_type=(
            reinforce_reward
        ),
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    return (
        (
            "Random",
            (
                find_target(
                    results,
                    "baseline_random",
                ),
            ),
        ),
        (
            "RuleBased",
            (
                find_target(
                    results,
                    "baseline_rule_based",
                ),
            ),
        ),
        (
            "DQN",
            dqn_targets,
        ),
        (
            "REINFORCE",
            reinforce_targets,
        ),
    )


def plot_action_funnel_by_street(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    series = _reference_series(
        results
    )

    series_actions = tuple(
        (
            label,
            tuple(
                sequence
                for target in targets
                for sequence in load_round_actions(
                    target.rounds_path
                )
            ),
        )
        for label, targets in series
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            14,
            4.6,
        ),
    )

    for axis, (street_label, street_index, street_action_order) in zip(
        axes,
        STREETS,
    ):
        reach_fractions = [
            (
                sum(
                    1
                    for sequence in sequences
                    if len(sequence) > street_index
                )
                / len(sequences)
            )
            for _, sequences in series_actions
        ]

        left_edges = [
            0.0
            for _ in series_actions
        ]

        for action in street_action_order:
            fractions = []

            for _, sequences in series_actions:
                relevant = tuple(
                    sequence[street_index]
                    for sequence in sequences
                    if len(sequence) > street_index
                )

                fractions.append(
                    (
                        sum(
                            1
                            for chosen_action in relevant
                            if chosen_action == action
                        )
                        / len(relevant)
                    )
                    if relevant
                    else 0.0
                )

            axis.barh( # type: ignore
                range(
                    len(series_actions)
                ),
                fractions,
                left=left_edges,
                color=ACTION_COLORS[
                    action
                ],
                label=action,
            )

            left_edges = [
                left + fraction
                for left, fraction in zip(
                    left_edges,
                    fractions,
                )
            ]

        if street_index > 0:
            for row_index, reach_fraction in enumerate(
                reach_fractions
            ):
                axis.text( # type: ignore
                    1.03,
                    row_index,
                    f"{reach_fraction * 100:.0f}% "
                    "dotarło",
                    va="center",
                    fontsize=7,
                )

        axis.set_yticks( # type: ignore
            range(
                len(series_actions)
            )
        )

        axis.set_yticklabels( # type: ignore
            [
                label
                for label, _ in series_actions
            ]
        )

        axis.invert_yaxis()

        axis.set_xlim(
            0,
            1.28
            if street_index > 0
            else 1.0,
        )

        axis.set_title( # type: ignore
            street_label
        )

        axis.grid( # type: ignore
            axis="x",
            alpha=0.25,
        )

        axis.legend( # type: ignore
            fontsize=8,
            loc="lower left",
            framealpha=1.0,
        )

    figure.suptitle( # type: ignore
        "Rozkład akcji na kolejnych "
        "ulicach rozdania"
    )

    figure.text( # type: ignore
        0.5,
        0.01,
        "Udziały akcji liczone wyłącznie wśród rozdań, które "
        "faktycznie dotarły do danej ulicy (preflop = 100% "
        "rozdań z definicji). DQN i REINFORCE: rozdania "
        "zebrane ze wszystkich pięciu seedów konfiguracji.",
        ha="center",
        fontsize=9,
    )

    figure.subplots_adjust(
        bottom=0.16
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="20_action_funnel_by_street",
    )


def plot_postflop_action_by_hand_strength(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    series = _reference_series(
        results
    )

    colors = {
        "Random": "0.55",
        "RuleBased": "0.15",
        "DQN": "C0",
        "REINFORCE": "C1",
    }

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            12,
            5,
        ),
    )

    specs = (
        (
            axes[0],
            "Flop",
            1,
            "BET_2X",
        ),
        (
            axes[1],
            "River",
            2,
            "BET_1X",
        ),
    )

    for axis, street_label, street_index, aggressive_action in specs:
        for label, targets in series:
            # Każdy seed liczony osobno, żeby oprócz przeciętnej
            # agresji pokazać też jej rozrzut pomiędzy seedami
            # (zamiast wybierać jeden reprezentatywny przebieg).
            per_seed_fractions: list[
                dict[
                    HandRank,
                    float,
                ]
            ] = []

            for target in targets:
                deck_seeds = load_deck_seeds(
                    target.rounds_path
                )

                actions = load_round_actions(
                    target.rounds_path
                )

                hand_ranks = reconstruct_postflop_hand_ranks(
                    deck_seeds
                )

                aggression_by_rank: dict[
                    HandRank,
                    list[
                        bool
                    ],
                ] = {
                    rank: []
                    for rank in HAND_RANK_ORDER
                }

                for sequence, (
                    flop_rank,
                    river_rank,
                ) in zip(
                    actions,
                    hand_ranks,
                ):
                    if len(sequence) <= street_index:
                        continue

                    hand_rank = (
                        flop_rank
                        if street_index == 1
                        else river_rank
                    )

                    aggression_by_rank[
                        hand_rank
                    ].append(
                        sequence[street_index]
                        == aggressive_action
                    )

                per_seed_fractions.append(
                    {
                        rank: (
                            sum(decisions)
                            / len(decisions)
                        )
                        for rank, decisions in aggression_by_rank.items()
                        if decisions
                    }
                )

            x_positions = []
            y_means = []
            y_errors = []

            for index, rank in enumerate(
                HAND_RANK_ORDER
            ):
                values_for_rank = [
                    fractions[rank]
                    for fractions in per_seed_fractions
                    if rank in fractions
                ]

                if not values_for_rank:
                    continue

                x_positions.append(
                    index
                )

                y_means.append(
                    fmean(
                        values_for_rank
                    )
                )

                y_errors.append(
                    stdev(
                        values_for_rank
                    )
                    if len(values_for_rank) > 1
                    else 0.0
                )

            axis.errorbar( # type: ignore
                x_positions,
                y_means,
                yerr=y_errors,
                marker="o",
                markersize=5,
                linewidth=1.6,
                capsize=3,
                color=colors[label],
                label=label,
            )

        axis.set_xticks( # type: ignore
            range(
                len(HAND_RANK_ORDER)
            )
        )

        axis.set_xticklabels( # type: ignore
            [
                HAND_RANK_LABELS[rank]
                for rank in HAND_RANK_ORDER
            ],
            fontsize=8,
        )

        axis.set_ylim(
            0,
            1,
        )

        axis.set_xlabel( # type: ignore
            "Najlepszy układ w tym momencie rozdania"
        )

        axis.set_ylabel( # type: ignore
            f"Udział decyzji "
            f"{aggressive_action}"
        )

        axis.set_title( # type: ignore
            street_label
        )

        axis.grid( # type: ignore
            alpha=0.25
        )

        axis.legend( # type: ignore
            fontsize=8,
            framealpha=1.0,
        )

    figure.suptitle( # type: ignore
        "Agresja w zależności od siły układu "
        "po odkryciu kart wspólnych"
    )

    figure.text( # type: ignore
        0.5,
        0.01,
        "Liczone wyłącznie wśród rozdań, które dotarły do danej "
        "ulicy. Puste kategorie (np. poker królewski na flopie) "
        "są pominięte na wykresie. DQN i REINFORCE: średnia "
        "i odchylenie standardowe po pięciu seedach konfiguracji.",
        ha="center",
        fontsize=9,
    )

    figure.subplots_adjust(
        bottom=0.18
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="21_postflop_action_by_hand_strength",
    )


def plot_agent_profile_radar(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    series = _reference_series(
        results
    )

    axis_labels = (
        "EV",
        "Agresja\npreflop",
        "Dotarcie\ndo river",
        "Spasowanie\nna river",
        "Zmienność\nwyniku",
    )

    colors = {
        "Random": "0.55",
        "RuleBased": "0.15",
        "DQN": "C0",
        "REINFORCE": "C1",
    }

    raw_values: list[
        tuple[
            str,
            tuple[
                float,
                ...,
            ],
        ]
    ] = []

    for label, targets in series:
        # Behawioralne stawki i zmienność liczone na wszystkich
        # rozdaniach ze wszystkich seedów razem (poolowanie), a EV
        # tą samą metodą co reszta raportu (średnia po seedach
        # z przedziałem ufności), zamiast wybierać jeden seed.
        actions = tuple(
            sequence
            for target in targets
            for sequence in load_round_actions(
                target.rounds_path
            )
        )

        net_profits = [
            float(value)
            for target in targets
            for value in load_round_net_profits(
                target.rounds_path
            )
        ]

        ev = (
            aggregate_model_ev(
                targets
            ).mean
            if len(targets) > 1
            else get_evaluation_ev(
                targets[0]
            )
        )

        preflop_aggression = (
            sum(
                1
                for sequence in actions
                if sequence[0] != "CHECK"
            )
            / len(actions)
        )

        reach_river = (
            sum(
                1
                for sequence in actions
                if len(sequence) == 3
            )
            / len(actions)
        )

        river_sequences = tuple(
            sequence
            for sequence in actions
            if len(sequence) == 3
        )

        fold_at_river = (
            (
                sum(
                    1
                    for sequence in river_sequences
                    if sequence[2] == "FOLD"
                )
                / len(river_sequences)
            )
            if river_sequences
            else 0.0
        )

        raw_values.append(
            (
                label,
                (
                    ev,
                    preflop_aggression,
                    reach_river,
                    fold_at_river,
                    stdev(
                        net_profits
                    ),
                ),
            )
        )

    axis_count = len(
        axis_labels
    )

    columns = [
        [
            values[axis_index]
            for _, values in raw_values
        ]
        for axis_index in range(
            axis_count
        )
    ]

    normalized: list[
        tuple[
            str,
            list[
                float,
            ],
        ]
    ] = []

    for label, values in raw_values:
        normalized_values = []

        for axis_index, value in enumerate(
            values
        ):
            column_min = min(
                columns[axis_index]
            )

            column_max = max(
                columns[axis_index]
            )

            normalized_values.append(
                (
                    (value - column_min)
                    / (column_max - column_min)
                )
                if column_max > column_min
                else 0.5
            )

        normalized.append(
            (
                label,
                normalized_values,
            )
        )

    angles = [
        2 * pi * index / axis_count
        for index in range(
            axis_count
        )
    ]

    angles.append(
        angles[0]
    )

    figure, axis = plt.subplots(
        figsize=(
            7,
            7,
        ),
        subplot_kw={
            "polar": True,
        },
    )

    for label, normalized_values in normalized:
        closed_values = (
            normalized_values
            + [
                normalized_values[0]
            ]
        )

        axis.plot( # type: ignore
            angles,
            closed_values,
            linewidth=1.8,
            color=colors[label],
            label=label,
        )

        axis.fill( # type: ignore
            angles,
            closed_values,
            alpha=0.08,
            color=colors[label],
        )

    axis.set_xticks( # type: ignore
        angles[:-1]
    )

    axis.set_xticklabels( # type: ignore
        axis_labels,
        fontsize=9,
    )

    axis.tick_params( # type: ignore
        axis="x",
        pad=18,
    )

    axis.set_ylim( # type: ignore
        0.0,
        1.35,
    )

    axis.set_yticks( # type: ignore
        (
            0.0,
            0.5,
            1.0,
        )
    )

    axis.set_yticklabels( # type: ignore
        (
            "min",
            "",
            "max",
        ),
        fontsize=7,
    )

    axis.set_title( # type: ignore
        "Profil behawioralny czterech agentów"
    )

    axis.legend( # type: ignore
        loc="upper right",
        bbox_to_anchor=(
            1.35,
            1.1,
        ),
        fontsize=8,
        framealpha=1.0,
    )

    figure.text( # type: ignore
        0.5,
        0.02,
        "Każda oś znormalizowana min-max niezależnie, "
        "względem czterech agentów, nie w skali absolutnej. "
        "DQN i REINFORCE: dane ze wszystkich pięciu seedów "
        "konfiguracji.",
        ha="center",
        fontsize=8,
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="22_agent_profile_radar",
    )


def plot_ev_advantage_waterfall(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    rule_based_target = find_target(
        results,
        "baseline_rule_based",
    )

    baseline_profits = [
        float(value)
        for value in load_round_net_profits(
            rule_based_target.rounds_path
        )
    ]

    (
        dqn_algorithm,
        dqn_state,
        dqn_reward,
    ) = BEST_VARIANTS[0]

    (
        reinforce_algorithm,
        reinforce_state,
        reinforce_reward,
    ) = BEST_VARIANTS[1]

    dqn_targets = find_model_targets(
        results,
        algorithm=dqn_algorithm,
        state_representation=dqn_state,
        reward_type=dqn_reward,
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    reinforce_targets = find_model_targets(
        results,
        algorithm=reinforce_algorithm,
        state_representation=(
            reinforce_state
        ),
        reward_type=(
            reinforce_reward
        ),
        training_episodes=(
            FINAL_TRAINING_EPISODES
        ),
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            13,
            5.2,
        ),
    )

    specs = (
        (
            axes[0],
            "DQN vs RuleBased",
            dqn_targets,
        ),
        (
            axes[1],
            "REINFORCE vs RuleBased",
            reinforce_targets,
        ),
    )

    for axis, title, agent_targets in specs:
        # Kontrybucję każdej ścieżki liczę osobno dla każdego z pięciu
        # seedów, a dopiero potem uśredniam po seedach, zamiast liczyć
        # ją raz na jednym wybranym przebiegu. Suma uśrednionych
        # kontrybucji nadal jest dokładnie równa średniej różnicy EV.
        per_seed_contributions: list[
            list[
                float
            ]
        ] = []

        for agent_target in agent_targets:
            agent_actions = load_round_actions(
                agent_target.rounds_path
            )

            agent_profits = [
                float(value)
                for value in load_round_net_profits(
                    agent_target.rounds_path
                )
            ]

            differences = [
                agent_value - baseline_value
                for agent_value, baseline_value in zip(
                    agent_profits,
                    baseline_profits,
                )
            ]

            seed_contributions = []

            for path in DECISION_PATHS:
                indices = [
                    index
                    for index, sequence in enumerate(
                        agent_actions
                    )
                    if sequence == path
                ]

                weight = (
                    len(indices)
                    / len(agent_actions)
                )

                mean_difference_on_path = (
                    fmean(
                        differences[index]
                        for index in indices
                    )
                    if indices
                    else 0.0
                )

                seed_contributions.append(
                    weight
                    * mean_difference_on_path
                )

            per_seed_contributions.append(
                seed_contributions
            )

        path_count = len(
            DECISION_PATHS
        )

        mean_contributions = [
            fmean(
                seed_contributions[path_index]
                for seed_contributions in per_seed_contributions
            )
            for path_index in range(
                path_count
            )
        ]

        contribution_estimates = [
            summarize_values(
                tuple(
                    seed_contributions[path_index]
                    for seed_contributions in per_seed_contributions
                ),
                critical_value=(
                    STUDENT_T_95_DF4_CRITICAL_VALUE
                ),
            )
            for path_index in range(
                path_count
            )
        ]

        cumulative = 0.0

        for index, (contribution, estimate) in enumerate(
            zip(
                mean_contributions,
                contribution_estimates,
            )
        ):
            segment_end = (
                cumulative
                + contribution
            )

            axis.bar( # type: ignore
                index,
                contribution,
                bottom=cumulative,
                color=(
                    "#55A868"
                    if contribution >= 0
                    else "#C44E52"
                ),
                width=0.6,
            )

            axis.errorbar( # type: ignore
                index,
                segment_end,
                yerr=(
                    estimate.ci95_high
                    - estimate.mean
                ),
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.0,
            )

            cumulative = segment_end

        total_per_seed = tuple(
            sum(
                seed_contributions
            )
            for seed_contributions in per_seed_contributions
        )

        total_estimate = summarize_values(
            total_per_seed,
            critical_value=(
                STUDENT_T_95_DF4_CRITICAL_VALUE
            ),
        )

        axis.bar( # type: ignore
            path_count,
            cumulative,
            color="0.3",
            width=0.6,
        )

        axis.errorbar( # type: ignore
            path_count,
            cumulative,
            yerr=(
                total_estimate.ci95_high
                - total_estimate.mean
            ),
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1.0,
        )

        axis.axhline( # type: ignore
            0.0,
            color="black",
            linewidth=0.8,
        )

        axis.set_xticks( # type: ignore
            range(
                len(DECISION_PATHS)
                + 1
            )
        )

        axis.set_xticklabels( # type: ignore
            [
                DECISION_PATH_LABELS[path]
                for path in DECISION_PATHS
            ]
            + [
                "Razem"
            ],
            fontsize=7,
        )

        axis.set_ylabel( # type: ignore
            "Wkład do różnicy EV "
            "[jednostki Ante]"
        )

        axis.set_title( # type: ignore
            title
        )

        axis.grid( # type: ignore
            axis="y",
            alpha=0.25,
        )

    figure.suptitle( # type: ignore
        "Źródła różnicy EV względem RuleBased, "
        "rozłożone na ścieżki decyzji"
    )

    figure.text( # type: ignore
        0.5,
        0.01,
        "Dekompozycja dokładna, nie kontrfaktyczna: wkład "
        "ścieżki = jej częstość razy średnia różnica EV na "
        "rozdaniach, które nią poszły. Suma słupków = "
        "całkowita średnia różnica. Kontrybucja liczona osobno "
        "dla każdego z pięciu seedów, słupki i wąsy błędu to "
        "średnia i 95% przedział ufności po seedach.",
        ha="center",
        fontsize=8,
    )

    figure.subplots_adjust(
        bottom=0.2
    )

    _save_figure(
        figure,
        output_dir=output_dir,
        filename="23_ev_advantage_waterfall",
    )


def _write_csv(
    path: Path,
    *,
    fieldnames: tuple[
        str,
        ...
    ],
    rows: list[
        dict[str, object]
    ],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def write_tables(
    results: FinalResultsData,
    *,
    output_dir: Path,
) -> None:
    variant_rows: list[
        dict[str, object]
    ] = []

    for summary in build_variant_summaries(
        results
    ):
        estimate = summary.estimate

        variant_rows.append(
            {
                "algorithm": (
                    summary.algorithm
                ),
                "state_representation": (
                    summary.state_representation
                ),
                "reward_type": (
                    summary.reward_type
                ),
                "mean_ev": estimate.mean,
                "sd_between_seeds": (
                    estimate.standard_deviation
                ),
                "standard_error": (
                    estimate.standard_error
                ),
                "ci95_low": (
                    estimate.ci95_low
                ),
                "ci95_high": (
                    estimate.ci95_high
                ),
                "minimum_ev": (
                    summary.minimum_ev
                ),
                "maximum_ev": (
                    summary.maximum_ev
                ),
            }
        )

    _write_csv(
        output_dir
        / "variant_summary_50k.csv",
        fieldnames=(
            "algorithm",
            "state_representation",
            "reward_type",
            "mean_ev",
            "sd_between_seeds",
            "standard_error",
            "ci95_low",
            "ci95_high",
            "minimum_ev",
            "maximum_ev",
        ),
        rows=variant_rows,
    )

    state_rows: list[
        dict[str, object]
    ] = []

    for effect in build_state_effects(
        results
    ):
        estimate = effect.estimate

        state_rows.append(
            {
                "algorithm": (
                    effect.algorithm
                ),
                "reward_type": (
                    effect.context
                ),
                "comparison": (
                    "features_minus_raw"
                ),
                "mean_difference": (
                    estimate.mean
                ),
                "standard_deviation": (
                    estimate.standard_deviation
                ),
                "standard_error": (
                    estimate.standard_error
                ),
                "ci95_low": (
                    estimate.ci95_low
                ),
                "ci95_high": (
                    estimate.ci95_high
                ),
            }
        )

    _write_csv(
        output_dir
        / "state_representation_effect.csv",
        fieldnames=(
            "algorithm",
            "reward_type",
            "comparison",
            "mean_difference",
            "standard_deviation",
            "standard_error",
            "ci95_low",
            "ci95_high",
        ),
        rows=state_rows,
    )

    reward_rows: list[
        dict[str, object]
    ] = []

    for effect in build_reward_effects(
        results
    ):
        estimate = effect.estimate

        reward_rows.append(
            {
                "algorithm": (
                    effect.algorithm
                ),
                "state_representation": (
                    effect.context
                ),
                "comparison": (
                    "net_minus_scaled"
                ),
                "mean_difference": (
                    estimate.mean
                ),
                "standard_deviation": (
                    estimate.standard_deviation
                ),
                "standard_error": (
                    estimate.standard_error
                ),
                "ci95_low": (
                    estimate.ci95_low
                ),
                "ci95_high": (
                    estimate.ci95_high
                ),
            }
        )

    _write_csv(
        output_dir
        / "reward_effect.csv",
        fieldnames=(
            "algorithm",
            "state_representation",
            "comparison",
            "mean_difference",
            "standard_deviation",
            "standard_error",
            "ci95_low",
            "ci95_high",
        ),
        rows=reward_rows,
    )

    budget_rows: list[
        dict[str, object]
    ] = []

    for summary in (
        build_training_budget_summaries(
            results
        )
    ):
        budget_rows.append(
            {
                "algorithm": (
                    summary.algorithm
                ),
                "state_representation": (
                    summary.state_representation
                ),
                "reward_type": (
                    summary.reward_type
                ),
                "mean_ev_50k": (
                    summary.ev_50k.mean
                ),
                "mean_ev_100k": (
                    summary.ev_100k.mean
                ),
                "mean_improvement": (
                    summary.improvement.mean
                ),
                "improvement_ci95_low": (
                    summary
                    .improvement
                    .ci95_low
                ),
                "improvement_ci95_high": (
                    summary
                    .improvement
                    .ci95_high
                ),
            }
        )

    _write_csv(
        output_dir
        / "training_budget_summary.csv",
        fieldnames=(
            "algorithm",
            "state_representation",
            "reward_type",
            "mean_ev_50k",
            "mean_ev_100k",
            "mean_improvement",
            "improvement_ci95_low",
            "improvement_ci95_high",
        ),
        rows=budget_rows,
    )

    baseline_rows: list[
        dict[str, object]
    ] = []

    for identifier in (
        "baseline_random",
        "baseline_rule_based",
    ):
        target = find_target(
            results,
            identifier,
        )

        evaluation_raw = (
            target.summary.get(
                "evaluation"
            )
        )

        if not isinstance(
            evaluation_raw,
            dict,
        ):
            raise ValueError(
                "baseline evaluation missing"
            )

        evaluation = cast(
            dict[str, object],
            evaluation_raw,
        )

        baseline_rows.append(
            {
                "identifier": identifier,
                "estimated_ev": (
                    get_evaluation_ev(
                        target
                    )
                ),
                "standard_error": (
                    evaluation.get(
                        "standard_error"
                    )
                ),
                "mean_staked": (
                    evaluation.get(
                        "mean_staked"
                    )
                ),
            }
        )

    _write_csv(
        output_dir
        / "baseline_summary.csv",
        fieldnames=(
            "identifier",
            "estimated_ev",
            "standard_error",
            "mean_staked",
        ),
        rows=baseline_rows,
    )


def generate_report(
    *,
    output_dir: Path = (
        DEFAULT_OUTPUT_DIR
    ),
) -> None:
    _configure_plot_style()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_final_results()

    final_training_runs = (
        load_training_runs(
            results.final_training_manifest
        )
    )

    extended_training_runs = (
        load_training_runs(
            results.extended_training_manifest
        )
    )

    write_tables(
        results,
        output_dir=output_dir,
    )

    plot_final_ev_50k(
        results,
        output_dir=output_dir,
    )

    plot_state_effects(
        results,
        output_dir=output_dir,
    )

    plot_reward_effects(
        results,
        output_dir=output_dir,
    )

    plot_learning_curves(
        final_training_runs,
        algorithm=DQN,
        output_dir=output_dir,
        filename=(
            "04_dqn_learning_curves"
        ),
    )

    plot_learning_curves(
        final_training_runs,
        algorithm=REINFORCE,
        output_dir=output_dir,
        filename=(
            "05_reinforce_learning_curves"
        ),
    )

    plot_extended_learning_curves(
        extended_training_runs,
        output_dir=output_dir,
    )

    plot_extended_training_per_seed(
        extended_training_runs,
        output_dir=output_dir,
    )

    plot_seed_variability(
        extended_training_runs,
        output_dir=output_dir,
    )

    plot_action_profiles(
        results,
        algorithm=REINFORCE,
        state_representation=(
            BEST_VARIANTS[1][1]
        ),
        reward_type=(
            BEST_VARIANTS[1][2]
        ),
        output_dir=output_dir,
        filename="09_reinforce_action_profiles",
    )

    plot_action_profiles(
        results,
        algorithm=DQN,
        state_representation=(
            BEST_VARIANTS[0][1]
        ),
        reward_type=(
            BEST_VARIANTS[0][2]
        ),
        output_dir=output_dir,
        filename="15_dqn_action_profiles",
    )

    plot_training_budget_seed_pairs(
        results,
        output_dir=output_dir,
    )

    plot_headline_comparison(
        results,
        output_dir=output_dir,
    )

    plot_payout_distribution(
        results,
        output_dir=output_dir,
    )

    plot_bankroll_curve(
        results,
        output_dir=output_dir,
    )

    plot_outcome_profile(
        results,
        output_dir=output_dir,
    )

    plot_preflop_agreement(
        results,
        output_dir=output_dir,
    )

    plot_ev_per_risk(
        results,
        output_dir=output_dir,
    )

    plot_preflop_range_chart(
        results,
        output_dir=output_dir,
    )

    plot_training_stability_by_reward_scale(
        final_training_runs,
        output_dir=output_dir,
    )

    plot_action_funnel_by_street(
        results,
        output_dir=output_dir,
    )

    plot_postflop_action_by_hand_strength(
        results,
        output_dir=output_dir,
    )

    plot_agent_profile_radar(
        results,
        output_dir=output_dir,
    )

    plot_ev_advantage_waterfall(
        results,
        output_dir=output_dir,
    )

    print(
        "Final result report generated"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )


def main() -> None:
    generate_report()


if __name__ == "__main__":
    main()