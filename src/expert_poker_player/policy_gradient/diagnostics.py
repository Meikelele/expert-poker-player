from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random
from statistics import mean
from types import MappingProxyType
from typing import Final

from expert_poker_player.cards import Deck
from expert_poker_player.evaluation import (
    SimulationConfig,
    calculate_metrics,
    run_simulation,
)
from expert_poker_player.policy_gradient.agent import (
    PolicyGradientAgent,
)
from expert_poker_player.policy_gradient.config import (
    PolicyGradientConfig,
)
from expert_poker_player.policy_gradient.seeding import (
    build_initial_policy_network,
)
from expert_poker_player.rl.actions import (
    ACTION_ORDER,
)
from expert_poker_player.state_representation import (
    StateEncoder,
)
from expert_poker_player.uth import (
    Action,
    GamePhase,
    UTHGame,
    UTHObservation,
)


DEFAULT_PROBE_COUNT: Final[int] = 100

# Ziarno wyłącznie dla generatora sond diagnostycznych, niezależne
# od ziarna treningu, żeby te same sondy dało się porównywać
# pomiędzy dowolnymi wariantami i seedami treningowymi.
DEFAULT_PROBE_SEED: Final[int] = 20260827

PROBE_CHECKPOINTS: Final[tuple[int, ...]] = (
    0,
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2000,
)


@dataclass(
    frozen=True,
    slots=True,
)
class ProbeState:
    """Bezpieczna, ustalona obserwacja używana do diagnostyki polityki."""

    probe_index: int
    observation: UTHObservation

    def __post_init__(self) -> None:
        if type(self.probe_index) is not int:
            raise TypeError(
                "probe_index must be an integer"
            )

        if self.probe_index < 0:
            raise ValueError(
                "probe_index must be non-negative"
            )

        if not isinstance(
            self.observation,
            UTHObservation,
        ):  # type: ignore
            raise TypeError(
                "observation must be an instance "
                "of UTHObservation"
            )

        if self.observation.terminated:
            raise ValueError(
                "probe observation cannot be terminal"
            )


@dataclass(
    frozen=True,
    slots=True,
)
class ProbeSnapshot:
    """Zrzut maskowanego rozkładu polityki dla jednej sondy."""

    update: int
    probe_index: int
    phase: GamePhase
    normalized_entropy: float
    max_probability: float
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.update) is not int or self.update < 0:
            raise ValueError(
                "update must be a non-negative integer"
            )

        if type(self.probe_index) is not int or self.probe_index < 0:
            raise ValueError(
                "probe_index must be a non-negative integer"
            )

        if not isinstance(self.phase, GamePhase):  # type: ignore
            raise TypeError(
                "phase must be an instance of GamePhase"
            )

        if not isinstance(self.probabilities, tuple):  # type: ignore
            raise TypeError(
                "probabilities must be a tuple"
            )

        if len(self.probabilities) != len(ACTION_ORDER):
            raise ValueError(
                "probabilities must contain exactly "
                "one value per global action"
            )


def generate_probe_states(
    *,
    count: int = DEFAULT_PROBE_COUNT,
    seed: int = DEFAULT_PROBE_SEED,
) -> tuple[ProbeState, ...]:
    """
    Buduje ustalony zestaw bezpiecznych obserwacji sond.

    Dla każdego z `count` rozdań rejestruje obserwację PREFLOP, a następnie
    dwukrotnie wykonuje akcję CHECK, aby dla tego samego rozdania zebrać
    również obserwacje FLOP i RIVER. Wykorzystuje wyłącznie publiczny
    interfejs `UTHGame`, więc karty krupiera i spalone karty nigdy nie są
    odczytywane. Zestaw jest niezależny od ziarna treningu, reprezentacji
    stanu i funkcji nagrody, dzięki czemu można go współdzielić między
    wszystkimi porównywanymi wariantami.
    """

    if type(count) is not int or count <= 0:
        raise ValueError(
            "count must be a positive integer"
        )

    if type(seed) is not int:
        raise TypeError(
            "seed must be an integer"
        )

    deck_seed_rng = random.Random(
        seed
    )

    deck_seeds = tuple(
        deck_seed_rng.getrandbits(63)
        for _ in range(count)
    )

    probes: list[ProbeState] = []

    for probe_index, deck_seed in enumerate(
        deck_seeds
    ):
        game = UTHGame()

        preflop_observation = game.reset(
            card_source=Deck(
                seed=deck_seed
            )
        )

        probes.append(
            ProbeState(
                probe_index=probe_index,
                observation=preflop_observation,
            )
        )

        flop_result = game.step(
            Action.CHECK
        )

        probes.append(
            ProbeState(
                probe_index=probe_index,
                observation=flop_result.observation,
            )
        )

        river_result = game.step(
            Action.CHECK
        )

        probes.append(
            ProbeState(
                probe_index=probe_index,
                observation=river_result.observation,
            )
        )

    return tuple(
        probes
    )


def normalized_entropy(
    probabilities: Sequence[float],
    legal_action_count: int,
) -> float:
    """
    Liczy entropię znormalizowaną przez log liczby legalnych akcji.

    Wynik bliski 1.0 oznacza rozkład bliski jednorodnemu, a bliski 0.0
    oznacza rozkład praktycznie deterministyczny. Dla jednej legalnej
    akcji entropia jest z definicji zerowa, więc unikamy dzielenia przez
    log(1) = 0.
    """

    if legal_action_count <= 1:
        return 0.0

    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )

    return entropy / math.log(
        legal_action_count
    )


def compute_probe_snapshots(
    *,
    agent: PolicyGradientAgent,
    probe_states: tuple[ProbeState, ...],
    update: int,
) -> tuple[ProbeSnapshot, ...]:
    """Liczy zrzuty polityki dla ustalonego zestawu sond."""

    if not isinstance(
        agent,
        PolicyGradientAgent,
    ):  # type: ignore
        raise TypeError(
            "agent must be an instance "
            "of PolicyGradientAgent"
        )

    if not isinstance(
        probe_states,
        tuple,
    ):  # type: ignore
        raise TypeError(
            "probe_states must be a tuple"
        )

    if type(update) is not int or update < 0:
        raise ValueError(
            "update must be a non-negative integer"
        )

    snapshots: list[ProbeSnapshot] = []

    for probe_state in probe_states:
        observation = probe_state.observation

        probabilities = agent.action_probabilities(
            observation
        )

        legal_probabilities = tuple(
            probability
            for action, probability in zip(
                ACTION_ORDER,
                probabilities,
            )
            if action in observation.legal_actions
        )

        snapshots.append(
            ProbeSnapshot(
                update=update,
                probe_index=probe_state.probe_index,
                phase=observation.phase,
                normalized_entropy=normalized_entropy(
                    probabilities,
                    len(observation.legal_actions),
                ),
                max_probability=max(
                    legal_probabilities
                ),
                probabilities=probabilities,
            )
        )

    return tuple(
        snapshots
    )


@dataclass(
    frozen=True,
    slots=True,
)
class UntrainedControlResult:
    """Wynik kontroli nieuczonej, losowo zainicjalizowanej sieci."""

    action_counts: Mapping[Action, int]
    probe_snapshots: tuple[ProbeSnapshot, ...]
    mean_max_probability: float
    mean_preflop_probabilities: tuple[float, ...]


def run_untrained_control(
    *,
    state_encoder: StateEncoder,
    config: PolicyGradientConfig,
    evaluation_config: SimulationConfig,
    probe_states: tuple[ProbeState, ...],
) -> UntrainedControlResult:
    """
    Ocenia świeżo zainicjalizowaną sieć bez żadnego treningu.

    Używa tej samej procedury seedowania co `train_policy_gradient`,
    dzięki czemu wynik jest bezpośrednio porównywalny z zachowaniem
    sieci na starcie prawdziwego treningu z tym samym ziarnem.
    """

    if not isinstance(
        evaluation_config,
        SimulationConfig,
    ):  # type: ignore
        raise TypeError(
            "evaluation_config must be an instance "
            "of SimulationConfig"
        )

    if not isinstance(
        probe_states,
        tuple,
    ):  # type: ignore
        raise TypeError(
            "probe_states must be a tuple"
        )

    policy_network, seeds = build_initial_policy_network(
        state_encoder=state_encoder,
        config=config,
    )

    policy_network.eval()

    agent = PolicyGradientAgent(
        policy_network=policy_network,
        state_encoder=state_encoder,
        deterministic=True,
        seed=seeds.agent_seed,
    )

    simulation_result = run_simulation(
        agent=agent,
        config=evaluation_config,
    )

    metrics = calculate_metrics(
        simulation_result
    )

    snapshots = compute_probe_snapshots(
        agent=agent,
        probe_states=probe_states,
        update=0,
    )

    mean_max_probability = mean(
        snapshot.max_probability
        for snapshot in snapshots
    )

    preflop_snapshots = tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.phase is GamePhase.PREFLOP
    )

    mean_preflop_probabilities = tuple(
        mean(
            snapshot.probabilities[index]
            for snapshot in preflop_snapshots
        )
        for index in range(
            len(ACTION_ORDER)
        )
    )

    return UntrainedControlResult(
        action_counts=MappingProxyType(
            dict(
                metrics.action_counts
            )
        ),
        probe_snapshots=snapshots,
        mean_max_probability=mean_max_probability,
        mean_preflop_probabilities=mean_preflop_probabilities,
    )
