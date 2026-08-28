from random import Random

from expert_poker_player.evaluation import SimulationConfig


VALIDATION_SCHEDULE_SEED = 20260829
FINAL_EVALUATION_SCHEDULE_SEED = 20260830


def build_deck_schedule(
    *,
    source_seed: int,
    round_count: int,
) -> SimulationConfig:
    if type(source_seed) is not int:  # type: ignore
        raise TypeError(
            "source_seed must be an integer"
        )

    if type(round_count) is not int:  # type: ignore
        raise TypeError(
            "round_count must be an integer"
        )

    if round_count <= 0:
        raise ValueError(
            "round_count must be positive"
        )

    random = Random(source_seed)

    return SimulationConfig(
        deck_seeds=tuple(
            random.getrandbits(63)
            for _ in range(round_count)
        )
    )


def build_validation_schedule(
    round_count: int,
) -> SimulationConfig:
    return build_deck_schedule(
        source_seed=VALIDATION_SCHEDULE_SEED,
        round_count=round_count,
    )


def build_final_evaluation_schedule(
    round_count: int,
) -> SimulationConfig:
    return build_deck_schedule(
        source_seed=FINAL_EVALUATION_SCHEDULE_SEED,
        round_count=round_count,
    )