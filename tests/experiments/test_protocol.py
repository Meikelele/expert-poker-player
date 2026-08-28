from itertools import product

from expert_poker_player.experiments import (
    FINAL_TRAINING_SEEDS,
    FINAL_VARIANTS,
    ExperimentVariant,
    RLAlgorithm,
)
from expert_poker_player.rewards import RewardType
from expert_poker_player.state_representation import (
    StateRepresentation,
)


def test_final_variants_cover_complete_experiment_matrix() -> None:
    actual = {
        (
            variant.algorithm,
            variant.state_representation,
            variant.reward_type,
        )
        for variant in FINAL_VARIANTS
    }

    expected = set(
        product(
            RLAlgorithm,
            StateRepresentation,
            RewardType,
        )
    )

    assert actual == expected
    assert len(FINAL_VARIANTS) == 8


def test_final_variant_names_are_unique() -> None:
    names = tuple(
        variant.name
        for variant in FINAL_VARIANTS
    )

    assert len(set(names)) == 8


def test_final_training_uses_five_unique_seeds() -> None:
    assert len(FINAL_TRAINING_SEEDS) == 5
    assert len(set(FINAL_TRAINING_SEEDS)) == 5


def test_experiment_variant_has_stable_name() -> None:
    variant = ExperimentVariant(
        algorithm=RLAlgorithm.REINFORCE,
        state_representation=StateRepresentation.FEATURES,
        reward_type=RewardType.NET_PROFIT,
    )

    assert (
        variant.name
        == "reinforce_features_net_profit"
    )