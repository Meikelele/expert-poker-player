import math

import pytest

from expert_poker_player.evaluation import SimulationConfig
from expert_poker_player.policy_gradient import (
    PolicyGradientConfig,
    compute_probe_snapshots,
    generate_probe_states,
    normalized_entropy,
    run_untrained_control,
    train_policy_gradient,
)
from expert_poker_player.policy_gradient.agent import (
    PolicyGradientAgent,
)
from expert_poker_player.policy_gradient.network import (
    PolicyNetwork,
)
from expert_poker_player.rewards import NetProfitReward
from expert_poker_player.rl.actions import ACTION_ORDER
from expert_poker_player.state_representation import (
    RAW_STATE_SIZE,
    RawStateEncoder,
)
from expert_poker_player.uth import GamePhase


def test_normalized_entropy_is_approximately_one_for_uniform_distribution() -> (
    None
):
    probabilities = (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
        0.0,
        0.0,
        0.0,
    )

    assert normalized_entropy(
        probabilities,
        3,
    ) == pytest.approx(
        1.0,
        rel=1e-6,
    )


def test_normalized_entropy_approaches_zero_for_concentrated_distribution() -> (
    None
):
    probabilities = (
        0.999,
        0.0005,
        0.0005,
        0.0,
        0.0,
        0.0,
    )

    assert normalized_entropy(
        probabilities,
        3,
    ) < 0.05


def test_normalized_entropy_handles_single_legal_action_safely() -> None:
    assert normalized_entropy(
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        1,
    ) == 0.0


def test_generate_probe_states_covers_all_three_phases_per_hand() -> None:
    probes = generate_probe_states(
        count=5
    )

    assert len(probes) == 15

    for hand_index in range(5):
        preflop, flop, river = probes[
            3 * hand_index : 3 * hand_index + 3
        ]

        assert preflop.probe_index == hand_index
        assert flop.probe_index == hand_index
        assert river.probe_index == hand_index

        assert preflop.observation.phase is GamePhase.PREFLOP
        assert flop.observation.phase is GamePhase.FLOP
        assert river.observation.phase is GamePhase.RIVER

        assert len(preflop.observation.community_cards) == 0
        assert len(flop.observation.community_cards) == 3
        assert len(river.observation.community_cards) == 5


def test_generate_probe_states_is_deterministic_for_the_same_seed() -> None:
    first = generate_probe_states(
        count=10,
        seed=999,
    )

    second = generate_probe_states(
        count=10,
        seed=999,
    )

    assert first == second


def test_compute_probe_snapshots_zeroes_illegal_actions_and_sums_to_one() -> (
    None
):
    network = PolicyNetwork(
        input_size=RAW_STATE_SIZE,
        hidden_sizes=(4,),
    )

    agent = PolicyGradientAgent(
        policy_network=network,
        state_encoder=RawStateEncoder(),
        deterministic=True,
        seed=1,
    )

    probe_state = generate_probe_states(
        count=1
    )[0]

    snapshots = compute_probe_snapshots(
        agent=agent,
        probe_states=(probe_state,),
        update=0,
    )

    assert len(snapshots) == 1

    snapshot = snapshots[0]

    legal_actions = probe_state.observation.legal_actions

    for action, probability in zip(
        ACTION_ORDER,
        snapshot.probabilities,
    ):
        if action not in legal_actions:
            assert probability == 0.0

    assert sum(
        snapshot.probabilities
    ) == pytest.approx(
        1.0,
        rel=1e-5,
    )

    assert snapshot.max_probability == max(
        snapshot.probabilities
    )


def _build_probe_test_config(
    *,
    seed: int = 42,
) -> PolicyGradientConfig:
    return PolicyGradientConfig(
        learning_rate=1e-3,
        gamma=1.0,
        batch_size=2,
        hidden_sizes=(8,),
        training_episodes=4,
        seed=seed,
    )


def test_training_captures_probe_snapshots_only_at_requested_checkpoints() -> (
    None
):
    probe_states = generate_probe_states(
        count=2
    )

    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=_build_probe_test_config(),
        probe_states=probe_states,
        probe_checkpoints=(0, 1, 5),
    )

    assert result.optimizer_updates == 2

    captured_updates = {
        snapshot.update
        for snapshot in result.probe_snapshots
    }

    assert captured_updates == {0, 1}


def test_training_without_probe_states_produces_no_snapshots() -> None:
    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=_build_probe_test_config(),
    )

    assert result.probe_snapshots == ()


def test_cumulative_steps_is_non_decreasing_across_updates() -> None:
    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=_build_probe_test_config(),
    )

    cumulative_steps = [
        stats.cumulative_steps
        for stats in result.update_stats
    ]

    assert cumulative_steps == sorted(
        cumulative_steps
    )


def test_update_stats_gradient_norm_is_finite() -> None:
    result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=_build_probe_test_config(),
    )

    assert all(
        math.isfinite(
            stats.gradient_norm
        )
        and stats.gradient_norm >= 0.0
        for stats in result.update_stats
    )


def test_same_training_seed_produces_identical_probe_snapshots() -> None:
    probe_states = generate_probe_states(
        count=2
    )

    config = _build_probe_test_config()

    first = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
        probe_states=probe_states,
    )

    second = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
        probe_states=probe_states,
    )

    assert first.probe_snapshots == second.probe_snapshots


def test_untrained_control_snapshot_matches_training_update_zero() -> None:
    probe_states = generate_probe_states(
        count=2
    )

    config = _build_probe_test_config()

    evaluation_config = SimulationConfig(
        deck_seeds=(101, 202, 303)
    )

    control = run_untrained_control(
        state_encoder=RawStateEncoder(),
        config=config,
        evaluation_config=evaluation_config,
        probe_states=probe_states,
    )

    training_result = train_policy_gradient(
        state_encoder=RawStateEncoder(),
        reward_function=NetProfitReward(),
        config=config,
        probe_states=probe_states,
        probe_checkpoints=(0,),
    )

    update_zero_snapshots = tuple(
        snapshot
        for snapshot in training_result.probe_snapshots
        if snapshot.update == 0
    )

    assert control.probe_snapshots == update_zero_snapshots


def test_untrained_control_is_deterministic_for_the_same_seed() -> None:
    probe_states = generate_probe_states(
        count=2
    )

    config = _build_probe_test_config()

    evaluation_config = SimulationConfig(
        deck_seeds=(101, 202, 303)
    )

    first = run_untrained_control(
        state_encoder=RawStateEncoder(),
        config=config,
        evaluation_config=evaluation_config,
        probe_states=probe_states,
    )

    second = run_untrained_control(
        state_encoder=RawStateEncoder(),
        config=config,
        evaluation_config=evaluation_config,
        probe_states=probe_states,
    )

    assert dict(first.action_counts) == dict(
        second.action_counts
    )

    assert first.probe_snapshots == second.probe_snapshots


def test_untrained_control_differs_for_a_different_seed() -> None:
    probe_states = generate_probe_states(
        count=2
    )

    evaluation_config = SimulationConfig(
        deck_seeds=(101, 202, 303)
    )

    first = run_untrained_control(
        state_encoder=RawStateEncoder(),
        config=_build_probe_test_config(
            seed=1
        ),
        evaluation_config=evaluation_config,
        probe_states=probe_states,
    )

    second = run_untrained_control(
        state_encoder=RawStateEncoder(),
        config=_build_probe_test_config(
            seed=2
        ),
        evaluation_config=evaluation_config,
        probe_states=probe_states,
    )

    assert (
        first.probe_snapshots
        != second.probe_snapshots
    )


def test_untrained_control_is_stateless_across_repeated_calls() -> None:
    probe_states = generate_probe_states(
        count=1
    )

    config = _build_probe_test_config()

    evaluation_config = SimulationConfig(
        deck_seeds=(101,)
    )

    control = run_untrained_control(
        state_encoder=RawStateEncoder(),
        config=config,
        evaluation_config=evaluation_config,
        probe_states=probe_states,
    )

    repeated = run_untrained_control(
        state_encoder=RawStateEncoder(),
        config=config,
        evaluation_config=evaluation_config,
        probe_states=probe_states,
    )

    assert control.mean_max_probability == pytest.approx(
        repeated.mean_max_probability,
        rel=1e-6,
    )
