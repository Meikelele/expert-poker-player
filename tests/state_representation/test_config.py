import pytest

from expert_poker_player.state_representation import (
    FEATURE_STATE_SIZE,
    RAW_STATE_SIZE,
    FeatureStateEncoder,
    RawStateEncoder,
    StateEncoder,
    StateRepresentation,
    build_state_encoder,
)


def test_raw_representation_has_stable_value() -> None:
    assert StateRepresentation.RAW.value == "raw"


def test_feature_representation_has_stable_value() -> None:
    assert StateRepresentation.FEATURES.value == "features"


def test_builds_raw_state_encoder() -> None:
    encoder = build_state_encoder(
        StateRepresentation.RAW
    )

    assert isinstance(
        encoder,
        RawStateEncoder,
    )

    assert isinstance(
        encoder,
        StateEncoder,
    )

    assert encoder.output_size == RAW_STATE_SIZE


def test_builds_feature_state_encoder() -> None:
    encoder = build_state_encoder(
        StateRepresentation.FEATURES
    )

    assert isinstance(
        encoder,
        FeatureStateEncoder,
    )

    assert isinstance(
        encoder,
        StateEncoder,
    )

    assert encoder.output_size == FEATURE_STATE_SIZE


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (
            "raw",
            StateRepresentation.RAW,
        ),
        (
            "features",
            StateRepresentation.FEATURES,
        ),
    ],
)
def test_representation_can_be_restored_from_string(
    raw_value: str,
    expected: StateRepresentation,
) -> None:
    assert StateRepresentation(
        raw_value
    ) is expected


def test_builder_rejects_invalid_representation() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "representation must be an instance "
            "of StateRepresentation"
        ),
    ):
        build_state_encoder(
            "raw"  # type: ignore[arg-type]
        )