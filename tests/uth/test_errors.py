from expert_poker_player.uth import (
    IllegalActionError,
    RoundFinishedError,
    RoundNotStartedError,
    UTHError,
)


def test_domain_errors_inherit_from_uth_error() -> None:
    assert issubclass(IllegalActionError, UTHError)
    assert issubclass(RoundNotStartedError, UTHError)
    assert issubclass(RoundFinishedError, UTHError)


def test_domain_errors_preserve_message() -> None:
    error = IllegalActionError("BET_2X is illegal during preflop")

    assert str(error) == "BET_2X is illegal during preflop"