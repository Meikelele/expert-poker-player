from enum import Enum


class Action(str, Enum):
    """Akcje dostępne dla gracza w Ultimate Texas Hold'em."""

    CHECK = "check"
    BET_4X = "bet_4x"
    BET_3X = "bet_3x"
    BET_2X = "bet_2x"
    BET_1X = "bet_1x"
    FOLD = "fold"


class GamePhase(str, Enum):
    """Fazy rozdania, w których może znajdować się silnik."""

    PREFLOP = "preflop"
    FLOP = "flop"
    RIVER = "river"
    TERMINAL = "terminal"


class RoundOutcome(str, Enum):
    """Końcowy wynik rozdania z perspektywy gracza."""

    PLAYER_WIN = "player_win"
    DEALER_WIN = "dealer_win"
    PUSH = "push"
    PLAYER_FOLD = "player_fold"


class WagerOutcome(str, Enum):
    """Sposób rozliczenia pojedynczego zakładu."""

    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    NOT_PLACED = "not_placed"