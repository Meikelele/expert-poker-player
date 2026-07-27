class UTHError(Exception):
    """Bazowy wyjątek dla błędów domenowych silnika UTH."""


class IllegalActionError(UTHError):
    """Akcja nie jest legalna w aktualnej fazie rozdania."""


class RoundNotStartedError(UTHError):
    """Operacja wymaga wcześniejszego rozpoczęcia rozdania."""


class RoundFinishedError(UTHError):
    """Próba wykonania akcji po zakończeniu rozdania."""

class InvalidPhaseTransitionError(UTHError):
    """Operacja rozdawania nie pasuje do aktualnej fazy."""