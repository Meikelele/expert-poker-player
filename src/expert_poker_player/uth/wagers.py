from dataclasses import dataclass
from fractions import Fraction

from expert_poker_player.uth.enums import WagerOutcome


@dataclass(frozen=True, slots=True)
class WagerSettlement:
    """
    Rozliczenie pojedynczego zakładu w jednostkach Ante.

    Przykłady:
    - stake=1, net_profit=1     oznacza wygraną 1:1,
    - stake=1, net_profit=0     oznacza push,
    - stake=1, net_profit=-1    oznacza przegraną,
    - stake=0, net_profit=0     oznacza brak zakładu.
    """

    stake: Fraction
    net_profit: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.stake, Fraction): # type: ignore
            raise TypeError("stake must be an instance of Fraction")

        if not isinstance(self.net_profit, Fraction): # type: ignore
            raise TypeError("net_profit must be an instance of Fraction")

        if self.stake < 0:
            raise ValueError("stake cannot be negative")

        if self.stake == 0 and self.net_profit != 0:
            raise ValueError(
                "a wager with zero stake must have zero net profit"
            )

        if self.net_profit < 0 and self.net_profit != -self.stake:
            raise ValueError(
                "a losing wager must lose exactly its full stake"
            )

    @property
    def outcome(self) -> WagerOutcome:
        """Określa wynik zakładu na podstawie stawki i zysku netto."""

        if self.stake == 0:
            return WagerOutcome.NOT_PLACED

        if self.net_profit > 0:
            return WagerOutcome.WIN

        if self.net_profit == 0:
            return WagerOutcome.PUSH

        return WagerOutcome.LOSS

    @property
    def gross_return(self) -> Fraction:
        """
        Zwraca całkowitą kwotę oddaną graczowi.

        Gross return obejmuje zwrot początkowej stawki.
        """

        return self.stake + self.net_profit


@dataclass(frozen=True, slots=True)
class Settlement:
    """Pełne rozliczenie zakładów Ante, Blind i Play."""

    ante: WagerSettlement
    blind: WagerSettlement
    play: WagerSettlement

    def __post_init__(self) -> None:
        for field_name in ("ante", "blind", "play"):
            value = getattr(self, field_name)

            if not isinstance(value, WagerSettlement):
                raise TypeError(
                    f"{field_name} must be an instance of WagerSettlement"
                )

    @property
    def total_staked(self) -> Fraction:
        """Łączna liczba postawionych jednostek Ante."""

        return (
            self.ante.stake
            + self.blind.stake
            + self.play.stake
        )

    @property
    def total_net_profit(self) -> Fraction:
        """Łączny zysk lub strata netto gracza."""

        return (
            self.ante.net_profit
            + self.blind.net_profit
            + self.play.net_profit
        )

    @property
    def total_gross_return(self) -> Fraction:
        """Łączna kwota zwrócona graczowi razem ze stawkami."""

        return (
            self.ante.gross_return
            + self.blind.gross_return
            + self.play.gross_return
        )