from dataclasses import dataclass
import math


@dataclass(
    frozen=True,
    slots=True,
)
class PolicyGradientConfig:
    """Konfiguracja treningu agenta Policy Gradient."""

    learning_rate: float = 1e-3
    gamma: float = 0.99

    hidden_sizes: tuple[int, ...] = (
        256,
        256,
    )

    training_episodes: int = 10_000
    seed: int = 42

    def __post_init__(
        self,
    ) -> None:
        self._validate_positive_float(
            self.learning_rate,
            "learning_rate",
        )

        if not isinstance(
            self.gamma,
            (int, float),
        ):  # type: ignore
            raise TypeError(
                "gamma must be a number"
            )

        gamma = float(
            self.gamma
        )

        if (
            not math.isfinite(
                gamma
            )
            or not 0.0 <= gamma <= 1.0
        ):
            raise ValueError(
                "gamma must be between 0 and 1"
            )

        if not isinstance(
            self.hidden_sizes,
            tuple,
        ):  # type: ignore
            raise TypeError(
                "hidden_sizes must be a tuple"
            )

        if not self.hidden_sizes:
            raise ValueError(
                "hidden_sizes cannot be empty"
            )

        if not all(
            type(size) is int
            for size in self.hidden_sizes
        ):
            raise TypeError(
                "hidden_sizes must contain integers"
            )

        if not all(
            size > 0
            for size in self.hidden_sizes
        ):
            raise ValueError(
                "hidden_sizes must contain "
                "positive values"
            )

        self._validate_positive_int(
            self.training_episodes,
            "training_episodes",
        )

        if type(self.seed) is not int:
            raise TypeError(
                "seed must be an integer"
            )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Zwraca konfigurację gotową do serializacji."""

        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "hidden_sizes": list(
                self.hidden_sizes
            ),
            "training_episodes": (
                self.training_episodes
            ),
            "seed": self.seed,
        }

    @staticmethod
    def _validate_positive_float(
        value: object,
        name: str,
    ) -> None:
        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a number"
            )

        normalized = float(
            value
        )

        if (
            not math.isfinite(
                normalized
            )
            or normalized <= 0.0
        ):
            raise ValueError(
                f"{name} must be positive and finite"
            )

    @staticmethod
    def _validate_positive_int(
        value: object,
        name: str,
    ) -> None:
        if type(value) is not int:
            raise TypeError(
                f"{name} must be an integer"
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be positive"
            )