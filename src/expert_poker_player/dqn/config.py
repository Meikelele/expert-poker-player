from dataclasses import dataclass
import math


@dataclass(
    frozen=True,
    slots=True,
)
class DQNConfig:
    """Konfiguracja treningu agenta DQN."""

    learning_rate: float = 1e-3
    gamma: float = 0.99

    batch_size: int = 64
    replay_capacity: int = 10_000
    warmup_steps: int = 1_000
    target_sync_interval: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000

    hidden_sizes: tuple[int, ...] = (
        256,
        256,
    )

    training_episodes: int = 10_000
    seed: int = 42

    def __post_init__(self) -> None:
        self._validate_positive_float(
            self.learning_rate,
            "learning_rate",
        )

        if not isinstance(
            self.gamma,
            (int, float),
        ): # type: ignore
            raise TypeError(
                "gamma must be a number"
            )

        gamma = float(
            self.gamma
        )

        if (
            not math.isfinite(gamma)
            or not 0.0 <= gamma <= 1.0
        ):
            raise ValueError(
                "gamma must be between 0 and 1"
            )

        self._validate_positive_int(
            self.batch_size,
            "batch_size",
        )

        self._validate_positive_int(
            self.replay_capacity,
            "replay_capacity",
        )

        if (
            self.replay_capacity
            < self.batch_size
        ):
            raise ValueError(
                "replay_capacity must be at least "
                "batch_size"
            )

        if type(self.warmup_steps) is not int:
            raise TypeError(
                "warmup_steps must be an integer"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps cannot be negative"
            )

        self._validate_positive_int(
            self.target_sync_interval,
            "target_sync_interval",
        )

        self._validate_epsilon(
            self.epsilon_start,
            "epsilon_start",
        )

        self._validate_epsilon(
            self.epsilon_end,
            "epsilon_end",
        )

        if (
            self.epsilon_end
            > self.epsilon_start
        ):
            raise ValueError(
                "epsilon_end cannot exceed "
                "epsilon_start"
            )

        self._validate_positive_int(
            self.epsilon_decay_steps,
            "epsilon_decay_steps",
        )

        if not isinstance(
            self.hidden_sizes,
            tuple,
        ): # type: ignore
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

    def epsilon_at_step(
        self,
        step: int,
    ) -> float:
        """Zwraca epsilon dla podanego kroku treningowego."""

        if type(step) is not int:
            raise TypeError(
                "step must be an integer"
            )

        if step < 0:
            raise ValueError(
                "step cannot be negative"
            )

        progress = min(
            step / self.epsilon_decay_steps,
            1.0,
        )

        return (
            self.epsilon_start
            + (
                self.epsilon_end
                - self.epsilon_start
            )
            * progress
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Zwraca konfigurację w formie gotowej do serializacji."""

        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "warmup_steps": self.warmup_steps,
            "target_sync_interval": (
                self.target_sync_interval
            ),
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": (
                self.epsilon_decay_steps
            ),
            "hidden_sizes": list(
                self.hidden_sizes
            ),
            "training_episodes": (
                self.training_episodes
            ),
            "seed": self.seed,
        }

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

        numeric_value = float(
            value
        )

        if (
            not math.isfinite(numeric_value)
            or numeric_value <= 0.0
        ):
            raise ValueError(
                f"{name} must be positive and finite"
            )

    @staticmethod
    def _validate_epsilon(
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

        numeric_value = float(
            value
        )

        if (
            not math.isfinite(numeric_value)
            or not 0.0 <= numeric_value <= 1.0
        ):
            raise ValueError(
                f"{name} must be between 0 and 1"
            )