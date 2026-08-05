from typing import Protocol, runtime_checkable

from expert_poker_player.uth import Action, UTHObservation


@runtime_checkable
class Agent(Protocol):
    """Wspólny kontrakt agentów podejmujących decyzje w UTH."""

    def select_action(
        self,
        observation: UTHObservation,
    ) -> Action:
        """Wybiera akcję na podstawie obserwacji dostępnej agentowi."""

        ...  # pragma: no cover