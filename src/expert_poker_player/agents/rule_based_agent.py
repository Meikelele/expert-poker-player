from expert_poker_player.agents.rule_based_strategy import (
    should_raise_flop,
    should_raise_preflop,
    should_raise_river,
)
from expert_poker_player.uth import (
    Action,
    GamePhase,
    UTHObservation,
)


class RuleBasedAgent:
    """Deterministyczny agent wykorzystujący reguły strategii UTH."""

    def select_action(
        self,
        observation: UTHObservation,
    ) -> Action:
        """Wybiera akcję zgodnie z regułami dla aktualnej fazy."""

        if not isinstance(observation, UTHObservation): # type: ignore
            raise TypeError(
                "observation must be an instance "
                "of UTHObservation"
            )

        if observation.terminated:
            raise ValueError(
                "cannot select an action for "
                "a terminal observation"
            )

        if observation.phase is GamePhase.PREFLOP:
            return (
                Action.BET_4X
                if should_raise_preflop(observation)
                else Action.CHECK
            )

        if observation.phase is GamePhase.FLOP:
            return (
                Action.BET_2X
                if should_raise_flop(observation)
                else Action.CHECK
            )

        return (
            Action.BET_1X
            if should_raise_river(observation)
            else Action.FOLD
        )