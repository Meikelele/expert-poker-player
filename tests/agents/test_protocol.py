from expert_poker_player.agents import Agent
from expert_poker_player.uth import Action, UTHObservation


class CompatibleAgent:
    def select_action(
        self,
        observation: UTHObservation,
    ) -> Action:
        return Action.CHECK


class IncompatibleAgent:
    pass

def test_compatible_agent_satisfies_agent_protocol() -> None:
    agent = CompatibleAgent()

    assert isinstance(agent, Agent)

def test_object_without_select_action_does_not_satisfy_protocol() -> None:
    agent = IncompatibleAgent()

    assert not isinstance(agent, Agent)