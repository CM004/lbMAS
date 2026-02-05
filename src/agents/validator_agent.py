from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class ValidatorAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Validator",
            model_client=model_client,
            system_message=(
                "You are the Validator agent.\n"
                "Verify outputs meet all requirements, constraints, and acceptance criteria.\n"
                "Check: correctness, completeness, edge cases, consistency.\n"
                "Do NOT generate new content beyond validation feedback and pass/fail judgments."
            ))
    
    async def run(self, content: str) -> str:
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="user")],
            cancellation)
        return response.chat_message.content
