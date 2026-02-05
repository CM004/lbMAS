from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class CriticAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Critic",
            model_client=model_client,
            system_message=(
                "You are the Critic agent.\n"
                "Review outputs and identify flaws, ambiguities, inefficiencies, or missing elements.\n"
                "Provide constructive, actionable feedback.\n"
                "Do NOT propose full solutions unless requested.\n"
                "Prioritize correctness, clarity, and robustness."
            ))
    
    async def run(self, content: str) -> str:
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="user")],
            cancellation)
        return response.chat_message.content
