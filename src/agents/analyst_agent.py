from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class AnalystAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Analyst",
            model_client=model_client,
            system_message=(
                """You are the Analyst agent.
                Analyze problems, evaluate trade-offs, complexity, performance, scalability, and risks.
                Verify logic, assumptions, and expected behavior.
                Do NOT write production code.
                Focus on correctness, feasibility, and implications."""
            ))
    
    async def run(self, content: str) -> str:
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="user")],
            cancellation)
        return response.chat_message.content
