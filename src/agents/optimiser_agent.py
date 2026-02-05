from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class OptimizerAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Optimizer",
            model_client=model_client,
            system_message=(
                "You are the Optimizer agent.\n"
                "Improve solutions by optimizing: performance, cost, scalability, memory, simplicity.\n"
                "Do NOT alter core requirements or functionality.\n"
                "Clearly explain trade-offs of proposed optimizations."
            ))
    
    async def run(self, content: str) -> str:
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="user")],
            cancellation
        )
        return response.chat_message.content
