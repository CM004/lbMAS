"""
Cleaner Agent - Removes useless/redundant messages to reduce token consumption
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class CleanerAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Cleaner",
            model_client=model_client,
            system_message=("""You are the Cleaner agent in a blackboard-based multi-agent system.

Your role:
1. Identify useless or redundant messages on the blackboard
2. Mark messages for removal to reduce token consumption
3. Keep important context and conclusions

Output format:
- If cleanup needed: {"clean_list": [{"message_index": 3, "reason": "redundant info"}, ...]}
- If clean: {"message": "no cleanup needed"}

Be conservative - only remove truly redundant messages."""
            )
        )
    
    async def run(self, content: str) -> str:
        """Execute cleaner with blackboard context"""
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="blackboard")],
            cancellation
        )
        return response.chat_message.content
