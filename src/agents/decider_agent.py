"""
Decider Agent - Assesses if blackboard has enough info for final solution
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
import re

class DeciderAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Decider",
            model_client=model_client,
            system_message=("""You are the Decider agent in a blackboard-based multi-agent system.

Your role:
1. Assess whether the blackboard has enough information to produce a final answer
2. If ready, provide the final solution
3. If not ready, indicate what's missing

Output format:
- If solution ready: {"final_answer": "comprehensive final answer", "confidence": "high/medium"}
- If not ready: {"message": "continue, need more information about X"}

Be decisive - if the blackboard contains sufficient analysis and solutions, provide the final answer."""
            )
        )
    
    async def run(self, content: str) -> str:
        """Execute decider with blackboard context"""
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="blackboard")],
            cancellation
        )
        return response.chat_message.content
    
    def has_final_answer(self, response: str) -> bool:
        """Check if decider provided final answer"""
        return "final_answer" in response.lower() or re.search(r'"answer":\s*"', response, re.IGNORECASE)
