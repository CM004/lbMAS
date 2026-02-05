"""
Conflict Resolver Agent - Detects contradictions and initiates private debates
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class ConflictResolverAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="ConflictResolver",
            model_client=model_client,
            system_message=("""You are the Conflict Resolver agent in a blackboard-based multi-agent system.

Your role:
1. Detect contradictions among messages on the blackboard
2. Identify agents with conflicting viewpoints
3. Initiate private space debates to resolve conflicts

Output format:
- If conflicts found: {"conflict_list": [{"agents": ["Agent1", "Agent2"], "issue": "description", "requires_debate": true}]}
- If no conflicts: {"message": "no conflicts detected"}

Focus on substantive conflicts, not minor differences."""
            )
        )
    
    async def run(self, content: str) -> str:
        """Execute conflict resolver with blackboard context"""
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="blackboard")],
            cancellation
        )
        return response.chat_message.content
