"""
Critic Agent - Modified for Blackboard Architecture
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class CriticAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Critic",
            model_client=model_client,
            system_message=("""You are the Critic agent in a blackboard-based multi-agent system.

Your role:
1. Review messages on the blackboard for errors, inconsistencies, or misleading information
2. Identify potential hallucinations or incorrect reasoning
3. Force relevant agents to rethink their output

Output format:
- If errors found: {"critic_list": [{"wrong_message": "...", "explanation": "why wrong", "affected_agent": "..."}]}
- If no errors: {"message": "no problems detected, waiting for more information"}

You communicate solely through the blackboard."""
            )
        )
    
    async def run(self, content: str) -> str:
        """Execute critic with blackboard context"""
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="blackboard")],
            cancellation
        )
        return response.chat_message.content

# from autogen_agentchat.agents import AssistantAgent
# from autogen_agentchat.messages import TextMessage
# from autogen_core import CancellationToken

# class CriticAgent:
#     def __init__(self, model_client):
#         self.agent = AssistantAgent(
#             name="Critic",
#             model_client=model_client,
#             system_message=(
#                 "You are the Critic agent.\n"
#                 "Review outputs and identify flaws, ambiguities, inefficiencies, or missing elements.\n"
#                 "Provide constructive, actionable feedback.\n"
#                 "Do NOT propose full solutions unless requested.\n"
#                 "Prioritize correctness, clarity, and robustness."
#             ))
    
#     async def run(self, content: str) -> str:
#         cancellation = CancellationToken()
#         response = await self.agent.on_messages(
#             [TextMessage(content=content, source="user")],
#             cancellation)
#         return response.chat_message.content
