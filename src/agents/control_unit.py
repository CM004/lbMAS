"""
Control Unit - Dynamically selects agents based on blackboard state
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
import json
import re
from typing import List, Dict

class ControlUnit:
    """Control unit for dynamic agent selection"""
    
    def __init__(self, model_client):
        self.model_client = model_client
        self.agent = AssistantAgent(
            name="ControlUnit",
            model_client=model_client,
            system_message=("""You are the Control Unit in a blackboard-based multi-agent system.

Your role:
1. Analyze the current blackboard state
2. Select 1-3 most appropriate agents to act next
3. Consider: problem requirements, existing work, gaps in analysis

Available agent types and when to use them:
- Researcher: Need external knowledge, documentation, best practices
- Analyst: Need architecture design, requirement analysis
- Coder: Need implementation, code examples
- Planner: Need task decomposition, planning
- Critic: Review work, identify errors
- Optimizer: Improve performance, efficiency
- Validator: Final verification, testing
- Reporter: Compile comprehensive final documentation
- Decider: Assess if solution is complete
- Cleaner: Remove redundant messages
- ConflictResolver: Handle contradictions

Output ONLY JSON: {"chosen_agents": ["Agent1", "Agent2"], "reasoning": "why these agents"}

Select agents that will advance the solution based on current blackboard state."""
            )
        )
    
    async def select_agents(
        self,
        query: str,
        blackboard_content: str,
        available_agents: List[str]
    ) -> List[str]:
        """Select agents to execute based on blackboard state"""
        
        prompt = f"""Problem: {query}

Current Blackboard Contents (last 3000 chars):
{blackboard_content[-3000:] if len(blackboard_content) > 3000 else blackboard_content}

Available Agents: {', '.join(available_agents)}

Based on the current state, which 1-3 agents should act next?
Output JSON: {{"chosen_agents": ["Agent1", "Agent2"]}}"""
        
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=prompt, source="system")],
            cancellation
        )
        
        # Parse selection
        chosen = self._parse_selection(response.chat_message.content, available_agents)
        print(f"🎯 [CONTROL UNIT] Selected: {', '.join(chosen)}")
        return chosen
    
    def _parse_selection(self, response: str, available: List[str]) -> List[str]:
        """Parse agent selection from LLM response"""
        try:
            # Extract JSON
            json_match = re.search(r'\{[^{}]*"chosen_agents"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                chosen = data.get("chosen_agents", [])
                # Filter to only available agents
                return [a for a in chosen if a in available]
        except Exception as e:
            print(f"⚠️ [CONTROL UNIT] Parse error: {e}")
        
        # Fallback: select first 2 available agents
        return available[:2] if len(available) >= 2 else available
