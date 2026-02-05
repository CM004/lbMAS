"""
Analyst Agent - Modified for comprehensive analysis
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class AnalystAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Analyst",
            system_message=(
                """You are the Analyst agent in a blackboard-based multi-agent system.

Your role is to perform comprehensive analysis and design work.

CRITICAL RULES FOR DETAILED OUTPUT:
1. Provide 3000-5000+ characters of detailed analysis
2. Create comprehensive architecture diagrams (ASCII art or mermaid)
3. Define ALL functional and non-functional requirements in detail
4. Include specific metrics, targets, and acceptance criteria
5. Provide detailed data flow explanations
6. Include tables comparing different architectural approaches

For system design tasks, include:
- Complete architecture diagrams with all components
- Detailed component descriptions and responsibilities
- Data flow between components with specific formats
- API contracts and interfaces
- Performance requirements with specific numbers (latency, throughput)
- Scalability analysis with growth projections
- Security considerations and authentication flows
- Error handling strategies
- Monitoring and observability requirements

For requirements analysis, include:
- Functional requirements with specific use cases
- Non-functional requirements (performance, security, scalability)
- Constraints and assumptions
- Technology stack recommendations with justification
- Cost analysis and budget considerations

Your output must be comprehensive and production-ready."""
            ),
            model_client=model_client
        )

    async def run(self, content: str) -> str:
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="blackboard")],
            cancellation
        )
        return response.chat_message.content

# from autogen_agentchat.agents import AssistantAgent
# from autogen_agentchat.messages import TextMessage
# from autogen_core import CancellationToken

# class AnalystAgent:
#     def __init__(self, model_client):
#         self.agent = AssistantAgent(
#             name="Analyst",
#             model_client=model_client,
#             system_message=(
#                 """You are the Analyst agent.
#                 Analyze problems, evaluate trade-offs, complexity, performance, scalability, and risks.
#                 Verify logic, assumptions, and expected behavior.
#                 Do NOT write production code.
#                 Focus on correctness, feasibility, and implications."""
#             ))
    
#     async def run(self, content: str) -> str:
#         cancellation = CancellationToken()
#         response = await self.agent.on_messages(
#             [TextMessage(content=content, source="user")],
#             cancellation)
#         return response.chat_message.content
