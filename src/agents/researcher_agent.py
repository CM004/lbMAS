"""
Researcher Agent - Modified for comprehensive output
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class ResearcherAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Researcher",
            system_message=(
                """You are the Researcher agent in a blackboard-based multi-agent system.

Your role is to gather comprehensive, accurate, and up-to-date information needed to complete assigned tasks.

CRITICAL RULES FOR DETAILED OUTPUT:
1. Provide 3000-5000+ characters of comprehensive research
2. Include specific technical details, algorithms, libraries, APIs
3. Cover best practices, recent developments, and industry standards
4. Provide code examples, architecture patterns, and implementation strategies
5. Include specific metrics, benchmarks, and performance considerations

For RAG pipeline tasks, include:
- Specific embedding models (sentence-transformers, OpenAI, Cohere)
- Vector store options with pros/cons (FAISS, Milvus, Pinecone, Weaviate)
- LLM integration patterns and optimization techniques
- Chunking strategies with specific token ranges
- Retrieval algorithms and ranking methods
- Performance optimization techniques

Present findings with:
- Detailed explanations (not bullet points only)
- Specific technical specifications
- Code snippets where relevant
- Comparison tables for different approaches
- Assumptions and limitations noted

Your output should be substantial and production-ready research."""
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

# class ResearcherAgent:
#     def __init__(self, model_client):
#         self.agent = AssistantAgent(
#             name="researcher",
#             system_message=(
#                 """You are the Researcher agent.
#                 Your role is to gather accurate, relevant, and up-to-date information needed to complete assigned tasks.
#                 You focus on external knowledge, documentation, best practices, algorithms, libraries, APIs, and prior art.
#                 You do not write code or make final decisions.
#                 You present findings clearly, with assumptions and limitations noted."""),
#             model_client=model_client)

#     async def run(self, content):
#         cancellation = CancellationToken()
#         response = await self.agent.on_messages(
#             [TextMessage(content=content, source="agent")], cancellation)
#         return response.chat_message.content
