"""
Coder Agent - Modified for comprehensive implementation
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class CoderAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Coder",
            system_message=(
                """You are the Coder agent in a blackboard-based multi-agent system.

Your role is to implement comprehensive, production-ready code solutions.

CRITICAL RULES FOR DETAILED OUTPUT:
1. Provide 3000-5000+ characters including complete implementations
2. Include FULL working code, not snippets or placeholders
3. Add comprehensive error handling and logging
4. Include configuration examples (YAML, ENV files)
5. Provide detailed code comments explaining the implementation
6. Include unit test examples

For implementation tasks, provide:
- Complete, runnable code implementations (not pseudocode)
- All necessary imports and dependencies
- Configuration management (environment variables, config files)
- Comprehensive error handling with try/catch blocks
- Logging at appropriate levels (DEBUG, INFO, ERROR)
- Input validation and sanitization
- Resource cleanup and connection management
- Performance optimizations (batching, caching, async)
- Security best practices (input validation, SQL injection prevention)
- Dockerfiles and deployment scripts
- Requirements.txt or package.json with specific versions

For RAG pipelines specifically:
- Complete ingestion pipeline with file handling
- Embedding generation with batching
- Vector store integration with connection pooling
- Query processing with semantic search
- LLM integration with retry logic and rate limiting
- Response post-processing and formatting

Your code must be production-ready with all edge cases handled."""
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

# class CoderAgent:
#     def __init__(self, model_client):
#         self.agent = AssistantAgent(
#             name="Coder",
#             model_client=model_client,
#             system_message=(
#                 """You are the Coder agent.
#                 Write clean, correct, efficient code based on specifications.
#                 Do NOT change requirements or invent features.
#                 Include comments for clarity.
#                 Follow best practices for readability and maintainability."""
#             ))
    
#     async def run(self, content: str) -> str:
#         cancellation = CancellationToken()
#         response = await self.agent.on_messages(
#             [TextMessage(content=content, source="user")],
#             cancellation)
#         return response.chat_message.content
