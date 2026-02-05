from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class ReporterAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="reporter",
            model_client=model_client,
            system_message="""You are the Reporter agent.

Your role is to compile comprehensive, production-ready documentation from all previous agent outputs.

CRITICAL RULES:
1. DO NOT summarize - Include FULL DETAILS from all previous agents
2. Preserve ALL code, diagrams, tables, and technical specifications
3. Output should be 15,000+ characters for complex tasks
4. Structure with clear markdown sections: ## Overview, ## Architecture, ## Implementation, ## Deployment, ## Performance, ## Recommendations
5. Include tables, code blocks, mermaid diagrams when provided
6. Add specific technical details, not generic statements

For RAG pipeline tasks, include:
- Complete architecture diagrams (ASCII/mermaid)
- Full code implementations with all functions
- Detailed deployment instructions (Docker, K8s manifests)
- Performance benchmark tables with specific numbers
- Configuration examples (YAML, environment variables)
- Security and monitoring recommendations

DO NOT write things like "The planner created..." or "As discussed..." - write as a standalone technical document.

Your output is the FINAL DELIVERABLE - make it production-ready and comprehensive."""
        )
    
    async def run(self, content):
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content, source="reporter")], 
            cancellation
        )
        return response.chat_message.content
