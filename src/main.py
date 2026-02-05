import asyncio
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.planner_agent import PlannerAgent
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import AnalystAgent
from agents.coder_agent import CoderAgent
from agents.critic_agent import CriticAgent
from agents.optimiser_agent import OptimizerAgent
from agents.validator_agent import ValidatorAgent
from agents.reporter_agent import ReporterAgent
from agents.orchestrator import MemoryEnabledOrchestrator

from memory.agent_memory import AgentMemorySystem

load_dotenv()

class TeeOutput:
    def __init__(self, file_path):
        self.terminal = os.sys.stdout
        self.file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def isatty(self):
        return self.terminal.isatty()
    
    def close(self):
        self.file.close()

async def main():
    tee = TeeOutput("output.md")
    os.sys.stdout = tee
    
    try:
        print("="*70)
        print("NEXUS AI - Multi-Agent System")
        print("="*70)
        print()
        
        key = os.getenv("GROQ_API_KEY")
        
        model_info = {
            "family": "oss",
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "context_length": 131072,
        }
        
        model_client = OpenAIChatCompletionClient(
            model="llama-3.3-70b-versatile",
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
            model_info=model_info,
        )
        
        planner = PlannerAgent(model_client)
        researcher = ResearcherAgent(model_client)
        analyst = AnalystAgent(model_client)
        coder = CoderAgent(model_client)
        critic = CriticAgent(model_client)
        optimizer = OptimizerAgent(model_client)
        validator = ValidatorAgent(model_client)
        reporter = ReporterAgent(model_client)
        
        agents = {
            "Researcher": researcher,
            "Analyst": analyst,
            "Coder": coder,
            "Critic": critic,
            "Optimizer": optimizer,
            "Validator": validator,
            "Reporter": reporter,
        }
        
        memory_system = AgentMemorySystem(
            session_max_turns=50,
            vector_k=5,
            vector_threshold=0.3,
            db_path="vectorstore/agent_long_term.db",
            vector_persist_path="vectorstore/agent_vectors.faiss"
        )
        
        orchestrator = MemoryEnabledOrchestrator(planner, agents, memory_system)
        
        task = "Design a RAG pipeline for 10k documents"
        
        print()
        print("="*70)
        print(f"TASK: {task}")
        print("="*70)
        print()
        
        result = await orchestrator.execute(task, use_memory=True)
        
        print()
        print("="*70)
        print("FINAL OUTPUT")
        print("="*70)
        print(result)
        print("="*70)
        print()
        
        stats = await orchestrator.get_memory_stats()
        print(f"Memory Stats: {stats}")
        print()
        
    except Exception as e:
        print(f"\nERROR: {e}\n")
        import traceback
        traceback.print_exc()
    
    finally:
        os.sys.stdout = os.sys.__stdout__
        tee.close()
        print("\nOutput saved to output.md")

if __name__ == "__main__":
    asyncio.run(main())
