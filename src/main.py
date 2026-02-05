"""
Main entry point - Blackboard-Based Multi-Agent System with Groq
"""
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()

# Import agents
from agents.planner_agent import PlannerAgent
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import AnalystAgent
from agents.coder_agent import CoderAgent
from agents.critic_agent import CriticAgent
from agents.optimiser_agent import OptimizerAgent
from agents.validator_agent import ValidatorAgent
from agents.reporter_agent import ReporterAgent

# Import orchestrator
from agents.orchestrator import BlackboardOrchestrator


def save_output_to_file(content: str, user_goal: str, stats: dict):
    """Save detailed output to file outside src directory"""
    # Get parent directory (outside src)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(parent_dir, "output.txt")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    output_content = f"""{'='*70}
LBMAS - BLACKBOARD-BASED MULTI-AGENT SYSTEM OUTPUT
{'='*70}

Timestamp: {timestamp}
Goal: {user_goal}

{'='*70}
BLACKBOARD STATISTICS
{'='*70}
Total Rounds: {stats.get('current_round', 0)}
Total Messages: {stats.get('total_messages', 0)}
Agents Participated: {stats.get('agents_participated', 0)}

{'='*70}
FINAL COMPREHENSIVE RESULT
{'='*70}

{content}

{'='*70}
END OF OUTPUT
{'='*70}
"""
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"\n✅ Output saved to: {output_file}")
    return output_file


async def main():
    # Get Groq API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("❌ Error: Please set GROQ_API_KEY environment variable")
        print("   export GROQ_API_KEY='gsk_your-groq-key-here'")
        return
    
    # Initialize Groq model client (OpenAI-compatible)
    model_client = OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        model_capabilities={
            "json_output": True,
            "vision": False,
            "function_calling": True,
        }
    )
    
    print(f"✅ Using Groq API with model: llama-3.3-70b-versatile\n")
    
    # Initialize agents with detailed prompts
    planner = PlannerAgent(model_client)
    
    agents = {
        "Researcher": ResearcherAgent(model_client),
        "Analyst": AnalystAgent(model_client),
        "Coder": CoderAgent(model_client),
        "Critic": CriticAgent(model_client),
        "Optimizer": OptimizerAgent(model_client),
        "Validator": ValidatorAgent(model_client),
        "Reporter": ReporterAgent(model_client)  # Your original detailed reporter
    }
    
    # Initialize Blackboard Orchestrator
    orchestrator = BlackboardOrchestrator(
        planner_agent=planner,
        agents_dict=agents,
        model_client=model_client,
        max_rounds=4  # Can increase if needed
    )
    
    # Example query
    user_goal = """Design a production-ready RAG pipeline that can handle 
    10,000 technical documents with sub-second query latency."""
    
    print("\n" + "="*70)
    print("🎯 LBMAS - Blackboard-Based Multi-Agent System")
    print("="*70 + "\n")
    
    # Execute with blackboard mode
    result = await orchestrator.execute(user_goal, use_blackboard=True)
    
    # Get blackboard statistics
    stats = orchestrator.blackboard.get_stats()
    
    print("\n" + "="*70)
    print("✅ FINAL RESULT")
    print("="*70)
    print(f"\n{result}\n")
    
    # Save to file
    output_file = save_output_to_file(result, user_goal, stats)
    
    print(f"\n📊 Character count: {len(result)}")
    print(f"📄 Full output saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

# import asyncio
# import os
# from dotenv import load_dotenv
# from autogen_ext.models.openai import OpenAIChatCompletionClient

# from agents.planner_agent import PlannerAgent
# from agents.researcher_agent import ResearcherAgent
# from agents.analyst_agent import AnalystAgent
# from agents.coder_agent import CoderAgent
# from agents.critic_agent import CriticAgent
# from agents.optimiser_agent import OptimizerAgent
# from agents.validator_agent import ValidatorAgent
# from agents.reporter_agent import ReporterAgent
# from agents.orchestrator import MemoryEnabledOrchestrator

# from memory.agent_memory import AgentMemorySystem

# load_dotenv()

# class TeeOutput:
#     def __init__(self, file_path):
#         self.terminal = os.sys.stdout
#         self.file = open(file_path, 'w', encoding='utf-8')
    
#     def write(self, message):
#         self.terminal.write(message)
#         self.file.write(message)
#         self.file.flush()
    
#     def flush(self):
#         self.terminal.flush()
#         self.file.flush()
    
#     def isatty(self):
#         return self.terminal.isatty()
    
#     def close(self):
#         self.file.close()

# async def main():
#     tee = TeeOutput("output.md")
#     os.sys.stdout = tee
    
#     try:
#         print("="*70)
#         print("NEXUS AI - Multi-Agent System")
#         print("="*70)
#         print()
        
#         key = os.getenv("GROQ_API_KEY")
        
#         model_info = {
#             "family": "oss",
#             "vision": False,
#             "function_calling": True,
#             "json_output": True,
#             "structured_output": True,
#             "context_length": 131072,
#         }
        
#         model_client = OpenAIChatCompletionClient(
#             model="llama-3.3-70b-versatile",
#             api_key=key,
#             base_url="https://api.groq.com/openai/v1",
#             model_info=model_info,
#         )
        
#         planner = PlannerAgent(model_client)
#         researcher = ResearcherAgent(model_client)
#         analyst = AnalystAgent(model_client)
#         coder = CoderAgent(model_client)
#         critic = CriticAgent(model_client)
#         optimizer = OptimizerAgent(model_client)
#         validator = ValidatorAgent(model_client)
#         reporter = ReporterAgent(model_client)
        
#         agents = {
#             "Researcher": researcher,
#             "Analyst": analyst,
#             "Coder": coder,
#             "Critic": critic,
#             "Optimizer": optimizer,
#             "Validator": validator,
#             "Reporter": reporter,
#         }
        
#         memory_system = AgentMemorySystem(
#             session_max_turns=50,
#             vector_k=5,
#             vector_threshold=0.3,
#             db_path="vectorstore/agent_long_term.db",
#             vector_persist_path="vectorstore/agent_vectors.faiss"
#         )
        
#         orchestrator = MemoryEnabledOrchestrator(planner, agents, memory_system)
        
#         task = "Design a RAG pipeline for 10k documents"
        
#         print()
#         print("="*70)
#         print(f"TASK: {task}")
#         print("="*70)
#         print()
        
#         result = await orchestrator.execute(task, use_memory=True)
        
#         print()
#         print("="*70)
#         print("FINAL OUTPUT")
#         print("="*70)
#         print(result)
#         print("="*70)
#         print()
        
#         stats = await orchestrator.get_memory_stats()
#         print(f"Memory Stats: {stats}")
#         print()
        
#     except Exception as e:
#         print(f"\nERROR: {e}\n")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         os.sys.stdout = os.sys.__stdout__
#         tee.close()
#         print("\nOutput saved to output.md")

# if __name__ == "__main__":
#     asyncio.run(main())
