"""
Blackboard-Based Multi-Agent Orchestrator
Replaces memory-based orchestration with blackboard architecture
"""
import json
import re
from typing import List, Optional, Dict, Any
# from memory.agent_memory import AgentMemorySystem  # COMMENTED OUT - Using Blackboard instead
from memory.blackboard import Blackboard
from agents.control_unit import ControlUnit
from agents.decider_agent import DeciderAgent
from agents.cleaner_agent import CleanerAgent
from agents.conflict_resolver_agent import ConflictResolverAgent

class BlackboardOrchestrator:
    """
    LbMAS (Blackboard-based LLM Multi-Agent System) Orchestrator
    
    Key differences from memory-based system:
    1. All communication through blackboard (no agent memory)
    2. Dynamic agent selection via control unit
    3. Iterative rounds until consensus
    4. Token-efficient via cleaner agent
    """
    
    def __init__(self, planner_agent, agents_dict, model_client, max_rounds: int = 4):
        self.planner = planner_agent
        self.agents = agents_dict
        self.model_client = model_client
        self.max_rounds = max_rounds
        
        # NEW: Blackboard system (replaces memory)
        self.blackboard = Blackboard()
        
        # NEW: Control unit for dynamic agent selection
        self.control_unit = ControlUnit(model_client)
        
        # NEW: Functional agents for blackboard system
        self.decider = DeciderAgent(model_client)
        self.cleaner = CleanerAgent(model_client)
        self.conflict_resolver = ConflictResolverAgent(model_client)
        
        # Add functional agents to agent dict
        self.agents["Decider"] = self.decider
        self.agents["Cleaner"] = self.cleaner
        self.agents["ConflictResolver"] = self.conflict_resolver
        
        # COMMENTED OUT: Memory system no longer used
        # self.memory = memory_system
        
    async def execute(self, user_goal: str, use_blackboard: bool = True) -> str:
        """
        Execute using blackboard-based multi-agent system
        
        Args:
            user_goal: The problem to solve
            use_blackboard: If True, use blackboard architecture; if False, use legacy mode
        """
        print(f"\n{'='*70}")
        print(f"🚀 BLACKBOARD MULTI-AGENT SYSTEM (LbMAS)")
        print(f"{'='*70}")
        print(f"\n📝 Goal: {user_goal}\n")
        
        if not use_blackboard:
            # Legacy mode - use original implementation
            return await self._execute_legacy(user_goal)
        
        # NEW: Blackboard-based execution
        return await self._execute_blackboard(user_goal)
    
    async def _execute_blackboard(self, user_goal: str) -> str:
        """Execute using blackboard architecture"""
        
        # Phase 1: Initial Planning
        print("📋 Phase 1: Initial Planning\n")
        plan_response = await self.planner.run(user_goal)
        self.blackboard.write_public("Planner", plan_response)
        
        # Phase 2: Blackboard Cycle
        print(f"\n⚡ Phase 2: Blackboard Cycle (Max {self.max_rounds} rounds)\n")
        
        for round_num in range(self.max_rounds):
            self.blackboard.increment_round()
            
            # Get current blackboard content
            blackboard_content = self.blackboard.get_public_content()
            
            # Control unit selects agents
            available_agent_names = list(self.agents.keys())
            selected_names = await self.control_unit.select_agents(
                query=user_goal,
                blackboard_content=blackboard_content,
                available_agents=available_agent_names
            )
            
            if not selected_names:
                print("⚠️ No agents selected, ending cycle")
                break
            
            # Execute selected agents
            for agent_name in selected_names:
                if agent_name not in self.agents:
                    continue
                    
                print(f"\n🔧 Executing: {agent_name}")
                agent = self.agents[agent_name]
                
                # Build context for agent
                context = self._build_agent_context(
                    user_goal,
                    blackboard_content,
                    agent_name
                )
                
                # Execute agent
                try:
                    result = await agent.run(context)
                    self.blackboard.write_public(agent_name, result)
                    
                    # Check if Decider provided final answer
                    if agent_name == "Decider" and self.decider.has_final_answer(result):
                        print("\n✅ Decider confirmed solution is ready!")
                        return self._extract_final_answer(result)
                        
                except Exception as e:
                    print(f"❌ Error executing {agent_name}: {e}")
            
            # Periodic cleanup
            if round_num > 0 and round_num % 2 == 0:
                await self._run_cleaner()
        
        # Phase 3: Final Solution
        print(f"\n{'='*70}")
        print("📊 Phase 3: Extracting Final Solution")
        print(f"{'='*70}\n")
        
        return await self._finalize_solution(user_goal)
    
    def _build_agent_context(self, user_goal: str, blackboard_content: str, agent_name: str) -> str:
        """Build context for agent from blackboard"""
        context = f"""=== ORIGINAL GOAL ===
{user_goal}

=== YOUR TASK (as {agent_name}) ===
Review the blackboard and contribute based on your role.

{blackboard_content}

Provide your analysis and output."""
        return context
    
    async def _run_cleaner(self):
        """Run cleaner to remove redundant messages"""
        try:
            blackboard_content = self.blackboard.get_public_content()
            context = f"""Review these messages and identify any that are redundant or can be removed:

{blackboard_content}

Output indices of messages to remove (if any)."""
            
            result = await self.cleaner.run(context)
            # Parse and remove messages (simplified)
            if "clean_list" in result.lower():
                print("🧹 Cleaner removed redundant messages")
        except:
            pass
    
    def _extract_final_answer(self, decider_response: str) -> str:
        """Extract final answer from decider response"""
        try:
            # Try JSON parsing
            json_match = re.search(r'\{[^{}]*"final_answer"[^{}]*\}', decider_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("final_answer", decider_response)
        except:
            pass
        return decider_response
    
    async def _finalize_solution(self, user_goal: str) -> str:
        """Finalize solution from blackboard contents"""
        # Get ALL blackboard content for comprehensive compilation
        blackboard_content = self.blackboard.get_public_content(max_chars=50000)  # Increase limit
        
        # ALWAYS use Reporter for final comprehensive compilation
        if "Reporter" in self.agents:
            print("📝 Asking Reporter to compile COMPREHENSIVE final solution...")
            context = f"""Based on ALL the work on the blackboard, compile the COMPREHENSIVE final solution.

CRITICAL: Your output must be 15,000+ characters and include:
- Complete architecture diagrams (mermaid/ASCII)
- Full code implementations with ALL functions
- Detailed deployment instructions (Docker, K8s)
- Performance benchmark tables
- Configuration examples
- Security and monitoring recommendations

Original Goal: {user_goal}

ALL BLACKBOARD CONTENT:
{blackboard_content}

Compile EVERYTHING into a production-ready comprehensive document."""
            
            final_result = await self.agents["Reporter"].run(context)
        else:
            final_result = blackboard_content
        
        # Print statistics
        stats = self.blackboard.get_stats()
        print(f"\n📈 Blackboard Statistics:")
        print(f"   - Total Rounds: {stats['current_round']}")
        print(f"   - Messages: {stats['total_messages']}")
        print(f"   - Agents Participated: {stats['agents_participated']}")
        
        return self._extract_final_answer(final_result)
        
        # Print statistics
        stats = self.blackboard.get_stats()
        print(f"\n📈 Blackboard Statistics:")
        print(f"   - Total Rounds: {stats['current_round']}")
        print(f"   - Messages: {stats['total_messages']}")
        print(f"   - Agents Participated: {stats['agents_participated']}")
        
        return self._extract_final_answer(final_result)
    
    # COMMENTED OUT: Legacy memory-based execution
    async def _execute_legacy(self, user_goal: str) -> str:
        """Legacy execution mode (original implementation)"""
        print("⚠️ Using legacy memory-based mode\n")
        
        # Original implementation would go here
        # For now, redirect to blackboard
        return await self._execute_blackboard(user_goal)
    
    """
    # COMMENTED OUT: Memory-based methods (no longer used in blackboard mode)
    
    async def _build_comprehensive_memory_context(self, query: str) -> str:
        # Memory retrieval logic
        pass
    
    def _format_planner_input(self, user_goal: str, memory_context: str) -> str:
        # Memory-based formatting
        pass
    
    async def _save_to_memory(self, content: str, importance: int, memory_type: str):
        # Memory saving logic
        pass
    """

# import json
# from autogen_core.memory import MemoryContent, MemoryMimeType

# class MemoryEnabledOrchestrator:
#     def __init__(self, planner_agent, agents_dict, memory_system=None):
#         self.planner = planner_agent
#         self.agents = agents_dict
#         self.memory = memory_system

#     def _truncate_content(self, content: str, max_length: int = 200) -> str:
#         """Truncate content to max_length characters"""
#         if len(content) > max_length:
#             return content[:max_length] + "..."
#         return content

#     async def execute(self, user_goal: str, use_memory: bool = True) -> str:
#         print(f"Goal: {user_goal}")
#         print(f"{'='*70}\n")
        
#         memory_context = ""
#         if self.memory and use_memory:
#             memory_context = await self._build_comprehensive_memory_context(user_goal)
        
#         print("Phase 1: Planning...")
#         plan_input = self._format_planner_input(user_goal, memory_context)
#         plan_response = await self.planner.run(plan_input)
        
#         if self.memory:
#             await self._save_to_memory(
#                 f"User asked: {user_goal}",
#                 importance=6,
#                 memory_type="episodic"
#             )
        
#         plan = self._parse_plan(plan_response)
#         steps = plan.get("steps", [])
#         print(f"Plan created with {len(steps)} steps\n")
        
#         for i, step in enumerate(steps, 1):
#             print(f" {i}. {step['agent']}: {step['task']}")
#         print()
        
#         print("Phase 2: Execution...\n")
#         results = []
        
#         for i, step in enumerate(steps, 1):
#             agent_name = step.get("agent")
#             task = step.get("task")

#             task_display = task[:80] + "..." if len(task) > 80 else task
#             print(f"[{i}/{len(steps)}] {agent_name}: {task_display}")

#             if agent_name not in self.agents:
#                 print(f"Agent '{agent_name}' not found, skipping\n")
#                 continue
            
#             context = await self._build_agent_context(
#                 task=task,
#                 previous_results=results,
#                 original_goal=user_goal,
#                 memory_context=memory_context,
#                 agent_name=agent_name
#             )
            
#             agent = self.agents[agent_name]
#             result = await agent.run(context)

#             print(f"Completed ({len(result)} characters)\n")

#             results.append({
#                 "agent": agent_name,
#                 "task": task,
#                 "output": result
#             })
            
#             if self.memory and agent_name in ["Researcher", "Analyst", "Reporter", "Coder"]:
#                 await self._save_to_memory(
#                     f"{agent_name} output: {self._truncate_content(result, 300)}",
#                     importance=6,
#                     memory_type="episodic"
#                 )
        
#         print(f"\n{'='*70}")
#         print("Execution Complete!")
#         print(f"{'='*70}\n")
        
#         final_result = self._compile_results(results)
        
#         if self.memory:
#             await self._save_to_memory(
#                 f"Completed task: {user_goal}. Result: {self._truncate_content(final_result, 200)}",
#                 importance=7,
#                 memory_type="semantic"
#             )
        
#         return final_result

#     async def _build_comprehensive_memory_context(self, query: str) -> str:
#         context_parts = []
        
#         important_memories = await self.memory.long_term.get_important_memories(
#             min_importance=7,
#             limit=3
#         )
        
#         if important_memories:
#             facts = []
#             for mem in important_memories:
#                 facts.append(f" • {self._truncate_content(mem.content, 150)}")
#             if facts:
#                 context_parts.append(
#                     "=== IMPORTANT INFORMATION ===\n" +
#                     "\n".join(facts)
#                 )
        
#         similar_memories = await self.memory.vector.query(query)
#         if similar_memories:
#             relevant = []
#             for mem in similar_memories[:2]:
#                 if mem.content not in [m.content for m in important_memories]:
#                     relevant.append(f" • {self._truncate_content(mem.content, 150)}")
#             if relevant:
#                 context_parts.append(
#                     "\n=== RELEVANT PAST CONTEXT ===\n" +
#                     "\n".join(relevant)
#                 )
        
#         recent = self.memory.session.get_recent(n=2)
#         if recent:
#             recent_items = []
#             for mem in recent:
#                 recent_items.append(f" • {self._truncate_content(mem.content, 150)}")
#             if recent_items:
#                 context_parts.append(
#                     "\n=== RECENT CONVERSATION ===\n" +
#                     "\n".join(recent_items)
#                 )
        
#         return "\n".join(context_parts) if context_parts else ""

#     def _format_planner_input(self, user_goal: str, memory_context: str) -> str:
#         if not memory_context:
#             return user_goal
#         return f"""{self._truncate_content(memory_context, 500)}

# === USER REQUEST ===
# {user_goal}

# Please create a detailed plan considering the above context."""

#     async def _build_agent_context(
#         self,
#         task: str,
#         previous_results: list,
#         original_goal: str,
#         memory_context: str,
#         agent_name: str
#     ) -> str:
#         context_parts = []
        
#         if memory_context:
#             context_parts.append(self._truncate_content(memory_context, 400))
        
#         if self.memory:
#             task_memories = await self.memory.vector.query(task)
#             if task_memories:
#                 task_context = []
#                 for mem in task_memories[:2]:
#                     task_context.append(f" • {self._truncate_content(mem.content, 100)}")
#                 if task_context:
#                     context_parts.append(
#                         f"\n=== RELEVANT TO THIS TASK ===\n" +
#                         "\n".join(task_context)
#                     )
        
#         context_parts.append(f"\n=== ORIGINAL GOAL ===\n{self._truncate_content(original_goal, 150)}")
#         context_parts.append(f"\n=== YOUR TASK ===\n{self._truncate_content(task, 250)}")
        
#         if previous_results:
#             prev = []
#             for r in previous_results[-1:]:
#                 prev.append(f" • {r['agent']}: {self._truncate_content(r['output'], 150)}")
#             if prev:
#                 context_parts.append(
#                     f"\n=== PREVIOUS STEPS ===\n" +
#                     "\n".join(prev)
#                 )
        
#         return "\n".join(context_parts)

#     async def _save_to_memory(
#         self,
#         content: str,
#         importance: int = 5,
#         memory_type: str = "episodic"
#     ) -> None:
#         if not self.memory:
#             return
        
#         content = self._truncate_content(content, 500)
        
#         memory_content = MemoryContent(
#             content=content,
#             mime_type=MemoryMimeType.TEXT,
#             metadata={
#                 "importance": importance,
#                 "type": memory_type
#             }
#         )
        
#         await self.memory.add(memory_content, store_long_term=True)

#     def _parse_plan(self, plan_response: str) -> dict:
#         try:
#             return json.loads(plan_response)
#         except json.JSONDecodeError:
#             start = plan_response.find("{")
#             end = plan_response.rfind("}") + 1
#             if start != -1 and end > start:
#                 return json.loads(plan_response[start:end])
#             raise ValueError(f"Could not parse plan: {plan_response}")

#     def _compile_results(self, results: list) -> str:
#         if not results:
#             return "No results generated."
#         return results[-1]['output']

#     async def get_memory_stats(self) -> dict:
#         if not self.memory:
#             return {"status": "No memory system attached"}
#         return self.memory.get_memory_stats()

#     async def clear_session_memory(self) -> None:
#         if self.memory:
#             await self.memory.clear_session()
#             print("Session memory cleared")

#     async def save_important_fact(self, fact: str, importance: int = 9) -> None:
#         if self.memory:
#             await self.memory.save_important_fact(fact, importance)
