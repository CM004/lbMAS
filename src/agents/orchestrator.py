import json
from autogen_core.memory import MemoryContent, MemoryMimeType

class MemoryEnabledOrchestrator:
    def __init__(self, planner_agent, agents_dict, memory_system=None):
        self.planner = planner_agent
        self.agents = agents_dict
        self.memory = memory_system

    def _truncate_content(self, content: str, max_length: int = 200) -> str:
        """Truncate content to max_length characters"""
        if len(content) > max_length:
            return content[:max_length] + "..."
        return content

    async def execute(self, user_goal: str, use_memory: bool = True) -> str:
        print(f"Goal: {user_goal}")
        print(f"{'='*70}\n")
        
        memory_context = ""
        if self.memory and use_memory:
            memory_context = await self._build_comprehensive_memory_context(user_goal)
        
        print("Phase 1: Planning...")
        plan_input = self._format_planner_input(user_goal, memory_context)
        plan_response = await self.planner.run(plan_input)
        
        if self.memory:
            await self._save_to_memory(
                f"User asked: {user_goal}",
                importance=6,
                memory_type="episodic"
            )
        
        plan = self._parse_plan(plan_response)
        steps = plan.get("steps", [])
        print(f"Plan created with {len(steps)} steps\n")
        
        for i, step in enumerate(steps, 1):
            print(f" {i}. {step['agent']}: {step['task']}")
        print()
        
        print("Phase 2: Execution...\n")
        results = []
        
        for i, step in enumerate(steps, 1):
            agent_name = step.get("agent")
            task = step.get("task")

            task_display = task[:80] + "..." if len(task) > 80 else task
            print(f"[{i}/{len(steps)}] {agent_name}: {task_display}")

            if agent_name not in self.agents:
                print(f"Agent '{agent_name}' not found, skipping\n")
                continue
            
            context = await self._build_agent_context(
                task=task,
                previous_results=results,
                original_goal=user_goal,
                memory_context=memory_context,
                agent_name=agent_name
            )
            
            agent = self.agents[agent_name]
            result = await agent.run(context)

            print(f"Completed ({len(result)} characters)\n")

            results.append({
                "agent": agent_name,
                "task": task,
                "output": result
            })
            
            if self.memory and agent_name in ["Researcher", "Analyst", "Reporter", "Coder"]:
                await self._save_to_memory(
                    f"{agent_name} output: {self._truncate_content(result, 300)}",
                    importance=6,
                    memory_type="episodic"
                )
        
        print(f"\n{'='*70}")
        print("Execution Complete!")
        print(f"{'='*70}\n")
        
        final_result = self._compile_results(results)
        
        if self.memory:
            await self._save_to_memory(
                f"Completed task: {user_goal}. Result: {self._truncate_content(final_result, 200)}",
                importance=7,
                memory_type="semantic"
            )
        
        return final_result

    async def _build_comprehensive_memory_context(self, query: str) -> str:
        context_parts = []
        
        important_memories = await self.memory.long_term.get_important_memories(
            min_importance=7,
            limit=3
        )
        
        if important_memories:
            facts = []
            for mem in important_memories:
                facts.append(f" • {self._truncate_content(mem.content, 150)}")
            if facts:
                context_parts.append(
                    "=== IMPORTANT INFORMATION ===\n" +
                    "\n".join(facts)
                )
        
        similar_memories = await self.memory.vector.query(query)
        if similar_memories:
            relevant = []
            for mem in similar_memories[:2]:
                if mem.content not in [m.content for m in important_memories]:
                    relevant.append(f" • {self._truncate_content(mem.content, 150)}")
            if relevant:
                context_parts.append(
                    "\n=== RELEVANT PAST CONTEXT ===\n" +
                    "\n".join(relevant)
                )
        
        recent = self.memory.session.get_recent(n=2)
        if recent:
            recent_items = []
            for mem in recent:
                recent_items.append(f" • {self._truncate_content(mem.content, 150)}")
            if recent_items:
                context_parts.append(
                    "\n=== RECENT CONVERSATION ===\n" +
                    "\n".join(recent_items)
                )
        
        return "\n".join(context_parts) if context_parts else ""

    def _format_planner_input(self, user_goal: str, memory_context: str) -> str:
        if not memory_context:
            return user_goal
        return f"""{self._truncate_content(memory_context, 500)}

=== USER REQUEST ===
{user_goal}

Please create a detailed plan considering the above context."""

    async def _build_agent_context(
        self,
        task: str,
        previous_results: list,
        original_goal: str,
        memory_context: str,
        agent_name: str
    ) -> str:
        context_parts = []
        
        if memory_context:
            context_parts.append(self._truncate_content(memory_context, 400))
        
        if self.memory:
            task_memories = await self.memory.vector.query(task)
            if task_memories:
                task_context = []
                for mem in task_memories[:2]:
                    task_context.append(f" • {self._truncate_content(mem.content, 100)}")
                if task_context:
                    context_parts.append(
                        f"\n=== RELEVANT TO THIS TASK ===\n" +
                        "\n".join(task_context)
                    )
        
        context_parts.append(f"\n=== ORIGINAL GOAL ===\n{self._truncate_content(original_goal, 150)}")
        context_parts.append(f"\n=== YOUR TASK ===\n{self._truncate_content(task, 250)}")
        
        if previous_results:
            prev = []
            for r in previous_results[-1:]:
                prev.append(f" • {r['agent']}: {self._truncate_content(r['output'], 150)}")
            if prev:
                context_parts.append(
                    f"\n=== PREVIOUS STEPS ===\n" +
                    "\n".join(prev)
                )
        
        return "\n".join(context_parts)

    async def _save_to_memory(
        self,
        content: str,
        importance: int = 5,
        memory_type: str = "episodic"
    ) -> None:
        if not self.memory:
            return
        
        content = self._truncate_content(content, 500)
        
        memory_content = MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT,
            metadata={
                "importance": importance,
                "type": memory_type
            }
        )
        
        await self.memory.add(memory_content, store_long_term=True)

    def _parse_plan(self, plan_response: str) -> dict:
        try:
            return json.loads(plan_response)
        except json.JSONDecodeError:
            start = plan_response.find("{")
            end = plan_response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(plan_response[start:end])
            raise ValueError(f"Could not parse plan: {plan_response}")

    def _compile_results(self, results: list) -> str:
        if not results:
            return "No results generated."
        return results[-1]['output']

    async def get_memory_stats(self) -> dict:
        if not self.memory:
            return {"status": "No memory system attached"}
        return self.memory.get_memory_stats()

    async def clear_session_memory(self) -> None:
        if self.memory:
            await self.memory.clear_session()
            print("Session memory cleared")

    async def save_important_fact(self, fact: str, importance: int = 9) -> None:
        if self.memory:
            await self.memory.save_important_fact(fact, importance)
