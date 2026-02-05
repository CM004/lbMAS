from typing import List, Any
from autogen_core.memory import Memory, MemoryContent
from datetime import datetime

class SessionMemory(Memory):
    def __init__(self, max_turns: int = 50):
        self.memory: List[MemoryContent] = []
        self.max_turns = max_turns
    
    async def add(self, content: MemoryContent) -> None:
        if content.metadata is None:
            content.metadata = {}
        content.metadata["timestamp"] = datetime.now().isoformat()
        
        self.memory.append(content)
        
        # Keep only last N turns
        if len(self.memory) > self.max_turns:
            self.memory = self.memory[-self.max_turns:]
    
    async def query(self, query: str) -> List[MemoryContent]:
        return self.memory.copy()
    
    async def clear(self) -> None:
        self.memory.clear()
    
    async def close(self) -> None:
        pass
    
    async def update_context(self, model_context: Any) -> None:
        pass
    
    def get_recent(self, n: int = 5) -> List[MemoryContent]:
        return self.memory[-n:] if n < len(self.memory) else self.memory.copy()
    
    def __len__(self) -> int:
        return len(self.memory)
