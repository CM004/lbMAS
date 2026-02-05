"""
Blackboard system for LbMAS - replaces individual agent memory
Based on research paper: Exploring Advanced LLM Multi-Agent Systems 
Based on Blackboard Architecture (2025)
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

class SpaceType(Enum):
    PUBLIC = "public"
    PRIVATE = "private"

@dataclass
class BlackboardMessage:
    """Message stored on blackboard"""
    agent_name: str
    content: str
    space_type: SpaceType = SpaceType.PUBLIC
    round_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "content": self.content,
            "space": self.space_type.value,
            "round": self.round_number,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }

class Blackboard:
    """
    Central blackboard for agent communication in LbMAS.
    Replaces individual agent memory modules.
    All agents communicate solely through this blackboard.
    """
    def __init__(self):
        self.public_messages: List[BlackboardMessage] = []
        self.private_spaces: Dict[str, List[BlackboardMessage]] = {}
        self.current_round = 0
        
    def write_public(self, agent_name: str, content: str, metadata: Optional[Dict] = None):
        """Write message to public blackboard space"""
        message = BlackboardMessage(
            agent_name=agent_name,
            content=content,
            space_type=SpaceType.PUBLIC,
            round_number=self.current_round,
            metadata=metadata
        )
        self.public_messages.append(message)
        print(f"✓ [BLACKBOARD] {agent_name} wrote to public space (Round {self.current_round})")
        
    def write_private(self, space_id: str, agent_name: str, content: str, metadata: Optional[Dict] = None):
        """Write to private space for debates/conflict resolution"""
        if space_id not in self.private_spaces:
            self.private_spaces[space_id] = []
        
        message = BlackboardMessage(
            agent_name=agent_name,
            content=content,
            space_type=SpaceType.PRIVATE,
            round_number=self.current_round,
            metadata=metadata
        )
        self.private_spaces[space_id].append(message)
        print(f"✓ [BLACKBOARD] {agent_name} wrote to private space '{space_id}'")
        
    def get_public_content(self, max_chars: int = 8000) -> str:
        """Get public messages formatted for agent consumption"""
        if not self.public_messages:
            return "The blackboard is empty."
        
        content = "=== BLACKBOARD PUBLIC SPACE ===\n\n"
        total_chars = 0
        
        # Get most recent messages first
        for msg in reversed(self.public_messages):
            msg_text = f"[Round {msg.round_number}] {msg.agent_name}:\n{msg.content}\n\n"
            if total_chars + len(msg_text) > max_chars:
                break
            content = msg_text + content
            total_chars += len(msg_text)
            
        return content
    
    def get_private_content(self, space_id: str) -> str:
        """Get private space messages"""
        if space_id not in self.private_spaces or not self.private_spaces[space_id]:
            return "This private space is empty."
        
        content = f"=== PRIVATE SPACE: {space_id} ===\n\n"
        for msg in self.private_spaces[space_id]:
            content += f"[Round {msg.round_number}] {msg.agent_name}:\n{msg.content}\n\n"
        return content
    
    def remove_messages(self, message_indices: List[int]):
        """Remove useless/redundant messages (Cleaner agent)"""
        for idx in sorted(message_indices, reverse=True):
            if 0 <= idx < len(self.public_messages):
                removed = self.public_messages.pop(idx)
                print(f"✓ [BLACKBOARD] Removed message from {removed.agent_name}")
    
    def increment_round(self):
        """Increment the round counter"""
        self.current_round += 1
        print(f"\n{'='*70}")
        print(f"⚡ BLACKBOARD CYCLE - ROUND {self.current_round}")
        print(f"{'='*70}\n")
    
    def get_all_content(self) -> Dict[str, Any]:
        """Get complete blackboard state"""
        return {
            "round": self.current_round,
            "public": self.get_public_content(),
            "private_spaces": {k: self.get_private_content(k) for k in self.private_spaces.keys()},
            "message_count": len(self.public_messages)
        }
    
    def get_recent_messages(self, n: int = 5) -> List[BlackboardMessage]:
        """Get most recent n messages"""
        return self.public_messages[-n:] if n < len(self.public_messages) else self.public_messages.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get blackboard statistics"""
        return {
            "current_round": self.current_round,
            "total_messages": len(self.public_messages),
            "private_spaces": len(self.private_spaces),
            "agents_participated": len(set(msg.agent_name for msg in self.public_messages))
        }
