from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import json
from pathlib import Path

class Message(BaseModel):
    speaker: str  
    text: str
    timestamp: datetime = Field(default_factory=datetime.now)
    topic: Optional[str] = None  
    flags: List[str] = Field(default_factory=list)


class ConversationState(BaseModel):
    conversation_id: str
    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    appointment_id: Optional[str] = None
    
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    status: str = "in_progress"  
    
    messages: List[Message] = Field(default_factory=list)
    
    checklist: Dict[str, Any] = Field(default_factory=dict)
    topics_completed: List[str] = Field(default_factory=list)
    topics_required: List[str] = Field(default_factory=list)
    topics_optional: List[str] = Field(default_factory=list)
    
    question_count: int = 0
    max_questions: int = 30
    
    checklist_template: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if self.checklist_template:
            self._initialize_checklist()
    
    def _initialize_checklist(self):
        """Initialize checklist from template"""
        template_checklist = self.checklist_template.get('checklist', {})
        self.checklist = {k: v.copy() for k, v in template_checklist.items()}
        
        for topic, details in self.checklist.items():
            if details.get('required', False):
                self.topics_required.append(topic)
            else:
                self.topics_optional.append(topic)
        
        rules = self.checklist_template.get('conversation_rules', {})
        self.max_questions = rules.get('max_questions', 30)

    def add_message(self, speaker: str, text: str, topic: Optional[str] = None, flags: Optional[List[str]] = None):
        """Add message to conversation"""
        message = Message(
            speaker=speaker,
            text=text,
            topic=topic,
            flags=flags or []
        )
        self.messages.append(message)
        
        if speaker == 'clara':
            self.question_count += 1
    
    def mark_topic_complete(self, topic: str):
        """Mark a topic as completed"""
        if topic in self.checklist:
            self.checklist[topic]['completed'] = True
            if topic not in self.topics_completed:
                self.topics_completed.append(topic)
    
    def is_topic_complete(self, topic: str) -> bool:
        """Check if topic is completed"""
        return self.checklist.get(topic, {}).get('completed', False)
    
    def get_incomplete_required_topics(self) -> List[str]:
        """Get list of incomplete required topics"""
        return [
            topic for topic in self.topics_required 
            if not self.is_topic_complete(topic)
        ]
    
    def get_next_priority_topic(self) -> Optional[str]:
        """Get next topic to ask about based on priority"""
        incomplete = self.get_incomplete_required_topics()
        
        if not incomplete:
            incomplete = [
                topic for topic in self.topics_optional
                if not self.is_topic_complete(topic)
            ]
        
        if not incomplete:
            return None
        
        incomplete_with_priority = [
            (topic, self.checklist[topic].get('priority', 999))
            for topic in incomplete
        ]
        incomplete_with_priority.sort(key=lambda x: x[1])
        
        return incomplete_with_priority[0][0]

    def should_end_conversation(self) -> tuple[bool, str]:
        """
        Determine if conversation should end
        Returns: (should_end, reason)
        """
        
        # Hit question limit - hard stop
        if self.question_count >= self.max_questions:
            return (True, "max_questions")
        
        # If closing topic is marked complete, END immediately
        if self.is_topic_complete('closing'):
            return (True, "completed")
        
        # If all required topics done (including closing)
        incomplete_required = self.get_incomplete_required_topics()
        if not incomplete_required:
            return (True, "completed")
        
        return (False, "continue")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get conversation progress summary"""
        return {
            "questions_asked": self.question_count,
            "max_questions": self.max_questions,
            "required_topics_completed": len([t for t in self.topics_required if self.is_topic_complete(t)]),
            "required_topics_total": len(self.topics_required),
            "optional_topics_completed": len([t for t in self.topics_optional if self.is_topic_complete(t)]),
            "optional_topics_total": len(self.topics_optional),
            "status": self.status
        }
    
    def end_conversation(self, status: str = "completed"):
        """End the conversation"""
        self.status = status
        self.ended_at = datetime.now()
    
    def get_transcript(self) -> List[Dict[str, Any]]:
        """Get full transcript as list of dicts"""
        return [
            {
                "speaker": msg.speaker,
                "text": msg.text,
                "timestamp": msg.timestamp.isoformat(),
                "topic": msg.topic,
                "flags": msg.flags
            }
            for msg in self.messages
        ]
    
    def save_to_file(self, filepath: Path):
        """Save conversation to JSON file"""
        data = {
            "conversation_id": self.conversation_id,
            "patient_name": self.patient_name,
            "doctor_name": self.doctor_name,
            "appointment_id": self.appointment_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "status": self.status,
            "question_count": self.question_count,
            "transcript": self.get_transcript(),
            "topics_completed": self.topics_completed,
            "progress": self.get_progress_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_template(cls, conversation_id: str, patient_name: str, doctor_name: str):
        """Load conversation state from template"""
        template_path = Path("data/checklist_template.json")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template = json.load(f)
        
        return cls(
            conversation_id=conversation_id,
            patient_name=patient_name,
            doctor_name=doctor_name,
            checklist_template=template
        )