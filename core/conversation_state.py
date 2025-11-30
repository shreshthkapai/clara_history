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

class RedFlagEvent(BaseModel):
    category: str  
    severity: str  
    triggered_at: datetime = Field(default_factory=datetime.now)
    patient_response: Optional[str] = None  
    action_taken: str  

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
    max_questions: int = 20
    
    red_flags_detected: List[RedFlagEvent] = Field(default_factory=list)
    conversation_ended_for_emergency: bool = False
    
    checklist_template: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if self.checklist_template:
            self._initialize_checklist()
    
    def _initialize_checklist(self):
        template_checklist = self.checklist_template.get('checklist', {})
        self.checklist = {k: v.copy() for k, v in template_checklist.items()}
        
        for topic, details in self.checklist.items():
            if details.get('required', False):
                self.topics_required.append(topic)
            else:
                self.topics_optional.append(topic)
        
        rules = self.checklist_template.get('conversation_rules', {})
        self.max_questions = rules.get('max_questions', 20)

    def add_message(self, speaker: str, text: str, topic: Optional[str] = None, flags: Optional[List[str]] = None):
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
        if topic in self.checklist:
            self.checklist[topic]['completed'] = True
            if topic not in self.topics_completed:
                self.topics_completed.append(topic)
    
    def is_topic_complete(self, topic: str) -> bool:
        return self.checklist.get(topic, {}).get('completed', False)
    
    def get_incomplete_required_topics(self) -> List[str]:
        return [
            topic for topic in self.topics_required 
            if not self.is_topic_complete(topic)
        ]
    
    def get_next_priority_topic(self) -> Optional[str]:
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

    def record_red_flag(self, category: str, severity: str, patient_response: Optional[str] = None, action_taken: str = "continued_with_warning"):
        event = RedFlagEvent(
            category=category,
            severity=severity,
            patient_response=patient_response,
            action_taken=action_taken
        )
        self.red_flags_detected.append(event)
        
        if action_taken == "ended_conversation":
            self.conversation_ended_for_emergency = True
            self.status = "ended_early_emergency"
            self.ended_at = datetime.now()
    
    def should_end_conversation(self) -> tuple[bool, str]:
        """
        Determine if conversation should end
        Returns: (should_end, reason)
        """
        
        # Already ended for emergency
        if self.conversation_ended_for_emergency:
            return (True, "emergency")
        
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
        return {
            "questions_asked": self.question_count,
            "max_questions": self.max_questions,
            "required_topics_completed": len([t for t in self.topics_required if self.is_topic_complete(t)]),
            "required_topics_total": len(self.topics_required),
            "optional_topics_completed": len([t for t in self.topics_optional if self.is_topic_complete(t)]),
            "optional_topics_total": len(self.topics_optional),
            "red_flags_detected": len(self.red_flags_detected),
            "status": self.status
        }
    
    
    def end_conversation(self, status: str = "completed"):
        self.status = status
        self.ended_at = datetime.now()
    
    def get_transcript(self) -> List[Dict[str, Any]]:
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
            "red_flags_detected": [
                {
                    "category": flag.category,
                    "severity": flag.severity,
                    "triggered_at": flag.triggered_at.isoformat(),
                    "patient_response": flag.patient_response,
                    "action_taken": flag.action_taken
                }
                for flag in self.red_flags_detected
            ],
            "progress": self.get_progress_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_template(cls, conversation_id: str, patient_name: str, doctor_name: str):
        
        template_path = Path("data/checklist_template.json")
        
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        return cls(
            conversation_id=conversation_id,
            patient_name=patient_name,
            doctor_name=doctor_name,
            checklist_template=template
        )
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_template(cls, conversation_id: str, patient_name: str, doctor_name: str):
        
        template_path = Path("data/checklist_template.json")
        
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        return cls(
            conversation_id=conversation_id,
            patient_name=patient_name,
            doctor_name=doctor_name,
            checklist_template=template
        )
    @classmethod
    def load_from_template(cls, conversation_id: str, patient_name: str, doctor_name: str):
        
        template_path = Path("data/checklist_template.json")
        
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        return cls(
            conversation_id=conversation_id,
            patient_name=patient_name,
            doctor_name=doctor_name,
            checklist_template=template
        )