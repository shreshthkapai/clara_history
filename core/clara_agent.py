from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import uuid

from core.conversation_state import ConversationState
from services.azure_openai import AzureOpenAIService


class ClaraAgent:

    def __init__(self, patient_name: str, doctor_name: str, appointment_id: Optional[str] = None):

        self.patient_name = patient_name
        self.doctor_name = doctor_name
        self.appointment_id = appointment_id or str(uuid.uuid4())

        self.openai_service = AzureOpenAIService()

        self.checklist_template = self._load_checklist_template()

        self.state = ConversationState.load_from_template(
            conversation_id=str(uuid.uuid4()),
            patient_name=patient_name,
            doctor_name=doctor_name
        )
        self.state.appointment_id = self.appointment_id

        self.conversation_started = False

    def _load_checklist_template(self) -> Dict:
        template_path = Path("data/checklist_template.json")
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def start_conversation(self) -> str:
        """Start conversation with greeting + emergency warning + first question"""
        self.conversation_started = True

        # Get opening message
        opening_message = self.checklist_template.get(
            'opening_message',
            f"Hi! I'm Clara. I'll be asking you some questions about your upcoming appointment with {self.doctor_name}."
        )

        # Get first question
        first_question = self.checklist_template['checklist']['chief_complaint']['questions'][0]

        # Combine: greeting + first question (emergency warning will be shown separately in UI)
        full_opening = f"{opening_message}\n\n{first_question}"

        self.state.add_message(
            speaker="clara",
            text=full_opening,
            topic="opening"
        )

        return full_opening

    def process_patient_response(self, patient_message: str) -> Tuple[str, bool, Optional[str]]:
        """Process patient response and generate next question"""
        
        # Add patient's message
        self.state.add_message(
            speaker="patient",
            text=patient_message
        )

        # Check max questions BEFORE getting AI decision
        if self.state.question_count >= self.state.max_questions:
            closing_message = self._generate_closing_message()
            self.state.add_message(
                speaker="clara",
                text=closing_message,
                topic="closing"
            )
            self.state.end_conversation(status="completed")
            return (closing_message, True, "max_questions")

        # Get AI decision
        decision = self._get_clara_decision()

        # Validate and mark completed topics (only if they exist in checklist)
        for topic in decision.get('topics_completed', []):
            if topic in self.state.checklist:
                self.state.mark_topic_complete(topic)
            else:
                print(f"⚠️ AI suggested unknown topic to complete: '{topic}' - ignoring")

        # Skip irrelevant optional topics (only if they exist and are optional)
        for topic in decision.get('optional_topics_to_skip', []):
            if topic in self.state.topics_optional:
                self.state.mark_topic_complete(topic)
            elif topic in self.state.checklist:
                print(f"⚠️ AI tried to skip required topic: '{topic}' - ignoring")
            else:
                print(f"⚠️ AI suggested unknown topic to skip: '{topic}' - ignoring")

        # Check if AI says conversation should end
        if decision.get('conversation_complete', False):
            closing_message = self._generate_closing_message()
            self.state.add_message(
                speaker="clara",
                text=closing_message,
                topic="closing"
            )
            self.state.end_conversation(status="completed")
            return (closing_message, True, "completed")

        # Continue with next question
        next_question = decision.get('next_question', "Is there anything else you'd like to share?")
        current_topic = decision.get('current_topic', 'closing')

        self.state.add_message(
            speaker="clara",
            text=next_question,
            topic=current_topic
        )

        return (next_question, False, None)

    def _get_clara_decision(self) -> Dict:
        """Get AI decision for next question"""
        system_prompt = self._build_smart_system_prompt()
        conversation_history = self._build_conversation_history()
        decision = self.openai_service.get_clara_decision_json(
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=500
        )
        return decision

    def _build_smart_system_prompt(self) -> str:
        """Build system prompt for AI decision-making"""
        
        required_topics = [
            topic for topic in self.state.topics_required 
            if not self.state.is_topic_complete(topic)
        ]

        optional_topics = [
            topic for topic in self.state.topics_optional
            if not self.state.is_topic_complete(topic)
        ]

        progress = self.state.get_progress_summary()

        # Build pacing warnings conditionally
        pacing_notes = []
        if progress['questions_asked'] >= 25:
            pacing_notes.append("- URGENT: Approaching question limit. Wrap up quickly. Focus only on critical missing info.")
        elif progress['questions_asked'] >= 20:
            pacing_notes.append("- Prioritize essential topics only. Be concise.")
        
        pacing_section = "\n".join(pacing_notes) if pacing_notes else ""

        system_prompt = f"""You are Clara, a medical history-taking assistant for {self.patient_name} before their appointment with {self.doctor_name}.

YOUR ROLE:
- You are NOT a doctor. You collect medical history systematically.
- NEVER give medical advice, diagnosis, reassurance, or treatment suggestions.
- NEVER comment on symptom severity ("that sounds serious" / "that's reassuring").
- Ask ONE clear, focused question at a time.

RESPOND WITH JSON ONLY (no markdown, no ```json blocks):
{{
  "conversation_complete": boolean,
  "topics_completed": ["topic1", "topic2"],
  "optional_topics_to_skip": ["topic1"],
  "current_topic": "topic_name",
  "next_question": "Your question here"
}}

REQUIRED TOPICS (must complete all):
{json.dumps(required_topics)}

OPTIONAL TOPICS (only ask if relevant):
{json.dumps(optional_topics)}
- family_history: Only if patient mentions family conditions
- systems_review: Only if symptoms suggest multi-system issues
- gynae_sexual: Only if clearly relevant (pelvic pain, menopause, etc.)

CONVERSATION COMPLETE when:
- All required topics covered AND
- Patient said "no"/"nothing"/"that's all"/"nope" to closing question
- When setting conversation_complete to true, next_question can be empty string

TOPIC COMPLETION:
- Mark topics in "topics_completed" ONLY when patient has given sufficient information
- Use exact topic names from the lists above
- Mark irrelevant optional topics in "optional_topics_to_skip"

PROGRESS:
- Questions: {progress['questions_asked']}/{progress['max_questions']}
- Required topics done: {progress['required_topics_completed']}/{progress['required_topics_total']}
- Topics completed: {self.state.topics_completed}

PACING:
{pacing_section}

Generate your JSON response now."""

        return system_prompt

    def _build_conversation_history(self) -> List[Dict[str, str]]:
        """Build conversation history for AI context"""
        recent_messages = self.state.messages[-30:]
        history = []
        for msg in recent_messages:
            role = "assistant" if msg.speaker == "clara" else "user"
            history.append({
                "role": role,
                "content": msg.text
            })
        return history

    def _generate_closing_message(self) -> str:
        """Generate closing message"""
        closing_script = f"""Thank you so much for taking the time to speak with me today, {self.patient_name}. Your responses will help {self.doctor_name} provide you with the best possible care.

If you feel there was anything you forgot to mention or would like to add, you can use the same link to speak with me again before your appointment.

Take care, and I hope your appointment goes well!"""
        return closing_script

    def get_conversation_summary(self) -> Dict[str, any]:
        """Get conversation summary stats"""
        return {
            "conversation_id": self.state.conversation_id,
            "patient_name": self.patient_name,
            "doctor_name": self.doctor_name,
            "status": self.state.status,
            "progress": self.state.get_progress_summary(),
            "transcript_length": len(self.state.messages)
        }

    def save_conversation(self, save_dir: Path = Path("data/conversations")):
        """Save conversation to file"""
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self.patient_name.replace(' ', '_')}_{self.state.conversation_id[:8]}.json"
        filepath = save_dir / filename
        self.state.save_to_file(filepath)
        return filepath