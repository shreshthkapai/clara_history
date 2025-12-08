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
        with open(template_path, 'r') as f:
            return json.load(f)

    def start_conversation(self) -> str:
        self.conversation_started = True

        opening_message = self.checklist_template.get(
            'opening_message',
            f"Hi! I'm Clara. I'll be asking you some questions about your upcoming appointment with {self.doctor_name}."
        )

        first_question = self.checklist_template['checklist']['chief_complaint']['questions'][0]

        full_opening = f"{opening_message}\n\n{first_question}"

        self.state.add_message(
            speaker="clara",
            text=full_opening,
            topic="opening"
        )

        return full_opening

    def process_patient_response(self, patient_message: str) -> Tuple[str, bool, Optional[str]]:
        self.state.add_message(
            speaker="patient",
            text=patient_message
        )

        decision = self._get_clara_decision()

        if decision.get('red_flag_detected', False):
            return self._handle_red_flag_detected(decision)

        for topic in decision.get('topics_completed', []):
            self.state.mark_topic_complete(topic)

        for topic in decision.get('optional_topics_to_skip', []):
            if topic in self.state.topics_optional:
                self.state.mark_topic_complete(topic)

        if decision.get('conversation_complete', False):
            closing_message = self._generate_closing_message()
            self.state.add_message(
                speaker="clara",
                text=closing_message,
                topic="closing"
            )
            self.state.end_conversation(status="completed")
            return (closing_message, True, "completed")

        if self.state.question_count >= self.state.max_questions:
            closing_message = self._generate_closing_message()
            self.state.add_message(
                speaker="clara",
                text=closing_message,
                topic="closing"
            )
            self.state.end_conversation(status="completed")
            return (closing_message, True, "max_questions")

        next_question = decision.get('next_question', "Is there anything else you'd like to share?")
        current_topic = decision.get('current_topic', 'closing')

        self.state.add_message(
            speaker="clara",
            text=next_question,
            topic=current_topic
        )

        return (next_question, False, None)

    def _get_clara_decision(self) -> Dict:
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
        required_topics = [
            topic for topic in self.state.topics_required 
            if not self.state.is_topic_complete(topic)
        ]

        optional_topics = [
            topic for topic in self.state.topics_optional
            if not self.state.is_topic_complete(topic)
        ]

        red_flag_categories = self.checklist_template['checklist']['red_flags']['red_flag_categories']
        red_flag_list = []
        for category, details in red_flag_categories.items():
            triggers = details.get('triggers', [])
            red_flag_list.append(f"  - {category}: {', '.join(triggers[:3])}")
        red_flag_text = "\n".join(red_flag_list)

        progress = self.state.get_progress_summary()

        system_prompt = f"""You are Clara, a medical history-taking assistant for {self.patient_name} before their appointment with {self.doctor_name}.

YOUR ROLE:
- You are NOT a doctor. You collect medical history systematically.
- NEVER give medical advice, diagnosis, reassurance, or treatment suggestions.
- NEVER comment on symptom severity ("that sounds serious" / "that's reassuring").
- Ask ONE clear, focused question at a time.

RESPOND WITH JSON ONLY:
{{
  "red_flag_detected": boolean,
  "red_flag_category": "category_name" or null,
  "conversation_complete": boolean,
  "topics_completed": ["topic1", "topic2"],
  "optional_topics_to_skip": ["topic1"],
  "current_topic": "topic_name",
  "next_question": "Your question here"
}}

RED FLAGS (set red_flag_detected=true if ANY detected):
{red_flag_text}

REQUIRED TOPICS (must complete all):
{json.dumps(required_topics)}

OPTIONAL TOPICS (only ask if relevant):
{json.dumps(optional_topics)}
- family_history: Only if patient mentions family conditions
- systems_review: Only if symptoms suggest multi-system issues
- gynae_sexual: Only if clearly relevant (pelvic pain, menopause, etc.)

CONVERSATION COMPLETE when:
- All required topics covered AND
- Patient said "no"/"nothing"/"that's all" to closing question

TOPIC COMPLETION:
- Mark topics in "topics_completed" when patient has given sufficient information
- Mark irrelevant optional topics in "optional_topics_to_skip"

PROGRESS:
- Questions: {progress['questions_asked']}/{progress['max_questions']}
- Required topics done: {progress['required_topics_completed']}/{progress['required_topics_total']}
- Topics completed: {self.state.topics_completed}

PACING:
{"- URGENT: Approaching question limit. Wrap up quickly. Focus only on critical missing info." if progress['questions_asked'] >= 25 else ""}
{"- Prioritize essential topics only. Be concise." if progress['questions_asked'] >= 20 else ""}

Generate your JSON response now."""

        return system_prompt

    def _handle_red_flag_detected(self, decision: Dict) -> Tuple[str, bool, Optional[str]]:
        category = decision.get('red_flag_category', 'unknown')
        red_flag_categories = self.checklist_template['checklist']['red_flags']['red_flag_categories']
        response_template_name = red_flag_categories.get(category, {}).get('response_template', 'medical_emergency_cardiac')
        response_data = self.checklist_template['red_flag_responses'].get(response_template_name, {})
        emergency_message = response_data.get('message', 'Please seek urgent medical attention.')
        self.state.record_red_flag(
            category=category,
            severity=red_flag_categories.get(category, {}).get('severity', 'critical'),
            action_taken="awaiting_response"
        )
        self.state.add_message(
            speaker="clara",
            text=emergency_message,
            topic="red_flag_response",
            flags=["red_flag", category]
        )
        return (emergency_message, False, None)

    def handle_red_flag_response(self, patient_response: str) -> Tuple[str, bool, str]:
        self.state.add_message(
            speaker="patient",
            text=patient_response,
            flags=["red_flag_response"]
        )

        response_lower = patient_response.lower()
        agreed_keywords = ['yes', 'okay', 'ok', 'will', "i'll", 'sure', 'agree']
        declined_keywords = ['no', "don't", "won't", 'wait', 'not', "can't"]

        agreed = any(keyword in response_lower for keyword in agreed_keywords)
        declined = any(keyword in response_lower for keyword in declined_keywords)

        if agreed and not declined:
            end_message = self.checklist_template['red_flag_responses']['end_after_accept']['message']
            end_message = end_message.replace('{doctor_name}', self.doctor_name)

            if self.state.red_flags_detected:
                self.state.red_flags_detected[-1].patient_response = "agreed"
                self.state.red_flags_detected[-1].action_taken = "ended_conversation"

            self.state.add_message(
                speaker="clara",
                text=end_message,
                flags=["red_flag_ending"]
            )

            self.state.end_conversation(status="ended_early_emergency")

            return (end_message, True, "emergency")

        else:
            continue_message = self.checklist_template['red_flag_responses']['continue_after_decline']['message']
            continue_message = continue_message.replace('{doctor_name}', self.doctor_name)

            if self.state.red_flags_detected:
                self.state.red_flags_detected[-1].patient_response = "declined"
                self.state.red_flags_detected[-1].action_taken = "continued_with_warning"

            self.state.add_message(
                speaker="clara",
                text=continue_message,
                flags=["red_flag_continue"]
            )

            decision = self._get_clara_decision()
            next_question = decision.get('next_question', 'Shall we continue?')
            next_topic = decision.get('current_topic', 'closing')

            self.state.add_message(
                speaker="clara",
                text=next_question,
                topic=next_topic
            )

            full_response = f"{continue_message}\n\n{next_question}"

            return (full_response, False, None)

    def _build_conversation_history(self) -> List[Dict[str, str]]:
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
        closing_script = f"""Thank you so much for taking the time to speak with me today, {self.patient_name}. Your responses will help {self.doctor_name} provide you with the best possible care.

If you feel there was anything you forgot to mention or would like to add, you can use the same link to speak with me again before your appointment.

Take care, and I hope your appointment goes well!"""
        return closing_script

    def get_conversation_summary(self) -> Dict[str, any]:
        return {
            "conversation_id": self.state.conversation_id,
            "patient_name": self.patient_name,
            "doctor_name": self.doctor_name,
            "status": self.state.status,
            "progress": self.state.get_progress_summary(),
            "red_flags": len(self.state.red_flags_detected),
            "transcript_length": len(self.state.messages)
        }

    def save_conversation(self, save_dir: Path = Path("data/conversations")):
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self.patient_name.replace(' ', '_')}_{self.state.conversation_id[:8]}.json"
        filepath = save_dir / filename
        self.state.save_to_file(filepath)
        return filepath