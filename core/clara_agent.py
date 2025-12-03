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

        # Check red flags
        red_flag_result = self._check_red_flags(patient_message)
        if red_flag_result:
            return red_flag_result

        # Update checklist
        self._update_checklist_from_response(patient_message)

        # Check if we should end BEFORE generating next question
        should_end, end_reason = self.state.should_end_conversation()

        if should_end:
            # Generate farewell and END
            closing_message = self._generate_closing_message()
            self.state.add_message(
                speaker="clara",
                text=closing_message,
                topic="closing"
            )
            self.state.end_conversation(status="completed")
            return (closing_message, True, "completed")

        # Generate next question
        clara_response = self._generate_next_question()
        next_topic = self.state.get_next_priority_topic()

        # CRITICAL: Detect if Clara generated a farewell message prematurely
        # If she did, mark closing complete and END next turn
        if next_topic == "closing" or next_topic is None:
            # Check if response contains farewell phrases
            farewell_indicators = [
                "you're all set",
                "you're good to go",
                "that's all for now",
                "we're all done",
                "that covers everything",
                "i'll make sure dr",
                "see you at your appointment",
                "dr. smith will see you soon"
            ]

            response_lower = clara_response.lower()
            contains_farewell = any(indicator in response_lower for indicator in farewell_indicators)

            if contains_farewell:
                # This is a farewell - mark closing complete NOW
                self.state.mark_topic_complete("closing")

                # Add the message
                self.state.add_message(
                    speaker="clara",
                    text=clara_response,
                    topic="closing"
                )

                # END immediately - don't wait for patient response
                self.state.end_conversation(status="completed")
                return (clara_response, True, "completed")

        # Normal flow - add message and continue
        self.state.add_message(
            speaker="clara",
            text=clara_response,
            topic=next_topic
        )

        return (clara_response, False, None)

    def _check_red_flags(self, patient_message: str) -> Optional[Tuple[str, bool, str]]:

        red_flag_categories = self.checklist_template['checklist']['red_flags']['red_flag_categories']

        detected = self.openai_service.detect_red_flags(patient_message, red_flag_categories)

        if not detected:
            return None

        category = detected['category']
        severity = detected['severity']

        response_template_name = red_flag_categories[category].get('response_template', 'medical_emergency_cardiac')
        response_data = self.checklist_template['red_flag_responses'].get(response_template_name, {})

        emergency_message = response_data.get('message', 'Please seek urgent medical attention.')

        self.state.record_red_flag(
            category=category,
            severity=severity,
            action_taken="awaiting_response"
        )

        self.state.add_message(
            speaker="clara",
            text=emergency_message,
            topic="red_flag_response",
            flags=["red_flag", category, severity]
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

            next_question = self._generate_next_question()
            next_topic = self.state.get_next_priority_topic()
            self.state.add_message(
                speaker="clara",
                text=next_question,
                topic=next_topic
            )

            full_response = f"{continue_message}\n\n{next_question}"

            return (full_response, False, None)

    def _update_checklist_from_response(self, patient_message: str):
        """Mark topics complete intelligently"""

        clara_messages = [msg for msg in self.state.messages if msg.speaker == "clara"]
        if not clara_messages:
            return

        last_topic = clara_messages[-1].topic

        if not last_topic or last_topic not in self.state.checklist:
            return

        # SPECIAL: Closing topic needs smart detection
        if last_topic == "closing":
            # Check if patient is saying "no/nothing" vs adding new info
            response_lower = patient_message.lower().strip()

            # Negative responses = they're done
            done_phrases = [
                "no", "nope", "nothing", "that's all", "that's everything",
                "no thanks", "i think that's it", "nothing else", "all good",
                "that covers it", "i'm good", "we're good", "nothing more"
            ]

            # If response is short AND contains done phrase, mark complete
            word_count = len(patient_message.split())
            is_done = any(phrase in response_lower for phrase in done_phrases)

            if is_done and word_count < 10:
                # They're done - mark closing complete
                self.state.mark_topic_complete("closing")
            else:
                # They added new info - DON'T mark complete yet
                # Clara will ask follow-ups, then ask closing again
                pass

            return

        # For other topics: if patient gave substantial response, mark complete
        if len(patient_message.strip().split()) > 3:  # More than 3 words
            self.state.mark_topic_complete(last_topic)

    def _generate_next_question(self) -> str:
        next_topic = self.state.get_next_priority_topic()

        if not next_topic:
            next_topic = "closing"

        topic_details = self.state.checklist.get(next_topic, {})
        topic_name = topic_details.get('name', next_topic)
        example_questions = topic_details.get('questions', [])

        system_prompt = self._build_system_prompt(next_topic, topic_name, example_questions)

        conversation_history = self._build_conversation_history()

        clara_response = self.openai_service.get_clara_response(
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300
        )

        return clara_response

    def _build_system_prompt(self, next_topic: str, topic_name: str, example_questions: List[str]) -> str:
        """Build the system prompt for generating next question"""

        # Get progress
        progress = self.state.get_progress_summary()
        incomplete_topics = self.state.get_incomplete_required_topics()

        system_prompt = f"""You are Clara, a medical history-taking assistant conducting a pre-consultation interview with {self.patient_name} before their appointment with {self.doctor_name}.

YOUR ROLE - READ CAREFULLY:
You are NOT a doctor. You are a medical history assistant collecting information systematically.
Your ONLY job is to ask clear, focused questions to gather medical history.

CRITICAL RULES:
1. NEVER provide medical advice, reassurance, diagnosis, or treatment suggestions
2. NEVER comment on whether symptoms are serious or not serious
3. NEVER suggest what might be causing their symptoms
4. NEVER say things like "that sounds concerning" or "that's reassuring"
5. DO NOT ask follow-up questions about topics already thoroughly covered
6. DO NOT ask about medications or past medical history unless that's the current topic
7. Stay focused on ONE topic at a time - don't jump around

WHAT YOU SHOULD DO:
- Ask ONE clear, direct question at a time
- Acknowledge what they said briefly ("Thank you for sharing that")

Example of GOOD question:
"When did the stomach pain first start?"

Example of BAD question (DO NOT DO THIS):
"I'm concerned about your stomach pain - have you thought about whether your cholesterol medication might be affecting your digestion?"

Generate your question now - ONE focused question about {topic_name}."""

        return system_prompt

    def _get_pacing_instructions(self, questions_asked: int, max_questions: int) -> str:
        remaining = max_questions - questions_asked

        if remaining <= 2:
            return """
URGENT - PACING WARNING:
You are about to reach the maximum question limit.
- Do NOT start any new major topics.
- Wrap up the current topic quickly.
- If you have critical missing information, ask ONE final combined question.
- Prepare to end the conversation naturally in the next turn.
"""
        elif remaining <= 5:
            return f"""
PACING WARNING:
You have {remaining} questions remaining before the limit.
- Prioritize only the most critical missing information.
- Be more concise.
- Combine related questions if possible.
- Start moving towards a natural conclusion.
"""
        return ""

    def _build_conversation_history(self) -> List[Dict[str, str]]:

        recent_messages = self.state.messages[-20:]

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