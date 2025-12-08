from openai import AzureOpenAI
from typing import List, Dict, Optional
from config.settings import settings
import json


class AzureOpenAIService:
    
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME

    def get_clara_response(
        self,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Standard LLM call - returns text response
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ Error calling Azure OpenAI: {e}")
            return "I apologize, but I'm having trouble processing your response. Could you please try again?"

    def get_clara_decision_json(
        self,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict:
        """
        Smart LLM call - returns structured JSON decision
        
        Handles:
        - Red flag detection
        - Next question generation
        - Topic completion
        - Conversation end detection
        - Optional topic relevance
        
        Returns dictionary with decision data
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                decision = json.loads(result_text)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing error: {e}")
                print(f"Raw response: {result_text}")
                
                # Fallback - extract what we can
                decision = self._parse_fallback_response(result_text)
            
            # Validate and set defaults
            decision = self._validate_decision(decision)
            
            return decision
        
        except Exception as e:
            print(f"❌ Error calling Azure OpenAI: {e}")
            
            # Emergency fallback
            return {
                "red_flag_detected": False,
                "red_flag_category": None,
                "conversation_complete": False,
                "topics_completed": [],
                "optional_topics_to_skip": [],
                "current_topic": "closing",
                "next_question": "Is there anything else you'd like to share?"
            }

    def _parse_fallback_response(self, text: str) -> Dict:
        """
        Fallback parser if JSON parsing fails
        Extracts key information from malformed response
        """
        decision = {
            "red_flag_detected": False,
            "red_flag_category": None,
            "conversation_complete": False,
            "topics_completed": [],
            "optional_topics_to_skip": [],
            "current_topic": "closing",
            "next_question": "Could you tell me more?"
        }
        
        # Try to extract question from text
        if "next_question" in text.lower():
            try:
                # Look for common patterns
                import re
                question_match = re.search(r'"next_question":\s*"([^"]+)"', text)
                if question_match:
                    decision["next_question"] = question_match.group(1)
            except:
                pass
        else:
            # Use the whole text as question if it looks like a question
            if "?" in text:
                decision["next_question"] = text
        
        # Check for red flag indicators
        if "red_flag" in text.lower() and "true" in text.lower():
            decision["red_flag_detected"] = True
        
        return decision

    def _validate_decision(self, decision: Dict) -> Dict:
        """
        Validate decision has all required fields with correct types
        """
        defaults = {
            "red_flag_detected": False,
            "red_flag_category": None,
            "conversation_complete": False,
            "topics_completed": [],
            "optional_topics_to_skip": [],
            "current_topic": "closing",
            "next_question": "Is there anything else?"
        }
        
        # Ensure all fields exist
        for key, default_value in defaults.items():
            if key not in decision:
                decision[key] = default_value
        
        # Type validation
        if not isinstance(decision["red_flag_detected"], bool):
            decision["red_flag_detected"] = False
        
        if not isinstance(decision["conversation_complete"], bool):
            decision["conversation_complete"] = False
        
        if not isinstance(decision["topics_completed"], list):
            decision["topics_completed"] = []
        
        if not isinstance(decision["optional_topics_to_skip"], list):
            decision["optional_topics_to_skip"] = []
        
        if not isinstance(decision["next_question"], str):
            decision["next_question"] = "Could you tell me more?"
        
        return decision

    # ==========================================
    # LEGACY METHODS - Keep for summaries
    # ==========================================
        
    def generate_summary(
        self,
        transcript: List[Dict[str, str]],
        summary_type: str = "short"
    ) -> str:
        """Generate summary from transcript (used by summary_generator.py)"""
        
        transcript_text = "\n".join([
            f"{msg['speaker'].upper()}: {msg['text']}"
            for msg in transcript
        ])
        
        if summary_type == "short":
            system_prompt = """You are a medical documentation assistant. 
                                Generate a brief 30-second read summary (2-3 sentences) of this patient conversation.
                                Focus on: primary concern, key symptoms, duration, and what the patient hopes to achieve. 
                                Keep it concise and clinical."""
        
        else: 
            system_prompt = """You are a medical documentation assistant.
                                Generate a structured clinical summary with these sections:
                                - Primary Concern: One line describing main reason for visit
                                - Key Symptoms: 2-4 short symptom tags
                                - Duration: How long symptoms have been present
                                - Current Medications: List medications mentioned
                                - Relevant Medical History: Previous conditions, surgeries, similar issues
                                - Ideas, Concerns, Expectations (ICE): What patient thinks/worries/hopes
                                - Red Flags: Any concerning symptoms mentioned or explicitly denied
                                - Social Context: Brief relevant lifestyle factors
                                Be thorough but concise. Use clinical language."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize this patient conversation:\n\n{transcript_text}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return "Error generating summary."

    def generate_prep_items(
        self,
        transcript: List[Dict[str, str]]
    ) -> List[str]:
        """Generate preparation items for GP (used by summary_generator.py)"""
        
        transcript_text = "\n".join([
            f"{msg['speaker'].upper()}: {msg['text']}"
            for msg in transcript
        ])

        system_prompt = """You are a medical preparation assistant. Based on this patient conversation, generate a list of 2-5 specific items the GP should prepare or have ready for the appointment.
                            Examples:
                                    - "Recent blood pressure readings"
                                    - "ECG results from last visit"
                                    - "Current medication list"
                                    - "Blood glucose monitoring log"
                                    - "Previous imaging reports"
                            Respond with ONLY a comma-separated list, no numbering or bullets."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Based on this conversation, what should the GP prepare?\n\n{transcript_text}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.4,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            items = [item.strip() for item in result.split(',')]
            return items
        
        except Exception as e:
            print(f"❌ Error generating prep items: {e}")
            return []
    
    def generate_probable_conditions(
        self,
        transcript: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Generate probable conditions (used by summary_generator.py)"""
        
        transcript_text = "\n".join([
            f"{msg['speaker'].upper()}: {msg['text']}"
            for msg in transcript
        ])

        system_prompt = """You are a clinical reasoning assistant helping a GP prepare for a consultation. Based on the patient's presentation, suggest 2-4 probable conditions that should be considered.
                            For each condition, provide:
                                                        1. Condition name
                                                        2. Brief rationale (which symptoms/factors support this)

                            Format your response as:    
                                                    CONDITION: [name]
                                                    RATIONALE: [1-2 sentences]
                                                    
                                                    CONDITION: [name]
                                                    RATIONALE: [1-2 sentences]
                            
                            Remember: This is clinical decision support for the GP, not a diagnosis. Focus on common presentations first."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"What conditions should the GP consider based on this presentation?\n\n{transcript_text}"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.5,
                max_tokens=600
            )
            
            result = response.choices[0].message.content.strip()
            
            conditions = []
            lines = result.split('\n')
            current_condition = None
            current_rationale = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('CONDITION:'):
                    if current_condition and current_rationale:
                        conditions.append({
                            'condition': current_condition,
                            'rationale': current_rationale
                        })
                    current_condition = line.replace('CONDITION:', '').strip()
                    current_rationale = None
                elif line.startswith('RATIONALE:'):
                    current_rationale = line.replace('RATIONALE:', '').strip()
            
            if current_condition and current_rationale:
                conditions.append({
                    'condition': current_condition,
                    'rationale': current_rationale
                })
            
            return conditions
        
        except Exception as e:
            print(f"❌ Error generating probable conditions: {e}")
            return []