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
        
        return decision

    def _validate_decision(self, decision: Dict) -> Dict:
        """
        Validate decision has all required fields with correct types
        """
        defaults = {
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
    # SUMMARY GENERATION METHODS
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
            system_prompt = """You are writing doctor-to-doctor handover notes. Be HYPER EFFICIENT.

Generate a 2-3 sentence summary ONLY. Use clinical shorthand.

Examples of GOOD style:
- "32F, 3/7 history lower abdominal pain, worse on movement. Denies fever, vomiting. Regular cycles."
- "58M, exertional chest tightness x 2/52. FHx IHD (father MI age 50). Smoker 20/day."
- "45F, worsening fatigue x 6/12. Known hypothyroid, compliant with levothyroxine. Recent stressors at work."

NO filler words. NO narrative prose. Just facts.

Format: [Age/Sex if mentioned], [duration] history of [chief complaint], [key modifiers]. [Critical context]."""
        
        else: 
            system_prompt = """You are writing clinical notes for a GP. Write in TIGHT, EFFICIENT doctor-to-doctor style.

NO waffle. NO filler. NO flowery language. Just clinical facts.

Use abbreviations: HPC, PMHx, FHx, SHx, BP, DM, IHD, etc.
Use shorthand: 3/7 (3 days), 2/52 (2 weeks), 6/12 (6 months)
Be concise: "No CP, SOB" not "The patient denies chest pain or shortness of breath"

Structure:
- PC: [one line]
- HPC: [tight bullets or abbreviated sentences with SOCRATES]
- ICE: [what they think/worry/want]
- PMHx: [list format]
- Meds: [list with doses]
- FHx: [relevant conditions]
- SHx: [smoking/alcohol/occupation/living]
- RFs: [red flags or relevant negatives]

This is CLINICAL DOCUMENTATION, not a story."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate clinical notes:\n\n{transcript_text}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.2,
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

        system_prompt = """You are writing prep items for a GP. Be TIGHT and SPECIFIC.

Examples of GOOD prep items:
- "Recent BP readings (last 3/12)"
- "HbA1c from Aug 2024"
- "ECG - query previous abnormalities"
- "Chest X-ray report 2023"
- "Current repeat prescriptions list"

NOT vague items like "medical history" or "test results"

Respond with ONLY a comma-separated list. No bullets. No numbering."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Based on this conversation, what should the GP prepare?\n\n{transcript_text}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.3,
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

        system_prompt = """You are a GP generating a differential diagnosis list. Write TIGHT clinical reasoning.

For each condition (2-4 max), provide:

CONDITION: [name]
RATIONALE: [Brief - why this fits. Use clinical shorthand.]

Examples of GOOD style:
CONDITION: Acute coronary syndrome
RATIONALE: Central CP radiating to L arm, exertional, FHx IHD. Risk factors: smoker, HTN.

CONDITION: Costochondritis  
RATIONALE: Sharp, localized, reproducible on palpation. No radiation. No cardiac RFs.

CONDITION: Iron deficiency anaemia
RATIONALE: 6/12 fatigue + menorrhagia. Likely cause of symptoms.

NO waffle. Just facts supporting each DDx. Think common things common."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"What conditions should the GP consider based on this presentation?\n\n{transcript_text}"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.4,
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