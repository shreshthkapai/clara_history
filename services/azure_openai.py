from openai import AzureOpenAI
from typing import List, Dict, Optional
from config.settings import settings

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
        
    def generate_summary(
        self,
        transcript: List[Dict[str, str]],
        summary_type: str = "short"
    ) -> str:
        
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
                temperature=0.3,  # Lower temperature for more factual summaries
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return "Error generating summary."
        
    def detect_red_flags(
        self,
        patient_message: str,
        red_flag_categories: Dict[str, Dict]
    ) -> Optional[Dict[str, str]]:
    
        red_flag_list = []
        for category, details in red_flag_categories.items():
            triggers = details.get('triggers', [])
            severity = details.get('severity', 'unknown')
            red_flag_list.append(f"- {category} ({severity}): {', '.join(triggers)}")
        
        red_flag_text = "\n".join(red_flag_list)

        system_prompt = f"""You are a medical safety assistant. Analyze patient statements for emergency red flags.
                            RED FLAG CATEGORIES TO CHECK: {red_flag_text}

                            If the patient's message contains ANY of these red flag symptoms, respond with:
                            RED_FLAG_DETECTED: [category_name]

                            If no red flags are present, respond with: NO_RED_FLAG
                            Be cautious - if unsure, err on the side of flagging for safety."""
                    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Patient said: \"{patient_message}\""}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.2,  # Very low - we want consistent detection
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            
            if result.startswith("RED_FLAG_DETECTED:"):
                category = result.split(":")[1].strip()
                if category in red_flag_categories:
                    return {
                        "category": category,
                        "severity": red_flag_categories[category].get('severity', 'unknown')
                    }
            
            return None
        
        except Exception as e:
            print(f"❌ Error detecting red flags: {e}")
            return None

    def detect_relevant_topics(
        self,
        patient_message: str,
        optional_topics: List[str]
    ) -> Optional[str]:
        
        if not optional_topics:
            return None
            
        topics_str = ", ".join(optional_topics)
        
        system_prompt = f"""You are a medical conversation analyst. 
                            Analyze the patient's message and decide if we MUST ask about any of these specific topics: [{topics_str}].
                            
                            Rules:
                            1. Only return a topic if the patient explicitly mentions something relevant to it.
                            2. Example: "My dad had heart issues" -> Return "family_history"
                            3. Example: "I smoke 20 a day" -> Return "social_history"
                            4. If nothing is relevant, return "NONE".
                            5. Return ONLY the exact topic name from the list.
                            """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Patient said: \"{patient_message}\""}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            
            if result in optional_topics:
                return result
            
            return None
            
        except Exception as e:
            print(f"❌ Error detecting topics: {e}")
            return None
        
    def generate_prep_items(
        self,
        transcript: List[Dict[str, str]]
    ) -> List[str]:
        
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
