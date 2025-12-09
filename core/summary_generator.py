# core/summary_generator.py

from typing import Dict, List, Any
from services.azure_openai import AzureOpenAIService
from core.conversation_state import ConversationState
from pathlib import Path
import json


class SummaryGenerator:
    """
    Generates all 5 outputs from a completed conversation:
    1. Full transcript (already in conversation state)
    2. Short summary (~30 second read)
    3. Long summary (detailed clinical)
    4. What to prepare (actionable items for GP)
    5. Probable conditions (differential diagnoses - GP only)
    """
    
    def __init__(self):
        self.openai_service = AzureOpenAIService()
    
    
    def generate_all_outputs(self, conversation_state: ConversationState) -> Dict[str, Any]:
        """
        Generate all 5 outputs from conversation
        
        Args:
            conversation_state: Completed conversation state
        
        Returns:
            Dictionary with all outputs
        """
        
        print("📊 Generating summaries...")
        
        # Get transcript
        transcript = conversation_state.get_transcript()
        
        # Generate each output
        outputs = {
            "conversation_id": conversation_state.conversation_id,
            "patient_name": conversation_state.patient_name,
            "doctor_name": conversation_state.doctor_name,
            "appointment_id": conversation_state.appointment_id,
            "completed_at": conversation_state.ended_at.isoformat() if conversation_state.ended_at else None,
            
            # Output 1: Full transcript (already available)
            "full_transcript": transcript,
            
            # Output 2: Short summary
            "short_summary": self._generate_short_summary(transcript),
            
            # Output 3: Long summary
            "long_summary": self._generate_long_summary(transcript),
            
            # Output 4: What to prepare
            "what_to_prepare": self._generate_prep_items(transcript),
            
            # Output 5: Probable conditions
            "probable_conditions": self._generate_probable_conditions(transcript),
            
            # Conversation stats
            "conversation_stats": {
                "total_messages": len(transcript),
                "questions_asked": conversation_state.question_count,
                "topics_covered": len(conversation_state.topics_completed),
                "duration_minutes": self._calculate_duration(conversation_state)
            }
        }
        
        print("✅ All summaries generated!")
        
        return outputs
    
    
    def _generate_short_summary(self, transcript: List[Dict]) -> str:
        """
        Generate brief 30-second read summary
        
        Args:
            transcript: Full conversation transcript
        
        Returns:
            Short summary text (2-3 sentences)
        """
        
        print("  ⏩ Generating short summary...")
        
        # Build transcript text
        transcript_text = "\n".join([
            f"{msg['speaker'].upper()}: {msg['text']}"
            for msg in transcript
        ])
        
        system_prompt = """You are a UK GP writing clinical handover notes. Write a tight 2-3 sentence summary using standard UK medical shorthand.

**KEY RULES:**
- Use UK notation: X/7 (days), X/52 (weeks), X/12 (months)
- Only include age/sex if explicitly stated - DO NOT INFER
- If duration uncertain, document the uncertainty (e.g., "?2-4/52")
- Use standard descriptors: intermittent, constant, progressive, etc.

Write as you would for a colleague in real clinical practice - efficient, factual, no waffle."""
        
        try:
            from openai import AzureOpenAI
            from config.settings import settings
            
            client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
            
            response = client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript_text}
                ],
                temperature=0.2,
                max_tokens=150
            )
            
            summary = response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"  ❌ Error generating short summary: {e}")
            summary = "Error generating summary."
        
        return summary
    
    
    def _generate_long_summary(self, transcript: List[Dict]) -> str:
        """
        Generate detailed structured clinical summary
        
        Args:
            transcript: Full conversation transcript
        
        Returns:
            Long summary with sections
        """
        
        print("  📄 Generating long summary...")
        
        # Build transcript text
        transcript_text = "\n".join([
            f"{msg['speaker'].upper()}: {msg['text']}"
            for msg in transcript
        ])
        
        system_prompt = """You are a UK GP writing clinical notes for a colleague. Write in tight, efficient doctor-to-doctor style using standard UK medical shorthand.

**KEY RULES:**
- Only document information explicitly mentioned - DO NOT INFER age, sex, or details
- Use UK notation: X/7 (days), X/52 (weeks), X/12 (months)
- Document uncertainty when present (e.g., "Onset: ?2-4/52")
- Use standard abbreviations: HPC, PMHx, FHx, SHx, CP, SOB, etc.

**Structure:**
- PC: [one line - duration + pattern + complaint]
- HPC: [SOCRATES format - tight bullets or brief sentences]
- ICE: [what they think/worry/want]
- PMHx: [list format]
- Medications: [list with doses if given, note allergies]
- FHx: [relevant conditions]
- SHx: [smoking/alcohol/occupation/living]
- RFs: [red flags or relevant negatives]

Write as you would in real clinical practice - concise, factual, no waffle."""
        
        try:
            from openai import AzureOpenAI
            from config.settings import settings
            
            client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
            
            response = client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate clinical notes:\n\n{transcript_text}"}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"  ❌ Error generating long summary: {e}")
            summary = "Error generating detailed summary."
        
        return summary
    
    
    def _generate_prep_items(self, transcript: List[Dict]) -> List[str]:
        """
        Generate "what to prepare" items for GP
        
        Args:
            transcript: Full conversation transcript
        
        Returns:
            List of preparation items
        """
        
        print("  📋 Generating preparation items...")
        
        # Use existing method from azure_openai service
        prep_items = self.openai_service.generate_prep_items(transcript)
        
        # Ensure we have at least some defaults if nothing generated
        if not prep_items or len(prep_items) == 0:
            prep_items = ["Recent vital signs", "Current medication list"]
        
        return prep_items
    
    
    def _generate_probable_conditions(self, transcript: List[Dict]) -> List[Dict[str, str]]:
        """
        Generate probable conditions (differential diagnoses)
        IMPORTANT: For GP eyes only, never shown to patient
        
        Args:
            transcript: Full conversation transcript
        
        Returns:
            List of conditions with rationale
        """
        
        print("  🔍 Generating probable conditions...")
        
        # Use existing method from azure_openai service
        conditions = self.openai_service.generate_probable_conditions(transcript)
        
        # Ensure we have at least one if nothing generated
        if not conditions or len(conditions) == 0:
            conditions = [{
                "condition": "Further assessment needed",
                "rationale": "Insufficient information to suggest specific differential diagnoses. Recommend comprehensive clinical examination."
            }]
        
        return conditions
    
    
    def _calculate_duration(self, conversation_state: ConversationState) -> float:
        """Calculate conversation duration in minutes"""
        if conversation_state.ended_at and conversation_state.started_at:
            duration = (conversation_state.ended_at - conversation_state.started_at).total_seconds() / 60
            return round(duration, 1)
        return 0.0
    
    
    def save_outputs(self, outputs: Dict[str, Any], save_dir: Path = Path("data/conversations")) -> Path:
        """
        Save all outputs to JSON file with standard naming format
        
        Args:
            outputs: Dictionary with all 5 outputs
            save_dir: Directory to save to
        
        Returns:
            Path to saved file
        """
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard filename format: SUMMARY_YYYYMMDD_HHMMSS_PatientName.json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_name = outputs.get('patient_name', 'Unknown').replace(' ', '_')
        
        # Standard format: SUMMARY_20251130_143045_John_Smith.json
        filename = f"SUMMARY_{timestamp}_{patient_name}.json"
        filepath = save_dir / filename
        
        # Save
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Summary saved: {filename}")
        
        return filepath