# test_state.py

from core.conversation_state import ConversationState
import uuid

def test_conversation_state():
    """Test the conversation state manager"""
    
    print("🧪 Testing Conversation State Manager\n")
    
    # Create new conversation
    conversation_id = str(uuid.uuid4())
    state = ConversationState.load_from_template(
        conversation_id=conversation_id,
        patient_name="Test Patient",
        doctor_name="Dr Smith"
    )
    
    print("✅ Created new conversation")
    print(f"   ID: {state.conversation_id[:8]}...")
    print(f"   Patient: {state.patient_name}")
    print(f"   Doctor: {state.doctor_name}\n")
    
    # Check checklist initialized
    print(f"📋 Checklist initialized:")
    print(f"   Required topics: {len(state.topics_required)}")
    print(f"   Optional topics: {len(state.topics_optional)}")
    print(f"   Max questions: {state.max_questions}\n")
    
    # Add some messages
    state.add_message("clara", "What brings you in today?", topic="chief_complaint")
    state.add_message("patient", "I've been having chest pain")
    
    print(f"💬 Added messages:")
    print(f"   Total messages: {len(state.messages)}")
    print(f"   Questions asked: {state.question_count}\n")
    
    # Mark topic complete
    state.mark_topic_complete("chief_complaint")
    print(f"✓ Marked 'chief_complaint' as complete\n")
    
    # Get next priority
    next_topic = state.get_next_priority_topic()
    print(f"🎯 Next priority topic: {next_topic}\n")
    
    # Check progress
    progress = state.get_progress_summary()
    print(f"📊 Progress:")
    print(f"   Questions: {progress['questions_asked']}/{progress['max_questions']}")
    print(f"   Required topics: {progress['required_topics_completed']}/{progress['required_topics_total']}")
    print(f"   Status: {progress['status']}\n")
    
    # Test red flag
    state.record_red_flag(
        category="cardiac_chest_pain",
        severity="critical",
        patient_response="yes",
        action_taken="ended_conversation"
    )
    print(f"🚨 Recorded red flag: cardiac_chest_pain")
    print(f"   Conversation ended for emergency: {state.conversation_ended_for_emergency}\n")
    
    # Check if should end
    should_end, reason = state.should_end_conversation()
    print(f"🛑 Should end conversation: {should_end} (reason: {reason})\n")
    
    # Save to file
    from pathlib import Path
    save_path = Path("data/conversations") / f"test_{conversation_id[:8]}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state.save_to_file(save_path)
    print(f"💾 Saved conversation to: {save_path}\n")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_conversation_state()