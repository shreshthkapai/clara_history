# test_clara_agent.py

from core.clara_agent import ClaraAgent
from pathlib import Path

def test_clara_agent():
    """Test the full Clara Agent conversation flow"""
    
    print("🧪 Testing Clara Agent\n")
    print("=" * 60)
    
    # Initialize Clara
    clara = ClaraAgent(
        patient_name="Test Patient",
        doctor_name="Dr Smith"
    )
    
    print("✅ Clara initialized")
    print(f"   Patient: {clara.patient_name}")
    print(f"   Doctor: {clara.doctor_name}")
    print(f"   Conversation ID: {clara.state.conversation_id[:8]}...\n")
    
    # Start conversation
    print("=" * 60)
    print("STARTING CONVERSATION")
    print("=" * 60)
    
    opening = clara.start_conversation()
    print(f"\n💬 CLARA:\n{opening}\n")
    
    # Simulate patient responses
    test_conversation = [
        {
            "patient": "I've been having bad headaches for about 2 weeks",
            "description": "Chief complaint"
        },
        {
            "patient": "They started suddenly about 2 weeks ago. Usually in the afternoon and they're quite painful, like a throbbing feeling",
            "description": "History of presenting complaint"
        },
        {
            "patient": "I think it might be stress from work. I'm worried it could be something serious though",
            "description": "ICE (Ideas, Concerns, Expectations)"
        },
        {
            "patient": "No chest pain or breathing problems. No weakness or anything like that",
            "description": "Red flag screening"
        },
        {
            "patient": "I have high blood pressure, been on medication for 5 years. No surgeries",
            "description": "Past medical history"
        },
        {
            "patient": "I take amlodipine 5mg daily and paracetamol when I get the headaches. No allergies",
            "description": "Medications"
        },
        {
            "patient": "I don't smoke, have maybe 2 glasses of wine on weekends. I work in IT, quite stressful",
            "description": "Social history"
        },
        {
            "patient": "No, I think we've covered everything. Just hoping to get some help with these headaches",
            "description": "Closing"
        }
    ]
    
    # Run through conversation
    for i, exchange in enumerate(test_conversation, 1):
        print("=" * 60)
        print(f"EXCHANGE {i}: {exchange['description']}")
        print("=" * 60)
        
        patient_msg = exchange['patient']
        print(f"\n💭 PATIENT:\n{patient_msg}\n")
        
        # Process response
        clara_response, should_end, end_reason = clara.process_patient_response(patient_msg)
        
        print(f"💬 CLARA:\n{clara_response}\n")
        
        if should_end:
            print(f"🛑 Conversation ended: {end_reason}")
            break
        
        # Show progress
        progress = clara.state.get_progress_summary()
        print(f"📊 Progress: {progress['questions_asked']}/{progress['max_questions']} questions | "
              f"{progress['required_topics_completed']}/{progress['required_topics_total']} required topics\n")
    
    # Final summary
    print("=" * 60)
    print("CONVERSATION COMPLETE")
    print("=" * 60)
    
    summary = clara.get_conversation_summary()
    print(f"\n📋 Final Summary:")
    print(f"   Status: {summary['status']}")
    print(f"   Total messages: {summary['transcript_length']}")
    print(f"   Questions asked: {summary['progress']['questions_asked']}")
    print(f"   Topics completed: {summary['progress']['required_topics_completed']}/{summary['progress']['required_topics_total']}")
    print(f"   Red flags: {summary['red_flags']}")
    
    # Save conversation
    print("\n💾 Saving conversation...")
    filepath = clara.save_conversation()
    print(f"   ✅ Saved to: {filepath}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)


def test_red_flag_scenario():
    """Test red flag detection and handling"""
    
    print("\n\n🚨 Testing Red Flag Scenario\n")
    print("=" * 60)
    
    clara = ClaraAgent(
        patient_name="Emergency Patient",
        doctor_name="Dr Johnson"
    )
    
    # Start
    opening = clara.start_conversation()
    print(f"💬 CLARA:\n{opening}\n")
    
    # Patient mentions red flag
    print("=" * 60)
    print("PATIENT MENTIONS CARDIAC SYMPTOMS")
    print("=" * 60)
    
    patient_msg = "I've been having crushing chest pain that spreads to my left arm"
    print(f"\n💭 PATIENT:\n{patient_msg}\n")
    
    clara_response, should_end, end_reason = clara.process_patient_response(patient_msg)
    print(f"💬 CLARA (Emergency Response):\n{clara_response}\n")
    
    print(f"🚨 Red flags detected: {len(clara.state.red_flags_detected)}")
    if clara.state.red_flags_detected:
        flag = clara.state.red_flags_detected[0]
        print(f"   Category: {flag.category}")
        print(f"   Severity: {flag.severity}")
    
    # Patient agrees to get help
    print("\n" + "=" * 60)
    print("PATIENT AGREES TO SEEK HELP")
    print("=" * 60)
    
    patient_response = "Yes, I'll call 999 right now"
    print(f"\n💭 PATIENT:\n{patient_response}\n")
    
    final_msg, should_end, end_reason = clara.handle_red_flag_response(patient_response)
    print(f"💬 CLARA:\n{final_msg}\n")
    
    print(f"🛑 Conversation ended: {should_end} (reason: {end_reason})")
    print(f"📊 Status: {clara.state.status}")
    
    # Save
    filepath = clara.save_conversation()
    print(f"\n💾 Saved to: {filepath}")
    
    print("\n✅ Red flag test complete!")


if __name__ == "__main__":
    # Run normal conversation test
    test_clara_agent()
    
    # Run red flag test
    test_red_flag_scenario()