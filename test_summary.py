# test_summary.py

from core.clara_agent import ClaraAgent
from core.summary_generator import SummaryGenerator
import json

def test_summary_generation():
    """Test generating all 5 outputs from a conversation"""
    
    print("🧪 Testing Summary Generator\n")
    print("=" * 60)
    
    # Create a test conversation
    clara = ClaraAgent(
        patient_name="Test Patient",
        doctor_name="Dr Smith"
    )
    
    # Start conversation
    clara.start_conversation()
    
    # Simulate a conversation
    test_exchanges = [
        "I've been having headaches for 2 weeks",
        "They started suddenly. Usually afternoon, throbbing pain around temples",
        "I think it's stress from work. Worried it could be something serious",
        "No chest pain or weakness. No vision problems",
        "I have high blood pressure on amlodipine 5mg. No surgeries",
        "Just the amlodipine and paracetamol for headaches. No allergies",
        "Don't smoke, 2 glasses wine on weekends. Work in IT, quite stressful",
        "No, I think we've covered everything"
    ]
    
    for exchange in test_exchanges:
        clara.process_patient_response(exchange)
    
    # End conversation
    clara.state.end_conversation(status="completed")
    
    print("✅ Test conversation completed")
    print(f"   Messages: {len(clara.state.messages)}")
    print(f"   Topics covered: {len(clara.state.topics_completed)}\n")
    
    # Generate summaries
    print("=" * 60)
    print("GENERATING SUMMARIES")
    print("=" * 60)
    print()
    
    generator = SummaryGenerator()
    outputs = generator.generate_all_outputs(clara.state)
    
    # Display outputs
    print("\n" + "=" * 60)
    print("OUTPUT 1: FULL TRANSCRIPT")
    print("=" * 60)
    print(f"Total messages: {len(outputs['full_transcript'])}")
    print(f"First message: {outputs['full_transcript'][0]['text'][:100]}...")
    
    print("\n" + "=" * 60)
    print("OUTPUT 2: SHORT SUMMARY (~30 sec read)")
    print("=" * 60)
    print(outputs['short_summary'])
    
    print("\n" + "=" * 60)
    print("OUTPUT 3: LONG SUMMARY (Detailed)")
    print("=" * 60)
    print(outputs['long_summary'])
    
    print("\n" + "=" * 60)
    print("OUTPUT 4: WHAT TO PREPARE")
    print("=" * 60)
    for i, item in enumerate(outputs['what_to_prepare'], 1):
        print(f"{i}. {item}")
    
    print("\n" + "=" * 60)
    print("OUTPUT 5: PROBABLE CONDITIONS (GP Only)")
    print("=" * 60)
    for i, condition in enumerate(outputs['probable_conditions'], 1):
        print(f"\n{i}. {condition['condition']}")
        print(f"   Rationale: {condition['rationale']}")
    
    print("\n" + "=" * 60)
    print("METADATA")
    print("=" * 60)
    print(f"Duration: {outputs['conversation_stats']['duration_minutes']} minutes")
    print(f"Questions asked: {outputs['conversation_stats']['questions_asked']}")
    print(f"Topics covered: {outputs['conversation_stats']['topics_covered']}")
    print(f"Red flags: {len(outputs['red_flags'])}")
    
    # Save outputs
    print("\n" + "=" * 60)
    filepath = generator.save_outputs(outputs)
    print(f"✅ All outputs saved!")
    
    print("\n✅ SUMMARY GENERATOR TEST COMPLETE!")


if __name__ == "__main__":
    test_summary_generation()