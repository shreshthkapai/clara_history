# test_openai.py

from services.azure_openai import AzureOpenAIService

def test_azure_openai():
    """Test Azure OpenAI connection and basic functionality"""
    
    print("🧪 Testing Azure OpenAI Service\n")
    
    service = AzureOpenAIService()
    print("✅ Service initialized\n")
    
    # Test 1: Basic conversation
    print("📝 Test 1: Basic Clara response")
    system_prompt = """You are Clara, a friendly medical history assistant. 
    Ask the patient what brings them in today."""
    
    conversation_history = []
    
    response = service.get_clara_response(
        conversation_history=conversation_history,
        system_prompt=system_prompt
    )
    
    print(f"   Clara: {response}\n")
    
    # Test 2: Follow-up question
    print("📝 Test 2: Follow-up question")
    conversation_history = [
        {"role": "assistant", "content": response},
        {"role": "user", "content": "I've been having chest pain for the past week"}
    ]
    
    system_prompt = """You are Clara, a medical history assistant.
    The patient mentioned chest pain. Ask a follow-up question about the character of the pain.
    Be empathetic but clinical."""
    
    response2 = service.get_clara_response(
        conversation_history=conversation_history,
        system_prompt=system_prompt
    )
    
    print(f"   Clara: {response2}\n")
    
    # Test 3: Red flag detection
    print("🚨 Test 3: Red flag detection")
    test_messages = [
        "I've been having crushing chest pain that radiates to my arm",
        "I've been feeling a bit tired lately",
        "I've been having thoughts of harming myself"
    ]
    
    from pathlib import Path
    import json
    with open(Path("data/checklist_template.json"), 'r') as f:
        template = json.load(f)
    
    red_flag_categories = template['checklist']['red_flags']['red_flag_categories']
    
    for msg in test_messages:
        print(f"   Testing: \"{msg}\"")
        result = service.detect_red_flags(msg, red_flag_categories)
        if result:
            print(f"   ⚠️  RED FLAG: {result['category']} ({result['severity']})")
        else:
            print(f"   ✓ No red flags")
        print()
    
    print("✅ All OpenAI tests completed!")

if __name__ == "__main__":
    test_azure_openai()