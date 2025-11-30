
import sys
import os
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

from core.clara_agent import ClaraAgent
from core.conversation_state import ConversationState

# Mock AzureOpenAIService
class MockAzureOpenAIService:
    def get_clara_response(self, conversation_history, system_prompt, temperature, max_tokens):
        return "Mock question from Clara?"

    def detect_red_flags(self, message, categories):
        return None

# Patch the service in clara_agent module
import core.clara_agent
core.clara_agent.AzureOpenAIService = MockAzureOpenAIService

def run_reproduction():
    print("Starting reproduction...")
    agent = ClaraAgent(patient_name="Test Patient", doctor_name="Dr. Test")
    
    # Start conversation
    response = agent.start_conversation()
    print(f"Start: {response}")
    
    # Simulate 20 turns
    for i in range(1, 25):
        print(f"\n--- Turn {i} ---")
        patient_response = f"Answer to question {i}"
        
        response, is_ending, end_reason = agent.process_patient_response(patient_response)
        
        print(f"Clara: {response}")
        print(f"Is Ending: {is_ending}")
        print(f"End Reason: {end_reason}")
        print(f"Question Count: {agent.state.question_count}")
        
        if is_ending:
            print("Conversation ended.")
            break

if __name__ == "__main__":
    run_reproduction()
