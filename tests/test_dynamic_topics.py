import sys
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.clara_agent import ClaraAgent
from core.conversation_state import ConversationState

class TestDynamicTopicActivation(unittest.TestCase):
    def setUp(self):
        # Mock AzureOpenAIService to avoid real API calls
        self.mock_openai_patcher = patch('core.clara_agent.AzureOpenAIService')
        self.mock_openai_class = self.mock_openai_patcher.start()
        self.mock_openai_service = self.mock_openai_class.return_value
        
        # Initialize agent
        self.agent = ClaraAgent(patient_name="Test Patient", doctor_name="Dr. Smith")
        
        # Ensure family_history is optional initially
        self.assertIn("family_history", self.agent.state.topics_optional)
        self.assertNotIn("family_history", self.agent.state.topics_required)

    def tearDown(self):
        self.mock_openai_patcher.stop()

    def test_activate_topic_positive(self):
        """Test that a relevant topic is activated when detected"""
        
        # Mock detect_relevant_topics to return 'family_history'
        self.mock_openai_service.detect_relevant_topics.return_value = "family_history"
        self.mock_openai_service.get_clara_response.return_value = "Next question?"
        self.mock_openai_service.detect_red_flags.return_value = None
        
        patient_message = "My father had a heart attack at 50."
        
        # Process response
        self.agent.process_patient_response(patient_message)
        
        # Verify detect_relevant_topics was called
        self.mock_openai_service.detect_relevant_topics.assert_called_once()
        
        # Verify family_history is now required
        self.assertIn("family_history", self.agent.state.topics_required)
        self.assertNotIn("family_history", self.agent.state.topics_optional)
        self.assertTrue(self.agent.state.checklist['family_history']['required'])
        self.assertTrue(self.agent.state.checklist['family_history'].get('dynamically_activated', False))
        
        print("\n✅ Positive test passed: 'family_history' activated successfully.")

    def test_activate_topic_negative(self):
        """Test that no topic is activated when none are relevant"""
        
        # Mock detect_relevant_topics to return None
        self.mock_openai_service.detect_relevant_topics.return_value = None
        self.mock_openai_service.get_clara_response.return_value = "Next question?"
        self.mock_openai_service.detect_red_flags.return_value = None
        
        patient_message = "I have a headache."
        
        # Process response
        self.agent.process_patient_response(patient_message)
        
        # Verify detect_relevant_topics was called
        self.mock_openai_service.detect_relevant_topics.assert_called_once()
        
        # Verify family_history is STILL optional
        self.assertIn("family_history", self.agent.state.topics_optional)
        self.assertNotIn("family_history", self.agent.state.topics_required)
        
        print("\n✅ Negative test passed: No topic activated for irrelevant input.")

if __name__ == '__main__':
    unittest.main()
