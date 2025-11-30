import streamlit as st
from typing import Optional, Tuple
from services.azure_speech import AzureSpeechService
from audio_recorder_streamlit import audio_recorder
import io


def initialize_chat_session():
    """Initialize session state for chat"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = True  # Always start with voice ON
    
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    
    if 'awaiting_red_flag_response' not in st.session_state:
        st.session_state.awaiting_red_flag_response = False
    
    if 'conversation_ended' not in st.session_state:
        st.session_state.conversation_ended = False


def render_chat_header():
    """Render the chat header with Clara branding"""
    st.title("🩺 Clara - Pre-Consultation Assistant")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # TTS toggle - MORE PROMINENT
        st.session_state.tts_enabled = st.toggle(
            "🔊 Clara's Voice",
            value=st.session_state.tts_enabled,
            help="Clara speaks her questions (recommended for accessibility)"
        )
        
        if st.session_state.tts_enabled:
            st.success("✓ Voice enabled")
        else:
            st.info("Voice is muted")
        
        st.divider()
        
        # Info
        st.info("💡 Clara will ask you questions to help your doctor provide better care.")
        
        st.caption("You can type or use voice to respond.")


def display_chat_history():
    """Display all messages in the chat"""
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.write(content)
            
            # Only auto-play if TTS is CURRENTLY enabled (respect the toggle)
            is_latest_clara = (role == "assistant" and i == len(st.session_state.messages) - 1)
            
            if role == "assistant" and "audio" in message and message["audio"] and st.session_state.tts_enabled:  # ← ADD CHECK HERE
                import base64
                audio_base64 = base64.b64encode(message["audio"]).decode()
                
                if is_latest_clara:
                    # Auto-play only the newest message
                    audio_id = f"clara_audio_{i}"
                    audio_html = f"""
                    <audio id="{audio_id}" autoplay>
                        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                    </audio>
                    <script>
                        document.getElementById('{audio_id}').play().catch(e => console.log("Autoplay blocked"));
                    </script>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    # Older messages - show playable audio (but don't autoplay)
                    st.audio(message["audio"], format="audio/wav")

def add_message(role: str, content: str, audio: Optional[bytes] = None):

    message = {
        "role": role,
        "content": content
    }
    
    if audio:
        message["audio"] = audio
    
    st.session_state.messages.append(message)


def get_clara_response_with_audio(text: str, speech_service: AzureSpeechService) -> Tuple[str, Optional[bytes]]:
    audio_bytes = None
    
    if st.session_state.tts_enabled:
        audio_bytes = speech_service.text_to_speech(text)
    
    return text, audio_bytes


def render_patient_input(speech_service: AzureSpeechService) -> Optional[str]:
    """
    Render input area for patient (text + microphone)
    """
    
    # Don't show input if conversation ended
    if st.session_state.conversation_ended:
        return None
    
    # Text input
    text_input = st.chat_input(
        "Type your response... (or use 🎤 in sidebar)",
        key="patient_text_input"
    )
    
    if text_input:
        return text_input
    
    # Microphone in sidebar - SIMPLE APPROACH
    with st.sidebar:
        st.divider()
        st.subheader("🎤 Voice Input")
        
        st.info("Click the button below and speak. It will stop automatically after you finish speaking.")
        
        if st.button("🎤 Press & Speak", type="primary", use_container_width=True, key=f"voice_btn_{len(st.session_state.messages)}"):
            with st.spinner("🎤 Listening... Speak now!"):
                recognized_text = speech_service.speech_to_text_from_mic()
                
                if recognized_text:
                    # Store in session state
                    st.session_state.pending_voice_input = recognized_text
                    st.success(f"✅ You said: {recognized_text}")
                    st.rerun()
                else:
                    st.error("Couldn't hear you clearly. Please try again.")
    
    # Check for pending voice input
    if 'pending_voice_input' in st.session_state:
        voice_text = st.session_state.pending_voice_input
        del st.session_state.pending_voice_input
        return voice_text
    
    return None

def show_conversation_ended_message():

    st.success("✅ **Conversation Complete**")
    st.info("""
    Thank you for completing the pre-consultation questions!
    
    Your responses have been saved and will be reviewed by your doctor before your appointment.
    
    If you need to add anything, you can use the same link again to continue the conversation.
    """)


def show_red_flag_warning():
    st.warning("⚠️ **IMPORTANT HEALTH INFORMATION ABOVE** - Please read Clara's message carefully.")


def render_progress_indicator(clara_agent):
    with st.sidebar:
        st.divider()
        st.subheader("Progress")
        
        progress = clara_agent.state.get_progress_summary()
        
        questions_progress = progress['questions_asked'] / progress['max_questions']
        st.progress(questions_progress, text=f"Questions: {progress['questions_asked']}/{progress['max_questions']}")
        
        topics_progress = progress['required_topics_completed'] / progress['required_topics_total']
        st.progress(topics_progress, text=f"Topics: {progress['required_topics_completed']}/{progress['required_topics_total']}")
        
        if progress['red_flags_detected'] > 0:
            st.error(f"🚨 Red flags detected: {progress['red_flags_detected']}")


def render_chat_interface(clara_agent, speech_service: AzureSpeechService):
   
    initialize_chat_session()

    render_chat_header()
    
    if not st.session_state.conversation_started:
        opening_message = clara_agent.start_conversation()
        
        audio_bytes = None
        if st.session_state.tts_enabled:
            audio_bytes = speech_service.text_to_speech(opening_message)
        
        add_message("assistant", opening_message, audio_bytes)
        st.session_state.conversation_started = True
        st.rerun()
    
    display_chat_history()
    
    render_progress_indicator(clara_agent)
    
    if st.session_state.conversation_ended:
        show_conversation_ended_message()
        return
    
    user_input = render_patient_input(speech_service)
    
    if user_input:
        # Add user message to chat
        add_message("user", user_input)
        
        with st.spinner("Clara is thinking..."):
            
            # Check if we're waiting for red flag response
            if st.session_state.awaiting_red_flag_response:
                clara_response, should_end, end_reason = clara_agent.handle_red_flag_response(user_input)
                st.session_state.awaiting_red_flag_response = False
            else:
                # Normal response processing
                clara_response, should_end, end_reason = clara_agent.process_patient_response(user_input)
                
                # Check if this triggered a red flag
                if clara_agent.state.red_flags_detected and not should_end:
                    last_red_flag = clara_agent.state.red_flags_detected[-1]
                    if last_red_flag.action_taken == "awaiting_response":
                        st.session_state.awaiting_red_flag_response = True
                        show_red_flag_warning()
            
            # Generate audio if enabled
            audio_bytes = None
            if st.session_state.tts_enabled:
                audio_bytes = speech_service.text_to_speech(clara_response)
            
            # Add Clara's response to chat
            add_message("assistant", clara_response, audio_bytes)
            
            # Check if conversation should end
            if should_end:
                st.session_state.conversation_ended = True
                
                # Save conversation transcript
                filepath = clara_agent.save_conversation()
                st.session_state.conversation_filepath = filepath
                
                # Generate summaries
                st.info("📊 Generating summary for your doctor...")
                
                try:
                    from core.summary_generator import SummaryGenerator
                    
                    generator = SummaryGenerator()
                    outputs = generator.generate_all_outputs(clara_agent.state)
                    summary_filepath = generator.save_outputs(outputs)
                    
                    st.session_state.summary_filepath = summary_filepath
                    st.success("✅ Summary generated and saved!")
                    
                except Exception as e:
                    st.error(f"⚠️ Error generating summary: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Rerun to update UI
        st.rerun()