# app.py

import streamlit as st
from core.clara_agent import ClaraAgent
from services.azure_speech import AzureSpeechService
from ui.chat_interface import render_chat_interface
from ui.summary_view import show_summary_selector
from config.settings import settings
import uuid

# Page config
st.set_page_config(
    page_title="Clara History",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application entry point"""
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🩺 Clara History")
        
        page = st.radio(
            "Navigation",
            ["Patient Chat", "GP Summary View"],
            help="Switch between patient conversation and GP summary viewer"
        )
    
    # Route to appropriate page
    if page == "Patient Chat":
        show_patient_chat()
    elif page == "GP Summary View":
        show_gp_view()


def show_patient_chat():
    if 'setup_complete' not in st.session_state:
        show_patient_setup()
    else:
        # Initialize services
        initialize_services()
        
        # Render chat interface
        render_chat_interface(
            clara_agent=st.session_state.clara_agent,
            speech_service=st.session_state.speech_service
        )


def show_gp_view():
    """Show the GP summary viewer"""
    show_summary_selector()


def initialize_services():
    """Initialize Clara and Speech services (cached)"""
    if 'clara_agent' not in st.session_state:
        # For MVP, use demo patient info
        patient_name = st.session_state.get('patient_name', 'Demo Patient')
        doctor_name = st.session_state.get('doctor_name', 'Dr Smith')
        
        st.session_state.clara_agent = ClaraAgent(
            patient_name=patient_name,
            doctor_name=doctor_name,
            appointment_id=str(uuid.uuid4())
        )
    
    if 'speech_service' not in st.session_state:
        st.session_state.speech_service = AzureSpeechService()


def show_patient_setup():
    """Show initial setup page to enter patient/doctor details (for MVP testing)"""
    
    st.title("🩺 Clara History - Pre-Consultation Assistant")
    
    st.markdown("""
    ### Welcome!
    
    This is Clara, your pre-consultation assistant.
    
    In production, you would access this via a unique link sent after booking your appointment.
    For now, please enter demo details below:
    """)
    
    with st.form("patient_setup"):
        st.subheader("Patient Information")
        
        patient_name = st.text_input(
            "Patient Name",
            value="Demo Patient",
            help="Your name"
        )
        
        doctor_name = st.text_input(
            "Doctor Name",
            value="Dr Smith",
            help="The doctor you're seeing"
        )
        
        appointment_reason = st.text_input(
            "Appointment Reason (optional)",
            placeholder="e.g., Follow-up, New concern",
            help="Brief reason for appointment if known"
        )
        
        submitted = st.form_submit_button("Start Consultation Questions", type="primary")
        
        if submitted:
            if patient_name and doctor_name:
                # Save to session state
                st.session_state.patient_name = patient_name
                st.session_state.doctor_name = doctor_name
                st.session_state.appointment_reason = appointment_reason
                st.session_state.setup_complete = True
                
                st.rerun()
            else:
                st.error("Please enter both patient and doctor names.")
    
    # Footer
    st.divider()
    st.caption("Clara History MVP - Pre-consultation medical history assistant")


if __name__ == "__main__":
    main()