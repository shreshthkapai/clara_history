import streamlit as st
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def render_summary_view(outputs: Dict[str, Any]):
    """
    Render the GP summary view with all 5 outputs
    
    Args:
        outputs: Dictionary containing all 5 outputs from summary generator
    """
    
    st.title("📋 Pre-Consultation Summary")
    
    # Header info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Patient", outputs.get('patient_name', 'Unknown'))
    
    with col2:
        st.metric("Doctor", outputs.get('doctor_name', 'Unknown'))
    
    with col3:
        duration = outputs.get('conversation_stats', {}).get('duration_minutes', 0)
        st.metric("Duration", f"{duration} min")
    
    st.divider()
    
    # Safety banner if red flags present
    red_flags = outputs.get('red_flags', [])
    if red_flags:
        render_red_flag_banner(red_flags)
    
    # Main content - 2 column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Output 2: Short Summary (top priority)
        render_short_summary(outputs.get('short_summary', ''))
        
        # Output 3: Long Summary (detailed)
        render_long_summary(outputs.get('long_summary', ''))
        
        # Output 1: Full Transcript (expandable)
        render_transcript(outputs.get('full_transcript', []))
    
    with col_right:
        # Output 4: What to Prepare
        render_what_to_prepare(outputs.get('what_to_prepare', []))
        
        # Output 5: Probable Conditions
        render_probable_conditions(outputs.get('probable_conditions', []))
        
        # Conversation Stats
        render_conversation_stats(outputs.get('conversation_stats', {}))


def render_red_flag_banner(red_flags):
    """Display warning banner for red flags"""
    st.error("🚨 **URGENT: Red Flag Detected**")
    
    for flag in red_flags:
        category = flag.get('category', 'Unknown')
        severity = flag.get('severity', 'Unknown')
        action = flag.get('action_taken', 'Unknown')
        
        st.warning(f"""
        **Category:** {category.replace('_', ' ').title()}  
        **Severity:** {severity.upper()}  
        **Action:** {action.replace('_', ' ').title()}  
        
        ⚠️ Patient was advised to seek emergency care during pre-consultation.
        """)
    
    st.divider()


def render_short_summary(short_summary: str):
    """Display the 30-second read summary"""
    st.subheader("⚡ Quick Summary (30-sec read)")
    
    if short_summary:
        st.info(short_summary)
    else:
        st.warning("No short summary available.")
    
    st.divider()


def render_long_summary(long_summary: str):
    """Display the detailed clinical summary"""
    st.subheader("📄 Detailed Clinical Summary")
    
    if long_summary:
        # Display in markdown for formatting
        st.markdown(long_summary)
    else:
        st.warning("No detailed summary available.")
    
    st.divider()


def render_transcript(transcript):
    """Display full transcript in expandable section"""
    with st.expander("💬 View Full Transcript", expanded=False):
        if transcript:
            st.caption(f"Total messages: {len(transcript)}")
            
            for msg in transcript:
                speaker = msg.get('speaker', 'unknown')
                text = msg.get('text', '')
                timestamp = msg.get('timestamp', '')
                
                # Parse timestamp
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = ""
                
                if speaker == 'clara':
                    st.chat_message("assistant").write(f"**Clara** ({time_str}): {text}")
                else:
                    st.chat_message("user").write(f"**Patient** ({time_str}): {text}")
        else:
            st.warning("No transcript available.")


def render_what_to_prepare(prep_items):
    """Display what to prepare section"""
    st.subheader("📦 What to Prepare")
    
    if prep_items:
        for item in prep_items:
            st.markdown(f"- {item}")
    else:
        st.info("No specific preparation items suggested.")
    
    st.divider()


def render_probable_conditions(conditions):
    """Display probable conditions (differential diagnoses)"""
    st.subheader("🔍 Probable Conditions")
    st.caption("AI-suggested differentials for consideration")
    
    if conditions:
        for i, condition in enumerate(conditions, 1):
            condition_name = condition.get('condition', 'Unknown')
            rationale = condition.get('rationale', 'No rationale provided')
            
            with st.expander(f"{i}. {condition_name}", expanded=(i == 1)):
                st.write(f"**Rationale:** {rationale}")
    else:
        st.info("No probable conditions suggested.")
    
    st.divider()


def render_conversation_stats(stats):
    """Display conversation statistics"""
    st.subheader("📊 Conversation Stats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Messages", stats.get('total_messages', 0))
        st.metric("Questions Asked", stats.get('questions_asked', 0))
    
    with col2:
        st.metric("Topics Covered", stats.get('topics_covered', 0))
        st.metric("Duration", f"{stats.get('duration_minutes', 0)} min")


def load_summary_from_file(filepath: Path) -> Dict[str, Any]:
    """
    Load a saved summary JSON file
    
    Args:
        filepath: Path to summary JSON file
    
    Returns:
        Dictionary with outputs
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading summary: {e}")
        return {}


def show_summary_selector():
    """
    Show a UI to select and view saved summaries
    (For testing/demo purposes)
    """
    st.title("📋 Clara History - GP Summary Viewer")
    
    st.markdown("""
    View pre-consultation summaries from completed patient conversations.
    """)
    
    # Get list of saved summaries
    summary_dir = Path("data/conversations")
    
    if not summary_dir.exists():
        st.warning("No summaries found. Complete a patient conversation first.")
        return
    
    # Find all summary files using standard naming (SUMMARY_*.json)
    summary_files = list(summary_dir.glob("SUMMARY_*.json"))
    
    # Fallback: also include old naming patterns
    if len(summary_files) == 0:
        summary_files = list(summary_dir.glob("summary_*.json"))
    
    if not summary_files:
        st.info("No summaries available yet. Patient conversations will appear here once completed.")
        return
    
    # Sort by most recent (filename has timestamp, so reverse alphabetical = newest first)
    summary_files.sort(reverse=True)
    
    # Create selection with better display names
    st.subheader("Select Summary")
    
    file_options = {}
    for f in summary_files:
        # Parse filename: SUMMARY_20251130_143045_John_Smith.json
        # Display as: John Smith - 2025-11-30 14:30
        try:
            parts = f.stem.replace('SUMMARY_', '').replace('summary_', '').split('_')
            if len(parts) >= 3:
                date_str = parts[0]  # 20251130
                time_str = parts[1]  # 143045
                patient_name = ' '.join(parts[2:])  # John Smith
                
                # Format nicely
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                formatted_time = f"{time_str[:2]}:{time_str[2:4]}"
                
                display_name = f"{patient_name} - {formatted_date} {formatted_time}"
            else:
                # Fallback to filename
                display_name = f.stem.replace('_', ' ')
        except:
            # Fallback to filename
            display_name = f.stem.replace('_', ' ')
        
        file_options[display_name] = f
    
    selected_name = st.selectbox(
        "Choose a conversation summary:",
        options=list(file_options.keys())
    )
    
    if selected_name:
        selected_file = file_options[selected_name]
        
        # Load and display
        outputs = load_summary_from_file(selected_file)
        
        if outputs:
            st.divider()
            render_summary_view(outputs)