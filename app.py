"""
medical_ai_ui.py - Simple Streamlit UI for Medical AI
No FastAPI needed - direct crew integration
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Fix imports
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.medical_nlp_crew_agent import EnhancedMedicalCrewMVP
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Medical AI Clinical Decision Support",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .agent-output {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-confidence {
        color: green;
        font-weight: bold;
    }
    .medium-confidence {
        color: orange;
    }
    .low-confidence {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

# Initialize crew (cached so it only loads once)
@st.cache_resource
def get_crew():
    """Initialize crew once and cache it"""
    with st.spinner("Loading Medical AI system..."):
        crew = EnhancedMedicalCrewMVP(umls_api_key=os.getenv("UMLS_API_KEY"))

        # IMPORTANT: Disable SQLite cache to avoid threading issues with Streamlit
        # Streamlit uses multiple threads and SQLite connections aren't thread-safe
        crew.nlp_agent_handler.ner_pipeline.pipeline.umls_client.use_cache = False

        st.success("✓ Medical AI loaded (cache disabled for Streamlit)")
    return crew

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.title("🏥 Medical AI Clinical Decision Support System")
st.markdown("**MVP Version - BioBERT + UMLS + CrewAI**")
st.markdown("---")

# Sidebar for settings and info
with st.sidebar:
    st.header("System Information")

    crew = get_crew()

    st.success("✓ System Ready")
    st.info("**Components:**\n- BioBERT NER\n- UMLS Validation\n- 5 Specialized Agents")

    st.markdown("---")

    # Settings
    st.header("Settings")
    show_ner = st.checkbox("Show NER Details", value=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=True)

    st.markdown("---")

    # Example cases
    st.header("Example Cases")
    if st.button("Load Example 1"):
        st.session_state.example_text = """28-year-old male presenting with severe sore throat for 4 days, fever of 101°F, difficulty swallowing, and swollen lymph nodes. No significant medical history."""

    if st.button("Load Example 2"):
        st.session_state.example_text = """58-year-old woman with chest pain and shortness of breath. History of hypertension and type 2 diabetes. Currently taking metformin 1000mg twice daily and lisinopril 10mg daily."""

    if st.button("Load Example 3"):
        st.session_state.example_text = """45-year-old male with severe headache, fever of 102°F, and neck stiffness for 2 days. Also experiencing photophobia and nausea."""

# # Main content area
# col1, col2 = st.columns([2, 1])

# with col1:
st.header("Patient Case Input")

# Text input
default_text = st.session_state.get('example_text', '')
patient_input = st.text_area(
    "Enter patient symptoms, history, and current medications:",
    height=150,
    value=default_text,
    placeholder="Example: 28-year-old male with fever, sore throat..."
)

# Clear example
if default_text:
    if st.button("Clear"):
        st.session_state.example_text = ""
        st.rerun()

# Analyze button
analyze_button = st.button("Analyze Case", type="primary", use_container_width=True)



# Processing
if analyze_button and patient_input.strip():
    st.markdown("---")
    st.header("Analysis Results")

    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        start_time = time.time()

        # Step 1: NER Extraction
        status_text.text("Step 1/5: Extracting medical entities...")
        progress_bar.progress(20)

        with st.spinner("Processing with BioBERT NER..."):
            ner_result = crew.nlp_agent_handler.process_medical_text(patient_input)

        # Display NER results
        if show_ner:
            with st.expander("📊 NER Extraction Results", expanded=True):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Diseases", ner_result['statistics']['diseases'])
                with col_b:
                    st.metric("Symptoms", ner_result['statistics']['symptoms'])
                with col_c:
                    st.metric("Medications", ner_result['statistics']['medications'])

                # Show entities
                if ner_result['entities']['diseases']:
                    st.subheader("Diseases Found:")
                    for disease in ner_result['entities']['diseases']:
                        conf = disease['confidence']
                        conf_class = "high-confidence" if conf > 0.8 else "medium-confidence" if conf > 0.5 else "low-confidence"
                        st.markdown(f"- {disease['text']} <span class='{conf_class}'>({conf:.2f})</span>", unsafe_allow_html=True)

                if ner_result['entities']['medications']:
                    st.subheader("Medications Found:")
                    for med in ner_result['entities']['medications']:
                        dosage = f" - {med.get('dosage')}" if med.get('dosage') else ""
                        st.markdown(f"- {med['name']}{dosage} ({med['confidence']:.2f})")

        # Step 2-5: Full crew processing
        status_text.text("Step 2/5: Clinical reasoning analysis...")
        progress_bar.progress(40)

        with st.spinner("Running multi-agent analysis..."):
            result = crew.process_medical_query_enhanced(patient_input)

        processing_time = time.time() - start_time

        progress_bar.progress(100)
        status_text.text("✓ Analysis complete!")

        st.success(f"Analysis completed in {processing_time:.1f} seconds")

        # Display agent outputs
        st.markdown("---")
        st.header("Agent Analysis")

        # Agent 1: NLP Analysis (already shown above)

        # Agent 2: Clinical Reasoning
        with st.expander("🧠 Clinical Reasoning Specialist", expanded=True):
            st.markdown('<div class="agent-output">', unsafe_allow_html=True)
            clinical_reasoning = result.get('clinical_analysis', {}).get('clinical_reasoning',
                                result.get('clinical_reasoning', 'No clinical reasoning available'))
            st.write(clinical_reasoning)
            st.markdown('</div>', unsafe_allow_html=True)

        # Agent 3: Pharmacology
        with st.expander("💊 Pharmacology Specialist", expanded=False):
            st.markdown('<div class="agent-output">', unsafe_allow_html=True)
            pharmacology = result.get('clinical_analysis', {}).get('drug_safety',
                          result.get('drug_safety', 'No drug interaction analysis available'))
            st.write(pharmacology)
            st.markdown('</div>', unsafe_allow_html=True)

        # Agent 4: Validation
        with st.expander("✓ Medical Knowledge Validator", expanded=False):
            st.markdown('<div class="agent-output">', unsafe_allow_html=True)
            validation = result.get('clinical_analysis', {}).get('validation',
                        result.get('validation', 'No validation report available'))
            st.write(validation)
            st.markdown('</div>', unsafe_allow_html=True)

        # Agent 5: Patient Education
        with st.expander("📚 Patient Education Specialist", expanded=False):
            st.markdown('<div class="agent-output">', unsafe_allow_html=True)
            education = result.get('clinical_analysis', {}).get('patient_summary',
                       result.get('patient_education', 'No patient education available'))
            st.write(education)
            st.markdown('</div>', unsafe_allow_html=True)

        # Recommendations
        st.markdown("---")
        st.header("Recommendations")
        recommendations = result.get('recommendations', {})
        if recommendations:
            st.json(recommendations)
        else:
            st.info("Check agent outputs above for recommendations")

        # Warning
        st.warning("⚠️ **IMPORTANT:** This system requires human review. All outputs must be validated by a qualified healthcare professional before any clinical decisions.")

        # Save to history
        st.session_state.history.append({
            'timestamp': datetime.now().isoformat(),
            'input': patient_input[:100] + "..." if len(patient_input) > 100 else patient_input,
            'processing_time': processing_time,
            'entities_found': ner_result['total_entities']
        })

    except Exception as e:
        st.error(f"Error during analysis: {e}")
        st.exception(e)
        progress_bar.progress(0)
        status_text.text("Analysis failed")

elif analyze_button:
    st.warning("Please enter patient case information")

# History section (bottom)
if st.session_state.history:
    st.markdown("---")
    st.header("Analysis History")

    for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5
        with st.expander(f"Case {len(st.session_state.history) - i}: {item['timestamp'][:19]}"):
            st.write(f"**Input:** {item['input']}")
            st.write(f"**Entities Found:** {item['entities_found']}")
            st.write(f"**Processing Time:** {item['processing_time']:.1f}s")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Medical AI Clinical Decision Support System v2.0 MVP<br>
For educational and research purposes only. Not for clinical use without human oversight.
</div>
""", unsafe_allow_html=True)