import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- SETTINGS & THEME ---
st.set_page_config(page_title="ProDiagnose AI", page_icon="🧪", layout="wide")

# Custom CSS for a modern "Clinic" look
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f0f2f6; border-right: 1px solid #e0e0e0; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .stProgress > div > div > div > div { background-color: #10b981; }
    .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED DATA LOADING ---
@st.cache_resource
def load_assets():
    with open("artifacts/model_trainer/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/data_transformation/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_assets()
symptoms_list = sorted(model.feature_names_in_)

# --- SIDEBAR: PATIENT DATA ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=100)
    st.header("Patient Profile")
    patient_name = st.text_input("Full Name", "Guest User")
    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    st.divider()
    st.info("💡 Tip: Select all symptoms that have persisted for more than 24 hours.")

# --- MAIN UI ---
st.markdown('<p class="main-header">Advanced Disease Analysis Portal</p>', unsafe_allow_html=True)

# Layout: 2 Columns
col_input, col_viz = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("🔍 Symptom Input")
    search_terms = st.multiselect(
        "Search and select symptoms:",
        options=symptoms_list,
        placeholder="e.g. Fever, Cough, Joint Pain",
        help="The model uses binary encoding to analyze these inputs."
    )
    
    predict_btn = st.button("Generate Diagnostic Report", type="primary", use_container_width=True)

# --- PREDICTION LOGIC ---
if predict_btn:
    if len(search_terms) < 2:
        st.error("Please select at least 2 symptoms for a valid analysis.")
    else:
        # Prepare Input
        input_vector = [1 if s in search_terms else 0 for s in symptoms_list]
        input_array = np.array(input_vector).reshape(1, -1)
        
        # Predictions
        pred_idx = model.predict(input_array)[0]
        disease = le.inverse_transform([pred_idx])[0]
        probs = model.predict_proba(input_array)[0]
        conf_score = np.max(probs)
        
        with col_viz:
            st.subheader("📊 Diagnostic Results")
            
            # 1. Gauge Chart for Confidence
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = conf_score * 100,
                title = {'text': "Confidence Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffcfcf"},
                        {'range': [50, 80], 'color': "#fef9c3"},
                        {'range': [80, 100], 'color': "#dcfce7"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # 2. Predicted Disease Highlight
            st.markdown(f"""
                <div style="background-color: #eff6ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1E3A8A;">
                    <small>PREDICTED CONDITION</small>
                    <h2 style="margin:0; color: #1E3A8A;">{disease}</h2>
                </div>
            """, unsafe_allow_html=True)

        # --- ADVANCED BREAKDOWN (Below Columns) ---
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["📈 Probability Breakdown", "🧪 Symptom Contribution", "📝 Next Steps"])
        
        with tab1:
            # Show Top 5 likely diseases
            top_5_idx = np.argsort(probs)[-5:][::-1]
            top_5_labels = le.inverse_transform(top_5_idx)
            top_5_probs = probs[top_5_idx]
            
            df_probs = pd.DataFrame({"Disease": top_5_labels, "Probability": top_5_probs})
            fig_bar = px.bar(df_probs, x='Probability', y='Disease', orientation='h', 
                             title="Top 5 Potential Matches", color='Probability',
                             color_continuous_scale='Blues')
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            st.write("The following symptoms were the primary drivers for this prediction:")
            # Display chips for selected symptoms
            cols = st.columns(4)
            for i, s in enumerate(search_terms):
                cols[i % 4].markdown(f"✅ `{s}`")
            
            st.caption("Note: This model uses a Random Forest/XGBoost approach to evaluate feature weights.")

        with tab3:
            st.warning("**Disclaimer:** This is an AI-generated assessment, not a clinical diagnosis.")
            st.markdown(f"""
            **Recommended Actions for {patient_name}:**
            1. **Consultation:** Please share this probability report with a General Practitioner.
            2. **Monitoring:** Keep a log of symptom intensity over the next 48 hours.
            3. **Urgency:** If symptoms like shortness of breath or high fever persist, seek immediate care.
            """)
            
            # Simple "Download Report" Mockup
            report_text = f"Patient: {patient_name}\nAge: {age}\nPredicted Disease: {disease}\nConfidence: {conf_score:.2%}"
            st.download_button("Download Summary Report", report_text, file_name="health_report.txt")