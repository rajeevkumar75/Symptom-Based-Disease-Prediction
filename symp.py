import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI · Diagnostic Portal",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS  —  Deep clinical dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --surface2:  #1a2235;
    --border:    rgba(99,179,237,0.12);
    --accent:    #38bdf8;
    --accent2:   #818cf8;
    --danger:    #f87171;
    --success:   #34d399;
    --warn:      #fbbf24;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-sans: 'DM Sans', sans-serif;
    --font-mono: 'Space Mono', monospace;
    --font-disp: 'Fraunces', serif;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Main header ── */
.hero-title {
    font-family: var(--font-disp);
    font-size: 3rem;
    font-weight: 300;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-title span { color: var(--accent); font-style: italic; }
.hero-sub {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Pill badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-cyan  { color: var(--accent);  border-color: var(--accent);  background: rgba(56,189,248,.08); }
.badge-indigo{ color: var(--accent2); border-color: var(--accent2); background: rgba(129,140,248,.08); }
.badge-green { color: var(--success); border-color: var(--success); background: rgba(52,211,153,.08); }
.badge-red   { color: var(--danger);  border-color: var(--danger);  background: rgba(248,113,113,.08); }

/* ── Card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-title {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
}

/* ── Disease result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(56,189,248,.06) 0%, rgba(129,140,248,.06) 100%);
    border: 1px solid rgba(56,189,248,.25);
    border-radius: 16px;
    padding: 28px 28px 20px;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 140px; height: 140px;
    background: radial-gradient(circle, rgba(56,189,248,.15) 0%, transparent 70%);
    border-radius: 50%;
}
.result-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
}
.result-disease {
    font-family: var(--font-disp);
    font-size: 2rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.result-conf {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 10px;
}

/* ── Symptom chips ── */
.chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.chip {
    background: rgba(56,189,248,.08);
    border: 1px solid rgba(56,189,248,.2);
    color: var(--accent);
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-family: var(--font-mono);
    letter-spacing: 0.04em;
}

/* ── Stat strip ── */
.stat-row { display: flex; gap: 16px; margin-bottom: 20px; }
.stat-box {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
}
.stat-value {
    font-family: var(--font-disp);
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
}
.stat-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 4px;
}

/* ── Multiselect pills ── */
.stMultiSelect [data-baseweb="tag"] {
    background-color: rgba(56,189,248,.15) !important;
    border: 1px solid rgba(56,189,248,.3) !important;
    color: var(--accent) !important;
    border-radius: 99px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
}
.stMultiSelect [data-baseweb="select"] {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #0b0f1a !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: 10px !important;
    gap: 4px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 7px !important;
    padding: 8px 18px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0b0f1a !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Sidebar slider ── */
[data-testid="stSlider"] .rc-slider-track { background-color: var(--accent) !important; }
[data-testid="stSlider"] .rc-slider-handle {
    border-color: var(--accent) !important;
    background: var(--accent) !important;
}

/* ── Warning / info boxes ── */
.stAlert { border-radius: 10px !important; }

/* ── Plotly chart container ── */
.js-plotly-plot .plotly { border-radius: 12px; }

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 8px !important;
}
.stDownloadButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* ── Session timestamp ── */
.timestamp {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.08em;
}

/* ── Sidebar section labels ── */
.sidebar-section {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 16px 0 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open("artifacts/model_trainer/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/data_transformation/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_assets()
symptoms_list = sorted(model.feature_names_in_)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px;">
        <div style="font-family:'Fraunces',serif; font-size:1.4rem; font-weight:300; color:#e2e8f0;">
            Medi<span style="color:#38bdf8; font-style:italic;">Scan</span>
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.55rem; color:#64748b;
                    letter-spacing:0.15em; text-transform:uppercase; margin-top:2px;">
            AI Diagnostic Engine v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Patient Profile</div>', unsafe_allow_html=True)
    patient_name = st.text_input("Full Name", "Guest User", label_visibility="collapsed",
                                  placeholder="Patient full name")
    
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age", 1, 120, 25, label_visibility="visible")
    with col_b:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    now = datetime.now()
    st.markdown(f"""
    <div class="timestamp">
        📅 {now.strftime("%d %b %Y")}<br>
        🕐 {now.strftime("%H:%M")} IST
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:rgba(56,189,248,.06); border:1px solid rgba(56,189,248,.15);
                border-radius:10px; padding:14px; font-size:0.78rem; color:#94a3b8;
                line-height:1.6;">
        <span style="color:#38bdf8; font-family:'Space Mono',monospace; font-size:0.65rem;">
        ⚕ GUIDANCE
        </span><br><br>
        Select symptoms persisting for <strong style="color:#e2e8f0">≥ 24 hours</strong>.
        More symptoms = higher confidence. Minimum <strong style="color:#e2e8f0">2 required</strong>.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
col_hdr, col_badge = st.columns([3, 1])
with col_hdr:
    st.markdown(f"""
    <div class="hero-title">Advanced <span>Diagnostic</span><br>Analysis Portal</div>
    <div class="hero-sub">AI-Powered · {len(symptoms_list)} Symptoms · Multi-class Classification</div>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style="text-align:right; padding-top:12px;">
        <span class="badge badge-green">● System Online</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT + RESULTS LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-title">▸ Symptom Selection</div>', unsafe_allow_html=True)
    search_terms = st.multiselect(
        "Select symptoms:",
        options=symptoms_list,
        placeholder="Search symptoms — e.g. Fever, Cough, Fatigue",
        label_visibility="collapsed"
    )

    # Live chip preview
    if search_terms:
        chips_html = "".join([f'<span class="chip">{s}</span>' for s in search_terms])
        st.markdown(f'<div class="chip-wrap">{chips_html}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#64748b;
                    margin-top:10px;">{len(search_terms)} symptom(s) selected</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#64748b; font-size:0.8rem; padding:16px 0; font-style:italic;">
            No symptoms selected yet.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚕  Run Diagnostic Analysis", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_btn:
    if len(search_terms) < 2:
        st.error("⚠ Please select at least **2 symptoms** for a meaningful analysis.", icon="🚫")
    else:
        input_vector = [1 if s in search_terms else 0 for s in symptoms_list]
        input_array  = np.array(input_vector).reshape(1, -1)

        pred_idx   = model.predict(input_array)[0]
        disease    = le.inverse_transform([pred_idx])[0]
        probs      = model.predict_proba(input_array)[0]
        conf_score = float(np.max(probs))

        # Confidence badge color
        if conf_score >= 0.80:
            conf_badge = "badge-green"; conf_label = "High Confidence"
        elif conf_score >= 0.50:
            conf_badge = "badge-indigo"; conf_label = "Moderate Confidence"
        else:
            conf_badge = "badge-red"; conf_label = "Low Confidence"

        with col_right:
            # ── Stat strip ──
            top5_idx    = np.argsort(probs)[-5:][::-1]
            top5_labels = le.inverse_transform(top5_idx)
            top5_probs  = probs[top5_idx]

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-value">{conf_score:.0%}</div>
                    <div class="stat-label">Confidence</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(search_terms)}</div>
                    <div class="stat-label">Symptoms</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{top5_probs[1]:.0%}</div>
                    <div class="stat-label">2nd Match</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Disease result card ──
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Condition</div>
                <div class="result-disease">{disease}</div>
                <div class="result-conf">
                    <span class="badge {conf_badge}">{conf_label}</span>
                    &nbsp; Confidence: {conf_score:.1%} · Patient: {patient_name}, {age}y
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── TABS ──
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Probability Breakdown", "Symptom Review", "Clinical Next Steps"])

        with tab1:
            df_probs = pd.DataFrame({"Disease": top5_labels, "Probability": top5_probs})

            fig_bar = go.Figure(go.Bar(
                x=top5_probs,
                y=top5_labels,
                orientation='h',
                marker=dict(
                    color=top5_probs,
                    colorscale=[[0, "#1a2235"], [0.5, "#0ea5e9"], [1, "#38bdf8"]],
                    line=dict(color="rgba(56,189,248,.2)", width=1)
                ),
                text=[f"{p:.1%}" for p in top5_probs],
                textposition='outside',
                textfont=dict(family="Space Mono, monospace", size=11, color="#94a3b8"),
                hovertemplate="<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>"
            ))
            fig_bar.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(family="DM Sans, sans-serif", color="#94a3b8"),
                margin=dict(l=10, r=60, t=20, b=10),
                height=240,
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(99,179,237,0.08)",
                    tickformat=".0%", color="#64748b",
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=False, color="#94a3b8",
                    automargin=True
                ),
                bargap=0.35,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf_score * 100,
                number=dict(suffix="%", font=dict(family="Fraunces, serif", color="#38bdf8", size=36)),
                title=dict(text="Confidence Score", font=dict(family="Space Mono, monospace",
                                                               color="#64748b", size=11)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#64748b",
                              tickfont=dict(family="Space Mono", size=9), tickwidth=1),
                    bar=dict(color="#38bdf8", thickness=0.6),
                    bgcolor="#1a2235",
                    bordercolor="rgba(99,179,237,0.15)",
                    steps=[
                        dict(range=[0, 50],  color="#2d1f1f"),
                        dict(range=[50, 80], color="#1f2820"),
                        dict(range=[80, 100],color="#1a2633"),
                    ],
                    threshold=dict(line=dict(color="#38bdf8", width=2), thickness=0.8, value=conf_score * 100)
                )
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#111827",
                height=220,
                margin=dict(l=20, r=20, t=40, b=0)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with tab2:
            st.markdown(f"""
            <div class="card-title" style="margin-bottom:16px;">
                {len(search_terms)} symptoms flagged for analysis
            </div>
            """, unsafe_allow_html=True)

            chips_html = "".join([f'<span class="chip">✓ {s}</span>' for s in search_terms])
            st.markdown(f'<div class="chip-wrap">{chips_html}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(129,140,248,.06); border:1px solid rgba(129,140,248,.15);
                        border-radius:10px; padding:14px 18px; font-size:0.78rem; color:#94a3b8; line-height:1.7;">
                <span style="color:#818cf8; font-family:'Space Mono',monospace; font-size:0.62rem;">
                ◈ MODEL NOTE
                </span><br><br>
                The underlying model uses a <strong style="color:#e2e8f0">Random Forest / XGBoost</strong>
                ensemble with binary symptom encoding. Each selected symptom contributes weighted
                feature importance toward the final classification.
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown(f"""
            <div style="background:rgba(251,191,36,.05); border:1px solid rgba(251,191,36,.2);
                        border-radius:10px; padding:14px 18px; font-size:0.78rem; color:#fbbf24;
                        font-family:'Space Mono',monospace; letter-spacing:0.06em; margin-bottom:20px;">
                ⚠ AI ASSESSMENT · NOT A CLINICAL DIAGNOSIS · CONSULT A LICENSED PHYSICIAN
            </div>
            """, unsafe_allow_html=True)

            steps = [
                ("01", "GP Consultation", f"Share this probability report with a General Practitioner, referencing the {disease} prediction.", "#38bdf8"),
                ("02", "Symptom Monitoring", "Log symptom intensity every 12 hours over the next 48 hours. Note any escalation.", "#818cf8"),
                ("03", "Urgent Escalation", "Seek immediate emergency care if shortness of breath, chest pain, or high fever (>103°F) develops.", "#f87171"),
                ("04", "Lab Work", "Request relevant bloodwork or imaging as advised by your physician to confirm or rule out the diagnosis.", "#34d399"),
            ]
            for num, title, desc, color in steps:
                st.markdown(f"""
                <div style="display:flex; gap:16px; padding:14px 0; border-bottom:1px solid rgba(99,179,237,.08);">
                    <div style="font-family:'Fraunces',serif; font-size:1.4rem; font-weight:600;
                                color:{color}; opacity:0.5; min-width:32px;">{num}</div>
                    <div>
                        <div style="font-family:'Space Mono',monospace; font-size:0.68rem;
                                    color:#e2e8f0; letter-spacing:0.08em; text-transform:uppercase;
                                    margin-bottom:4px;">{title}</div>
                        <div style="font-size:0.8rem; color:#94a3b8; line-height:1.6;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            report_lines = [
                "═══════════════════════════════════════",
                "        MEDISCAN AI · HEALTH REPORT    ",
                "═══════════════════════════════════════",
                f"Patient Name   : {patient_name}",
                f"Age / Gender   : {age} / {gender}",
                f"Generated At   : {now.strftime('%d %b %Y, %H:%M IST')}",
                "───────────────────────────────────────",
                f"Predicted Condition : {disease}",
                f"Confidence Score    : {conf_score:.1%}",
                f"Confidence Level    : {conf_label}",
                "───────────────────────────────────────",
                "Symptoms Selected:",
                *[f"  • {s}" for s in search_terms],
                "───────────────────────────────────────",
                "Top 5 Differential Diagnoses:",
                *[f"  {i+1}. {label:<30} {prob:.1%}" for i, (label, prob) in enumerate(zip(top5_labels, top5_probs))],
                "═══════════════════════════════════════",
                "DISCLAIMER: AI-generated, not a clinical",
                "diagnosis. Consult a licensed physician.",
                "═══════════════════════════════════════",
            ]
            report_text = "\n".join(report_lines)

            st.download_button(
                label="↓  Download Full Report (.txt)",
                data=report_text,
                file_name=f"mediscan_report_{patient_name.replace(' ', '_').lower()}_{now.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )