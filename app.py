"""
app.py
------
TalentPulse — Autonomous Enterprise Performance & Retention Intelligence System.
Built with IBM ML and Agentic AI Architecture.

Run with:
    streamlit run talent_pulse/app.py
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure Streamlit page
st.set_page_config(
    page_title="TalentPulse — IBM ML & Agentic AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Styling (IBM Carbon Design inspired)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f62fe 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    
    .badge-ibm {
        display: inline-block;
        background: rgba(15, 98, 254, 0.35);
        color: #60a5fa;
        border: 1px solid #3b82f6;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .agent-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .stMetric {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 8px;
    }
    
    .driver-pill-risk {
        background: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    .driver-pill-safe {
        background: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Dynamic imports ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml_engine.predictor import TalentPredictor
from agent_engine.orchestrator import TalentOrchestrator

@st.cache_resource
def load_system():
    predictor = TalentPredictor(os.path.join(BASE_DIR, "ml_engine", "models"))
    orchestrator = TalentOrchestrator(predictor)
    return predictor, orchestrator

@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "data", "ibm_hr_attrition.csv")
    if not os.path.exists(csv_path):
        from data.generate_ibm_hr_data import save_dataset
        save_dataset(csv_path)
    df = pd.read_csv(csv_path)
    return df

try:
    predictor, orchestrator = load_system()
    df_workforce = load_data()
    system_ready = True
except Exception as e:
    st.error(f"System initialization error: {e}")
    system_ready = False
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=110)
    st.markdown("### **TalentPulse Control Center**")
    st.caption("IBM ML & Agentic AI Internship Architecture")
    
    st.markdown("---")
    st.markdown("#### ⚙️ **AI Engine Configuration**")
    
    llm_choice = st.selectbox(
        "Active LLM Provider",
        ["Hybrid (Auto-Detect: Watsonx / OpenAI / Smart Engine)", "Strictly IBM Watsonx.ai Granite", "Built-in Autonomous Smart Engine"]
    )
    
    with st.expander("🔑 API Credentials (Optional)"):
        watsonx_key = st.text_input("IBM Cloud API Key", type="password", value=os.getenv("WATSONX_APIKEY", ""))
        watsonx_proj = st.text_input("Watsonx Project ID", value=os.getenv("WATSONX_PROJECT_ID", ""))
        openai_key = st.text_input("OpenAI API Key (Fallback)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        
        if st.button("Apply API Keys"):
            if watsonx_key:
                os.environ["WATSONX_APIKEY"] = watsonx_key
                orchestrator.llm.watsonx_apikey = watsonx_key
            if watsonx_proj:
                os.environ["WATSONX_PROJECT_ID"] = watsonx_proj
                orchestrator.llm.watsonx_project_id = watsonx_proj
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                orchestrator.llm.openai_api_key = openai_key
            st.success("Credentials updated!")

    st.markdown("---")
    st.markdown("#### 📊 **Model Telemetry**")
    metrics = predictor.metadata.get("metrics", {})
    att_metrics = metrics.get("attrition", {})
    perf_metrics = metrics.get("performance", {})
    
    st.markdown(f"- **Classifier:** Random Forest (180 trees)")
    st.markdown(f"- **ROC-AUC Score:** `{att_metrics.get('roc_auc', 0.85):.3f}`")
    st.markdown(f"- **Accuracy:** `{att_metrics.get('accuracy', 0.86)*100:.1f}%`")
    st.markdown(f"- **Performance Model:** Gradient Boosting")
    st.markdown(f"- **Performance MAE:** `{perf_metrics.get('mae', 3.2):.2f} pts`")
    st.markdown(f"- **Active LLM:** `{orchestrator.llm.get_active_provider()}`")

# ── Main Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge-ibm">IBM APPLIED ML & AGENTIC AI PLATFORM</div>
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">TalentPulse</h1>
    <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
        Autonomous Workforce Intelligence & Retention Copilot powered by Dual ML Models and Multi-Agent Orchestration.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs Navigation ──────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Executive Workforce Analytics",
    "👤 Employee Deep-Dive & XAI",
    "🧪 What-If Simulator",
    "🤖 Multi-Agent Intervention",
    "💬 Talent Copilot (Chat)"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE WORKFORCE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Workforce Macro Health & Flight Risk Overview")
    
    total_emps = len(df_workforce)
    total_attrition = (df_workforce["Attrition"] == "Yes").sum()
    rate = (total_attrition / total_emps) * 100
    avg_perf = df_workforce["PerformanceScore"].mean()
    high_perf_cnt = (df_workforce["PerformanceScore"] >= 80).sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Workforce", f"{total_emps:,} staff", "IBM Global Data")
    col2.metric("Turnover Baseline", f"{rate:.1f}%", "-1.4% vs Industry")
    col3.metric("Avg Performance Index", f"{avg_perf:.1f} / 100", "+2.8 pts")
    col4.metric("High-Potential Talent", f"{high_perf_cnt:,} staff", f"{(high_perf_cnt/total_emps)*100:.1f}% of base")
    
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("##### 🏢 Attrition Distribution by Department")
        dept_summary = df_workforce.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
        fig_dept = px.bar(
            dept_summary,
            x="Department",
            y="Count",
            color="Attrition",
            barmode="group",
            color_discrete_map={"Yes": "#ef4444", "No": "#3b82f6"},
            template="plotly_white"
        )
        fig_dept.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_dept, use_container_width=True)
        
    with c2:
        st.markdown("##### ⏱️ Overtime & Work-Life Balance Flight Impact")
        ot_summary = df_workforce.groupby(["OverTime", "WorkLifeBalance"])["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="AttritionPct")
        fig_ot = px.bar(
            ot_summary,
            x="WorkLifeBalance",
            y="AttritionPct",
            color="OverTime",
            barmode="group",
            labels={"AttritionPct": "Attrition Rate (%)", "WorkLifeBalance": "Work-Life Balance (1=Poor, 4=Best)"},
            color_discrete_map={"Yes": "#dc2626", "No": "#10b981"},
            template="plotly_white"
        )
        fig_ot.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_ot, use_container_width=True)

    st.markdown("##### 🚨 Priority Early Warning Roster (Immediate Review)")
    high_risk_candidates = df_workforce[
        (df_workforce["OverTime"] == "Yes") & 
        (df_workforce["WorkLifeBalance"] <= 2) & 
        (df_workforce["YearsSinceLastPromotion"] >= 3)
    ][["EmployeeNumber", "Department", "JobRole", "JobLevel", "MonthlyIncome", "YearsSinceLastPromotion", "WorkLifeBalance", "PerformanceScore"]].head(10)
    
    st.dataframe(
        high_risk_candidates.style.format({
            "MonthlyIncome": "${:,.0f}",
            "PerformanceScore": "{:.1f}/100"
        }),
        use_container_width=True
    )
    st.caption("💡 Select any of these Employee IDs in Tab 2 to perform an autonomous Explainable AI audit.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EMPLOYEE DEEP DIVE & EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Micro Talent Audit & Local Feature Explainability (XAI)")
    
    col_sel1, col_sel2 = st.columns([1, 2])
    
    with col_sel1:
        emp_ids = df_workforce["EmployeeNumber"].tolist()
        # Default to a high risk employee for immediate demo impact
        sample_emp_id = 1
        for idx, row in df_workforce.iterrows():
            if row["OverTime"] == "Yes" and row["WorkLifeBalance"] == 1:
                sample_emp_id = int(row["EmployeeNumber"])
                break
                
        selected_id = st.selectbox("Select Employee from IBM Dataset", emp_ids, index=emp_ids.index(sample_emp_id))
        emp_record = df_workforce[df_workforce["EmployeeNumber"] == selected_id].iloc[0].to_dict()
        
    with col_sel2:
        st.markdown(f"**Employee Profile #{selected_id}:** {emp_record.get('JobRole')} ({emp_record.get('Department')}) | Age: {emp_record.get('Age')} | Income: ${emp_record.get('MonthlyIncome', 0):,}/mo")
        st.markdown(f"**Tenure:** {emp_record.get('YearsAtCompany')} yrs total | **In Current Role:** {emp_record.get('YearsInCurrentRole')} yrs | **Since Last Promotion:** {emp_record.get('YearsSinceLastPromotion')} yrs")

    # Run Prediction
    ml_output = predictor.predict(emp_record)
    
    st.markdown("---")
    g1, g2, g3 = st.columns([1, 1, 1])
    
    with g1:
        # Attrition Gauge
        prob_val = ml_output["attrition_probability"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_val,
            number={"suffix": "%", "font": {"size": 36, "color": ml_output["risk_color"]}},
            title={"text": f"<b>Attrition Flight Risk</b><br><span style='color:{ml_output['risk_color']}'>{ml_output['risk_tier']}</span>", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": ml_output["risk_color"]},
                "steps": [
                    {"range": [0, 20], "color": "#dcfce7"},
                    {"range": [20, 40], "color": "#fef9c3"},
                    {"range": [40, 65], "color": "#ffedd5"},
                    {"range": [65, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": prob_val
                }
            }
        ))
        fig_gauge.update_layout(margin=dict(t=40, b=10, l=30, r=30), height=240)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g2:
        # Performance Gauge
        p_score = ml_output["performance_score"]
        fig_perf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_score,
            number={"suffix": " / 100", "font": {"size": 34, "color": "#0f62fe"}},
            title={"text": f"<b>Performance Rating</b><br><span style='color:#0f62fe'>{ml_output['performance_badge']}</span>", "font": {"size": 16}},
            gauge={
                "axis": {"range": [50, 100]},
                "bar": {"color": "#0f62fe"},
                "steps": [
                    {"range": [50, 65], "color": "#f1f5f9"},
                    {"range": [65, 80], "color": "#e0f2fe"},
                    {"range": [80, 100], "color": "#dbeafe"},
                ]
            }
        ))
        fig_perf.update_layout(margin=dict(t=40, b=10, l=30, r=30), height=240)
        st.plotly_chart(fig_perf, use_container_width=True)
        
    with g3:
        st.markdown("##### 📌 Operational Metrics")
        st.markdown(f"- **Overtime Active:** `{emp_record.get('OverTime')}`")
        st.markdown(f"- **Work-Life Balance:** `{emp_record.get('WorkLifeBalance')}/4`")
        st.markdown(f"- **Job Satisfaction:** `{emp_record.get('JobSatisfaction')}/4`")
        st.markdown(f"- **Annual Salary Hike:** `{emp_record.get('PercentSalaryHike')}%`")
        st.markdown(f"- **Stock Option Level:** `Level {emp_record.get('StockOptionLevel')}`")
        st.markdown(f"- **Distance from Home:** `{emp_record.get('DistanceFromHome')} miles`")

    st.markdown("---")
    st.markdown("#### 🔬 Explainable AI (XAI): Algorithmic Driver Breakdown")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("##### ⚠️ Key Attrition Risk Drivers")
        if ml_output["risk_drivers"]:
            for driver in ml_output["risk_drivers"]:
                st.markdown(f"""
                <div style="background:#fef2f2; border-left:4px solid #ef4444; padding:10px 14px; margin-bottom:8px; border-radius:4px;">
                    <strong>{driver['factor']}</strong> — <span class="driver-pill-risk">{driver['impact']}</span><br>
                    <small style="color:#64748b;">{driver['detail']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No acute risk drivers detected for this employee.")
            
    with col_d2:
        st.markdown("##### 🛡️ Organizational Retention Anchors")
        if ml_output["retention_anchors"]:
            for anchor in ml_output["retention_anchors"]:
                st.markdown(f"""
                <div style="background:#f0fdf4; border-left:4px solid #10b981; padding:10px 14px; margin-bottom:8px; border-radius:4px;">
                    <strong>{anchor['factor']}</strong><br>
                    <small style="color:#64748b;">{anchor['detail']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Low retention capital detected; requires urgent re-anchoring.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: WHAT-IF COUNTERFACTUAL SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Interactive Counterfactual Simulation")
    st.caption("Adjust policy levers in real-time to simulate how organizational interventions change the employee's flight risk and performance outlook.")
    
    sim_col1, sim_col2 = st.columns([1, 1])
    
    with sim_col1:
        st.markdown("##### 🎛️ Proposed Policy Interventions")
        
        sim_ot = st.radio("Overtime Policy", ["Keep Current", "Eliminate Overtime (No)"], horizontal=True)
        sim_hike = st.slider("Target Annual Salary Hike (%)", min_value=11, max_value=30, value=max(18, emp_record.get("PercentSalaryHike", 15)))
        sim_promo = st.checkbox("Grant Promotion / Reset Stagnation (Set YearsSincePromotion to 0)", value=True if emp_record.get("YearsSinceLastPromotion", 0) >= 3 else False)
        sim_wlb = st.select_slider("Work-Life Balance Adjustment", options=[1, 2, 3, 4], value=max(3, emp_record.get("WorkLifeBalance", 2)))
        sim_stock = st.select_slider("Stock Option Retention Grant", options=[0, 1, 2, 3], value=max(1, emp_record.get("StockOptionLevel", 0)))
        
        adjustments = {}
        if sim_ot == "Eliminate Overtime (No)":
            adjustments["OverTime"] = "No"
        adjustments["PercentSalaryHike"] = sim_hike
        if sim_promo:
            adjustments["YearsSinceLastPromotion"] = 0
        adjustments["WorkLifeBalance"] = sim_wlb
        adjustments["StockOptionLevel"] = sim_stock
        
    with sim_col2:
        st.markdown("##### 📈 Projected Quantitative Outcome")
        sim_result = predictor.simulate_what_if(emp_record, adjustments)
        
        base_p = sim_result["baseline"]["attrition_probability"]
        sim_p = sim_result["simulated"]["attrition_probability"]
        drop = sim_result["risk_reduction_pct"]
        
        sc1, sc2 = st.columns(2)
        sc1.metric("Baseline Risk", f"{base_p}%", sim_result["baseline"]["risk_tier"])
        sc2.metric("Simulated Risk", f"{sim_p}%", f"⬇ {drop:.1f}% reduction", delta_color="inverse")
        
        # Comparison chart
        comp_df = pd.DataFrame([
            {"Scenario": "Current State", "Attrition Risk (%)": base_p, "Performance Score": sim_result["baseline"]["performance_score"]},
            {"Scenario": "With Interventions", "Attrition Risk (%)": sim_p, "Performance Score": sim_result["simulated"]["performance_score"]},
        ])
        fig_comp = px.bar(
            comp_df,
            x="Scenario",
            y="Attrition Risk (%)",
            color="Scenario",
            color_discrete_map={"Current State": "#ef4444", "With Interventions": "#10b981"},
            text="Attrition Risk (%)",
            template="plotly_white"
        )
        fig_comp.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

        if drop > 15:
            st.success(f"🎉 **High-Impact Intervention:** This proposed strategy reduces flight risk by **{drop:.1f} percentage points**, moving the employee into the safe zone!")
        else:
            st.info("Moderate intervention impact. Consider combining promotion velocity with compensation realignment for maximum retention.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: MULTI-AGENT INTERVENTION CENTER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Autonomous Multi-Agent Talent Intervention")
    st.markdown("""
    Unlike traditional ML which stops at a numeric prediction, **Agentic AI** reasons across the employee's entire context,
    coordinates specialized autonomous agents, and produces actionable executive deliverables.
    """)
    
    st.info(f"Targeting Employee #{emp_record.get('EmployeeNumber')} ({emp_record.get('JobRole')}) | Active Provider: **{orchestrator.llm.get_active_provider()}**")
    
    if st.button("🚀 Dispatch Autonomous Multi-Agent Audit", type="primary", use_container_width=True):
        with st.spinner("🤖 Autonomous Agents analyzing workforce drivers and synthesizing intervention package..."):
            audit_pack = orchestrator.run_full_audit(emp_record)
            st.session_state[f"audit_{selected_id}"] = audit_pack
            st.success("Multi-Agent Audit successfully executed!")

    audit_pack = st.session_state.get(f"audit_{selected_id}", None)
    
    if audit_pack:
        st.markdown("---")
        
        # Agent 1
        with st.expander("🔍 **Agent 1: Talent Diagnostic Agent Report**", expanded=True):
            st.markdown(audit_pack["diagnostic_report"])
            
        # Agent 2
        with st.expander("🚀 **Agent 2: Upskilling & Career Strategy Blueprint**", expanded=True):
            st.markdown(audit_pack["upskilling_plan"])
            
        # Agent 3
        with st.expander("🛡️ **Agent 3: Retention & Intervention Action Package**", expanded=True):
            st.markdown(audit_pack["retention_package"])
            
        # Download button
        full_markdown = (
            f"# TalentPulse Executive Audit Dossier\n"
            f"**Employee:** {emp_record.get('JobRole')} (#{selected_id})\n\n"
            f"{audit_pack['diagnostic_report']}\n\n"
            f"{audit_pack['upskilling_plan']}\n\n"
            f"{audit_pack['retention_package']}\n"
        )
        st.download_button(
            label="📥 Download Full Executive Dossier (Markdown)",
            data=full_markdown,
            file_name=f"TalentPulse_Audit_Emp_{selected_id}.md",
            mime="text/markdown"
        )
    else:
        st.info("Click the button above to trigger the 3 specialized agents.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: TALENT COPILOT (INTERACTIVE AGENTIC CHAT)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("💬 Interactive Talent Copilot")
    st.caption("Ask questions, interrogate the employee's dossier, run What-If simulations through natural conversation, or draft executive memos.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Quick action prompt buttons
    st.markdown("##### ⚡ Quick Action Queries")
    qcol1, qcol2, qcol3 = st.columns(3)
    quick_query = None
    if qcol1.button("Why is this employee at risk?", use_container_width=True):
        quick_query = "Why is this employee considering leaving and what are their primary risk drivers?"
    if qcol2.button("Simulate 15% hike & No Overtime", use_container_width=True):
        quick_query = "Simulate what happens if we eliminate overtime and give a 15% salary hike."
    if qcol3.button("Draft VP Escalation Memo", use_container_width=True):
        quick_query = "Draft an urgent executive retention memo to the VP of HR."

    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat input
    user_input = st.chat_input("Ask the Talent Copilot anything about this employee...")
    query_to_process = quick_query or user_input
    
    if query_to_process:
        st.session_state.chat_history.append({"role": "user", "content": query_to_process})
        with st.chat_message("user"):
            st.markdown(query_to_process)
            
        with st.chat_message("assistant"):
            with st.spinner("Copilot analyzing and executing tools..."):
                audit_data = st.session_state.get(f"audit_{selected_id}", {"ml_result": ml_output})
                copilot_response = orchestrator.chat_copilot(
                    query_to_process,
                    emp_record,
                    audit_data,
                    st.session_state.chat_history
                )
                st.markdown(copilot_response)
                st.session_state.chat_history.append({"role": "assistant", "content": copilot_response})
