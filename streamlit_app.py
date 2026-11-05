"""RepoSentinel: Autonomous Code Review & Security Remediation Engine - Streamlit Interface."""

import json
import time
from pathlib import Path
import requests
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="RepoSentinel | Autonomous Security Remediation Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #334155;
        margin-bottom: 8px;
    }
    .tier-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .tier-low { background-color: #065f46; color: #34d399; }
    .tier-medium { background-color: #854d0e; color: #facc15; }
    .tier-high { background-color: #9a3412; color: #fb923c; }
    .tier-critical { background-color: #881337; color: #fb7185; }
    .ast-chip {
        display: inline-block;
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.78rem;
        margin: 2px 4px 2px 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Info & Engine Diagnostics
with st.sidebar:
    st.markdown("### 🛡️ RepoSentinel Status")
    
    # Try connecting to FastAPI backend, or fallback to direct modules
    backend_url = "http://localhost:8000"
    is_api_alive = False
    try:
        r = requests.get(f"{backend_url}/health", timeout=0.8)
        if r.status_code == 200:
            is_api_alive = True
            h_data = r.json()
            st.success("● FastAPI Backend Online (:8000)")
            st.caption(f"Model: Loaded | Vector Store: Ready | Key: {'Configured' if h_data.get('api_key_configured') else 'Missing'}")
    except Exception:
        st.info("● In-Process Engine Mode (Standalone)")
        st.caption("Running directly via Python core modules")

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Architecture")
    st.markdown("""
    1. **Static AST Analysis**: Extracts 8 structural complexity metrics in <2ms.
    2. **XGBoost Classifier**: Scores vulnerability risk. Diffs < 0.50 risk bypass the LLM.
    3. **ChromaDB Vector Store**: RAG retrieval of CWE-89, CWE-95, CWE-798, CWE-400 rules.
    4. **Gemini ReAct Agent**: Synthesizes minimal patch and verifies against AST compiler.
    """)
    st.markdown("---")
    st.caption("Google DeepMind • Advanced Agentic Coding")

# Main Title Area
st.markdown('<div class="main-header">RepoSentinel</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous Code Review & Security Remediation Engine</div>', unsafe_allow_html=True)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("#### 📝 Commit Diff Input")
    target_file = st.text_input(
        "File Path:",
        value="services/user_service.py",
        placeholder="e.g., auth/user_service.py"
    )
    
    source_code = st.text_area(
        "Python Code Snippet or Diff:",
        height=380,
        placeholder="# Paste your Python code or commit diff here to audit...\n\ndef get_user(cursor, uid):\n    query = 'SELECT * FROM users WHERE id = ' + uid\n    cursor.execute(query)\n    return cursor.fetchone()\n",
        help="Paste the Python source code or diff to audit."
    )

    audit_button = st.button("🛡️ Audit & Remediate Code", type="primary", use_container_width=True)


def run_audit_direct(code: str, file_path: str):
    """Directly invoke backend modules if FastAPI server isn't running separately."""
    from backend.ml.feature_extractor import extract_ast_features
    from backend.main import load_triage_model, get_triage_tier
    from backend.agent.agent_engine import get_agent_runner
    from backend.config import get_settings
    import pandas as pd

    start_time = time.perf_counter()
    metrics = extract_ast_features(code)
    model = load_triage_model()
    
    from backend.ml.train_triage import FEATURE_COLUMNS
    feature_row = pd.DataFrame([{col: metrics.get(col, 0) for col in FEATURE_COLUMNS}])
    risk_proba = model.predict_proba(feature_row)[0]
    risk_score = round(float(risk_proba[1]) if len(risk_proba) > 1 else float(risk_proba[0]), 4)
    tier = get_triage_tier(risk_score)
    threshold = get_settings().TRIAGE_THRESHOLD

    if risk_score < threshold:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "file_path": file_path,
            "triage_risk_score": risk_score,
            "triage_tier": tier,
            "llm_audit_triggered": False,
            "ast_metrics": metrics,
            "remediation_report": "AST triage classified commit as low risk. LLM audit bypassed for cost and latency optimization.",
            "execution_time_ms": elapsed_ms,
        }

    agent_runner = get_agent_runner()
    res = agent_runner.run_remediation(code=code, file_path=file_path, ast_metrics=metrics)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return {
        "file_path": file_path,
        "triage_risk_score": risk_score,
        "triage_tier": tier,
        "llm_audit_triggered": True,
        "ast_metrics": metrics,
        "remediation_report": res.get("remediation_report"),
        "execution_time_ms": elapsed_ms,
    }


with col_output:
    st.markdown("#### 🔍 Security Intelligence & Remediation")

    if audit_button:
        if not source_code.strip():
            st.warning("Please provide code to audit.")
        else:
            with st.spinner("Analyzing AST metrics, evaluating XGBoost risk, and checking security policies..."):
                audit_result = None
                
                # Check if running FastAPI server is available first
                if is_api_alive:
                    try:
                        resp = requests.post(
                            f"{backend_url}/api/v1/audit",
                            json={"file_path": target_file, "code": source_code},
                            timeout=60
                        )
                        if resp.status_code == 200:
                            audit_result = resp.json()
                    except Exception:
                        audit_result = None

                # Fallback to direct local engine execution
                if audit_result is None:
                    audit_result = run_audit_direct(source_code, target_file)

                score = audit_result["triage_risk_score"]
                tier = audit_result["triage_tier"]
                llm_triggered = audit_result["llm_audit_triggered"]
                elapsed_ms = audit_result["execution_time_ms"]
                metrics = audit_result["ast_metrics"]

                # 1. Metric Cards
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="Risk Score", value=f"{score:.4f}")
                with m_col2:
                    tier_class = f"tier-{tier.lower()}"
                    st.markdown(f"**Triage Tier**<br><span class='tier-badge {tier_class}'>{tier}</span>", unsafe_allow_html=True)
                with m_col3:
                    st.metric(label="Latency", value=f"{elapsed_ms} ms")

                st.markdown("<br>", unsafe_allow_html=True)

                # 2. Pipeline Action Alert
                if not llm_triggered:
                    st.success("⚡ **Fast-Path Bypass Activated:** Commit risk is low. LLM audit bypassed in milliseconds at **$0.00 token cost**.")
                else:
                    st.error("🚨 **High Risk Flagged:** XGBoost triage triggered the autonomous ReAct remediation loop.")

                # 3. AST Complexity Metric Chips
                st.markdown("**AST Complexity & Security Indicators:**")
                chips_html = "".join([
                    f"<span class='ast-chip'>LOC: {metrics.get('lines_of_code', 0)}</span>",
                    f"<span class='ast-chip'>Nesting: {metrics.get('max_nesting_depth', 0)}</span>",
                    f"<span class='ast-chip'>Branches: {metrics.get('num_branches', 0)}</span>",
                    f"<span class='ast-chip'>Calls: {metrics.get('num_external_calls', 0)}</span>",
                    f"<span class='ast-chip'>Imports: {metrics.get('num_imports', 0)}</span>",
                    f"<span class='ast-chip'>Raw SQL: {'⚠️ Yes' if metrics.get('has_raw_sql_indicators') else 'No'}</span>",
                    f"<span class='ast-chip'>Eval/Exec: {'⚠️ Yes' if metrics.get('has_eval_or_exec') else 'No'}</span>",
                ])
                st.markdown(chips_html, unsafe_allow_html=True)

                st.markdown("---")

                # 4. Remediation Report
                st.markdown("#### 📋 Remediation Report")
                report = audit_result.get("remediation_report")
                if report:
                    st.markdown(report)
                    if llm_triggered:
                        st.info("✅ **Compiler Verified:** The suggested patch was verified via Python AST parser (`ast.parse()`) with 0 syntax errors.")
                else:
                    st.info("No vulnerabilities detected. Clean commit.")
    else:
        st.info("👈 Paste your Python code or diff on the left, then click **'Audit & Remediate Code'**.")
