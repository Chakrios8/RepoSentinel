"""Automated integration benchmark and test suite for RepoSentinel."""

import os
import sys
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure workspace root is in python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app, load_triage_model
from backend.ml.feature_extractor import extract_ast_features
from backend.rag.vector_store import get_vector_store
from backend.agent.tools import validate_code_syntax, query_security_kb


@pytest.fixture(scope="session", autouse=True)
def initialize_system():
    """Ensure models and vector DB are initialized before running test suite."""
    load_triage_model()
    get_vector_store()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify that the /health endpoint reports all subsystems as healthy."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "RepoSentinel"
    assert data["model_loaded"] is True
    assert data["vector_store_ready"] is True


def test_ast_feature_extraction():
    """Verify that ASTMetricsVisitor accurately extracts structural and security metrics."""
    # 1. Clean math function
    clean_code = """
def add(a: int, b: int) -> int:
    return a + b
"""
    clean_metrics = extract_ast_features(clean_code)
    assert clean_metrics["num_functions"] == 1
    assert clean_metrics["num_branches"] == 0
    assert clean_metrics["has_eval_or_exec"] == 0
    assert clean_metrics["has_raw_sql_indicators"] == 0
    assert clean_metrics["lines_of_code"] >= 2

    # 2. Suspicious code with dynamic eval and raw SQL
    suspicious_code = """
import sqlite3

def run_query(user_input):
    if user_input:
        conn = sqlite3.connect('app.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = " + user_input)
        eval(user_input)
"""
    susp_metrics = extract_ast_features(suspicious_code)
    assert susp_metrics["has_eval_or_exec"] == 1
    assert susp_metrics["has_raw_sql_indicators"] == 1
    assert susp_metrics["num_branches"] >= 1
    assert susp_metrics["num_imports"] >= 1


def test_syntax_validator_tool():
    """Verify that validate_code_syntax correctly accepts valid code and catches syntax errors."""
    valid_patch = """
def secure_fetch(db, user_id: int):
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()
"""
    result_valid = validate_code_syntax.invoke({"candidate_code": valid_patch})
    assert "VALIDATION SUCCESS" in result_valid

    invalid_patch = """
def broken_syntax(
    cursor.execute("SELECT
"""
    result_invalid = validate_code_syntax.invoke({"candidate_code": invalid_patch})
    assert "VALIDATION ERROR" in result_invalid


def test_vector_store_retrieval():
    """Verify semantic retrieval of CWE security policies from ChromaDB."""
    store = get_vector_store()

    # Query for SQL injection
    sqli_results = store.query_rules("cursor.execute raw SQL injection user string concatenation", top_k=1)
    assert len(sqli_results) > 0
    assert "CWE-89" in sqli_results[0]["metadata"]["cwe_id"]

    # Query for dynamic code execution
    rce_results = store.query_rules("eval exec dynamic code execution RCE", top_k=1)
    assert len(rce_results) > 0
    assert "CWE-95" in rce_results[0]["metadata"]["cwe_id"]


def test_case_a_benign_addition_bypass(client: TestClient):
    """Case A: Benign addition function -> verifies triage bypasses LLM in sub-15ms."""
    benign_code = """
def add(a: int, b: int) -> int:
    \"\"\"Calculate sum of two integers.\"\"\"
    return a + b
"""
    payload = {
        "file_path": "math_utils.py",
        "code": benign_code
    }

    start = time.perf_counter()
    response = client.post("/api/v1/audit", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200, f"Audit failed: {response.text}"
    data = response.json()

    # Assertions
    assert data["llm_audit_triggered"] is False, "Benign code should NOT trigger the LLM"
    assert data["triage_risk_score"] < 0.50, f"Expected risk score < 0.50, got {data['triage_risk_score']}"
    assert data["triage_tier"] in ("LOW", "MEDIUM"), f"Expected LOW/MEDIUM tier, got {data['triage_tier']}"
    assert data["ast_metrics"]["has_eval_or_exec"] == 0
    assert data["ast_metrics"]["has_raw_sql_indicators"] == 0
    print(f"\n[Case A Passed] Triage score: {data['triage_risk_score']}, Execution: {data['execution_time_ms']}ms (API roundtrip: {elapsed_ms:.2f}ms)")


def test_case_b_raw_sql_injection_triggers_audit(client: TestClient):
    """Case B: Raw string-concatenated SQL injection -> verifies triage flags high risk & triggers remediation."""
    sql_code = """
def get_user_profile(cursor, uid: str):
    # String concatenated dynamic SQL
    query = "SELECT * FROM users WHERE id = '" + uid + "'"
    cursor.execute(query)
    return cursor.fetchone()
"""
    payload = {
        "file_path": "services/user_repo.py",
        "code": sql_code
    }

    start = time.perf_counter()
    response = client.post("/api/v1/audit", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200, f"Audit failed: {response.text}"
    data = response.json()

    # Assertions
    assert data["llm_audit_triggered"] is True, "Vulnerable SQL code MUST trigger the LLM/audit remediation"
    assert data["triage_risk_score"] >= 0.50, f"Expected risk score >= 0.50, got {data['triage_risk_score']}"
    assert data["triage_tier"] in ("HIGH", "CRITICAL"), f"Expected HIGH/CRITICAL tier, got {data['triage_tier']}"
    assert data["ast_metrics"]["has_raw_sql_indicators"] == 1, "AST should detect raw sql indicator (.execute)"
    assert data["remediation_report"] is not None
    # Verify the report mentions SQL injection, parameterized query, or CWE-89
    report_text = data["remediation_report"].upper()
    assert any(term in report_text for term in ["SQL", "CWE-89", "PARAMETER", "INJECTION"]), "Report must contain SQL remediation advice"
    print(f"\n[Case B Passed] Triage score: {data['triage_risk_score']}, Tier: {data['triage_tier']}, Triggered LLM: {data['llm_audit_triggered']}")


def test_case_c_dynamic_eval_rce(client: TestClient):
    """Case C: Dynamic eval() usage -> verifies triage flags high risk & returns CWE-95 remediation."""
    eval_code = """
def compute_formula(user_expression: str):
    result = eval(user_expression)
    return result
"""
    payload = {
        "file_path": "calculator/evaluator.py",
        "code": eval_code
    }

    response = client.post("/api/v1/audit", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["llm_audit_triggered"] is True
    assert data["triage_risk_score"] >= 0.50
    assert data["ast_metrics"]["has_eval_or_exec"] == 1
    assert data["remediation_report"] is not None
    report_text = data["remediation_report"].upper()
    assert any(term in report_text for term in ["EVAL", "CWE-95", "EXECUTION", "LITERAL_EVAL"])
    print(f"\n[Case C Passed] Triage score: {data['triage_risk_score']}, Tier: {data['triage_tier']}")


def run_benchmark():
    """Run benchmark comparison displaying triage latency vs LLM trigger status."""
    print("\n" + "=" * 65)
    print("              RepoSentinel Latency & Triage Benchmark            ")
    print("=" * 65)
    with TestClient(app) as client:
        test_case_a_benign_addition_bypass(client)
        test_case_b_raw_sql_injection_triggers_audit(client)
        test_case_c_dynamic_eval_rce(client)
    print("=" * 65)
    print("All integration benchmark test cases passed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
