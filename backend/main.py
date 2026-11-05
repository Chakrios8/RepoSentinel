"""RepoSentinel FastAPI application and endpoint routing."""

from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.ml.feature_extractor import extract_ast_features
from backend.ml.train_triage import train_and_save_model, FEATURE_COLUMNS
from backend.rag.vector_store import get_vector_store
from backend.agent.agent_engine import get_agent_runner

# Global ML model and feature columns cache
_triage_model = None
_feature_columns: List[str] = FEATURE_COLUMNS


def load_triage_model():
    """Load or automatically train the XGBoost triage model."""
    global _triage_model, _feature_columns
    settings = get_settings()
    model_path = Path(settings.MODEL_PATH)
    columns_path = Path(settings.FEATURE_COLUMNS_PATH)

    if not model_path.exists() or not columns_path.exists():
        print(f"Model artifacts not found at {model_path}. Training new model...")
        model, _ = train_and_save_model(
            model_output_path=str(model_path),
            columns_output_path=str(columns_path)
        )
        _triage_model = model
    else:
        _triage_model = joblib.load(model_path)
        with open(columns_path, "r", encoding="utf-8") as f:
            _feature_columns = json.load(f)

    return _triage_model


def get_triage_tier(score: float) -> str:
    """Map risk probability score [0.0, 1.0] to descriptive risk tiers."""
    if score < 0.25:
        return "LOW"
    elif score < 0.50:
        return "MEDIUM"
    elif score < 0.75:
        return "HIGH"
    return "CRITICAL"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to initialize models and vector store on startup."""
    print("Initializing RepoSentinel engine...")
    load_triage_model()
    get_vector_store()
    get_agent_runner()
    print("RepoSentinel engine initialized successfully.")
    yield
    print("Shutting down RepoSentinel.")


settings = get_settings()
app = FastAPI(
    title="RepoSentinel API",
    description="Autonomous Code Review & Security Remediation Engine combining AST parsing, XGBoost triage, RAG, and ReAct agent.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS middleware for developer dashboards or integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response schemas
class AuditRequest(BaseModel):
    file_path: str = Field(default="snippet.py", description="Target file path or commit diff identifier")
    code: str = Field(..., description="Python source code snippet or commit diff to audit")

    model_config = {
        "json_schema_extra": {
            "example": {
                "file_path": "auth/user_service.py",
                "code": "def get_user(uid):\n    query = 'SELECT * FROM users WHERE id = ' + uid\n    return cursor.execute(query)\n"
            }
        }
    }


class AuditResponse(BaseModel):
    file_path: str
    triage_risk_score: float
    triage_tier: str
    llm_audit_triggered: bool
    ast_metrics: Dict[str, Any]
    remediation_report: Optional[str] = None
    execution_time_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    vector_store_ready: bool
    api_key_configured: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint reporting system component readiness."""
    global _triage_model
    store_ready = False
    try:
        store = get_vector_store()
        store_ready = store.collection.count() >= 0
    except Exception:
        store_ready = False

    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        model_loaded=_triage_model is not None,
        vector_store_ready=store_ready,
        api_key_configured=settings.is_api_key_configured,
    )


@app.post("/api/v1/audit", response_model=AuditResponse)
def audit_code(payload: AuditRequest):
    """Audit source code using 2-tier architecture: AST + XGBoost triage followed by conditional LangChain agent."""
    start_time = time.perf_counter()

    if not payload.code or not payload.code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'code' parameter cannot be empty."
        )

    # Step 1: Static AST Feature Extraction (< 2ms)
    metrics = extract_ast_features(payload.code)

    # Step 2: XGBoost Triage Evaluation (< 2ms)
    global _triage_model, _feature_columns
    if _triage_model is None:
        load_triage_model()

    # Align extracted features with training feature columns
    feature_row = pd.DataFrame([{col: metrics.get(col, 0) for col in _feature_columns}])
    risk_probabilities = _triage_model.predict_proba(feature_row)[0]
    # Probability of high-risk class (index 1)
    risk_score = float(risk_probabilities[1]) if len(risk_probabilities) > 1 else float(risk_probabilities[0])
    risk_score = round(risk_score, 4)

    tier = get_triage_tier(risk_score)
    threshold = settings.TRIAGE_THRESHOLD

    # Tier 1 Bypass: If score < threshold, bypass LLM completely (optimizes cost and latency)
    if risk_score < threshold:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return AuditResponse(
            file_path=payload.file_path,
            triage_risk_score=risk_score,
            triage_tier=tier,
            llm_audit_triggered=False,
            ast_metrics=metrics,
            remediation_report="AST triage classified commit as low risk. LLM audit bypassed for cost and latency optimization.",
            execution_time_ms=elapsed_ms,
        )

    # Tier 2: Score >= threshold -> Trigger ReAct remediation agent
    agent_runner = get_agent_runner()
    remediation_result = agent_runner.run_remediation(
        code=payload.code,
        file_path=payload.file_path,
        ast_metrics=metrics
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return AuditResponse(
        file_path=payload.file_path,
        triage_risk_score=risk_score,
        triage_tier=tier,
        llm_audit_triggered=True,
        ast_metrics=metrics,
        remediation_report=remediation_result.get("remediation_report"),
        execution_time_ms=elapsed_ms,
    )
