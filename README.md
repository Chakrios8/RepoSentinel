# RepoSentinel: Autonomous Code Review & Security Remediation Engine

RepoSentinel is an intelligent, production-grade code auditing and automated remediation system designed to solve the cost, latency, and hallucination bottlenecks of LLM-only code reviews. By coupling **Static AST Parsing** and an **XGBoost ML Triage Classifier** with a **ChromaDB Vector Store (RAG)** and a **LangChain ReAct Agent powered by Google Gemini**, RepoSentinel filters benign diffs in sub-15ms and autonomously generates syntactically validated security patches for high-risk vulnerabilities.

---

## Architecture Diagram

```
                              +---------------------------+
                              |   Incoming Code Commit    |
                              |   (POST /api/v1/audit)    |
                              +-------------+-------------+
                                            |
                                            v
                              +---------------------------+
                              |   AST Feature Extractor   |
                              |  (8 Structural Metrics)   |
                              +-------------+-------------+
                                            |
                                            v
                              +---------------------------+
                              |  XGBoost Triage Engine    |
                              |    (Risk Score [0, 1])    |
                              +-------------+-------------+
                                            |
                       +--------------------+--------------------+
                       |                                         |
            [Risk Score < 0.50]                       [Risk Score >= 0.50]
                       |                                         |
                       v                                         v
          +-------------------------+               +-------------------------+
          |     Fast-Path Bypass    |               |  ChromaDB Security KB   |
          |  - Latency: < 15ms      |               |  (CWE/CVE Guidelines)   |
          |  - LLM Cost: $0.00      |               +------------+------------+
          |  - Auto-Approved        |                            |
          +-------------------------+                            v
                                                    +-------------------------+
                                                    |  LangChain ReAct Agent  |
                                                    |   (Google Gemini 1.5)   |
                                                    +------------+------------+
                                                                 |
                                                                 v
                                                    +-------------------------+
                                                    |   Compiler AST Tool     |
                                                    | (validate_code_syntax)  |
                                                    +------------+------------+
                                                                 |
                                                [Syntax Error?]  |  [Valid AST]
                                                +----------------+       |
                                                |  (Self-Heal Loop)      v
                                                +---------------->  Final JSON Patch
```

---

## Core Capabilities

1. **Static AST Complexity & Threat Extraction (`ASTMetricsVisitor`)**:
   Parses Python code into an Abstract Syntax Tree to extract 8 metrics without executing untrusted code:
   - `lines_of_code`, `num_functions`, `max_nesting_depth`, `num_branches`, `num_imports`, `num_external_calls`, `has_eval_or_exec`, `has_raw_sql_indicators`.

2. **Ultra-Low Latency Machine Learning Triage (`XGBoost`)**:
   Evaluates commit risk probability in < 2ms. Commits below the 0.50 risk threshold bypass the LLM entirely, saving > 70% in cloud LLM inference tokens.

3. **2-Stage RAG Security Knowledge Base (`ChromaDB` + `all-MiniLM-L6-v2`)**:
   Provides semantic similarity lookups across CWE/CVE definitions (SQL Injection CWE-89, Remote Code Execution CWE-95, Hardcoded Secrets CWE-798, ReDoS CWE-400).

4. **ReAct Self-Correcting Remediation Agent (`Google Gemini`)**:
   Equipped with `query_security_kb` and `validate_code_syntax`. Before emitting a patch, the agent compiles candidate patches through Python's `ast.parse()`, repairing any syntax errors autonomously in an internal feedback loop.

5. **Production FastAPI Service & Benchmark Suite**:
   Provides `/api/v1/audit` and `/health` endpoints with full Pydantic validation and comprehensive integration tests (`tests/test_audit.py`).

---

## Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository and navigate to root
cd RepoSentinel

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and set your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=AIzaSyYourActualGeminiApiKeyHere
GEMINI_MODEL=gemini-1.5-flash
HOST=0.0.0.0
PORT=8000
TRIAGE_THRESHOLD=0.50
```

> **Note**: Even if you haven't filled in the `GOOGLE_API_KEY` yet, RepoSentinel will start, train the triage classifier, seed the vector database, process low-risk diffs in sub-15ms, and return ChromaDB-grounded security policies for high-risk diffs.

### 3. Launch the Application

Run the single bootstrap entry-point script:

```bash
python run.py
```

This will automatically:
1. Generate synthetic AST commit data and train `backend/ml/ast_triage_model.joblib`.
2. Seed the ChromaDB collection `sec_guidelines` with `seed_rules.json`.
3. Launch Uvicorn on `http://0.0.0.0:8000`.

---

## API Documentation & Curl Commands

### Health Check

```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "RepoSentinel",
  "version": "1.0.0",
  "model_loaded": true,
  "vector_store_ready": true,
  "api_key_configured": true
}
```

---

### Audit Case A: Benign Addition Function (Bypasses LLM in Sub-15ms)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/audit \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "math_utils.py",
    "code": "def add(a: int, b: int) -> int:\n    return a + b\n"
  }'
```

**Response**:
```json
{
  "file_path": "math_utils.py",
  "triage_risk_score": 0.0215,
  "triage_tier": "LOW",
  "llm_audit_triggered": false,
  "ast_metrics": {
    "lines_of_code": 2,
    "num_functions": 1,
    "max_nesting_depth": 1,
    "num_branches": 0,
    "num_imports": 0,
    "num_external_calls": 0,
    "has_eval_or_exec": 0,
    "has_raw_sql_indicators": 0
  },
  "remediation_report": "AST triage classified commit as low risk. LLM audit bypassed for cost and latency optimization.",
  "execution_time_ms": 3.42
}
```

---

### Audit Case B: SQL Injection (Triggers ReAct Agent & AST Validation)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/audit \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "services/user_service.py",
    "code": "def get_user_profile(cursor, uid: str):\n    query = \"SELECT * FROM users WHERE id = \" + uid\n    cursor.execute(query)\n    return cursor.fetchone()\n"
  }'
```

**Response**:
```json
{
  "file_path": "services/user_service.py",
  "triage_risk_score": 0.9421,
  "triage_tier": "CRITICAL",
  "llm_audit_triggered": true,
  "ast_metrics": {
    "lines_of_code": 4,
    "num_functions": 1,
    "max_nesting_depth": 1,
    "num_branches": 0,
    "num_imports": 0,
    "num_external_calls": 2,
    "has_eval_or_exec": 0,
    "has_raw_sql_indicators": 1
  },
  "remediation_report": "### Vulnerability Identified: CWE-89 (SQL Injection)\n\n**Root Cause:** Dynamic string concatenation in SQL query (`\"SELECT * FROM users WHERE id = \" + uid`) executed via `cursor.execute()`.\n\n**Verified Secure Patch:**\n```python\ndef get_user_profile(cursor, uid: str):\n    query = \"SELECT * FROM users WHERE id = %s\"\n    cursor.execute(query, (uid,))\n    return cursor.fetchone()\n```\n\n**AST Verification:** Candidate code verified via `validate_code_syntax` (AST parsed cleanly with 0 syntax errors).",
  "execution_time_ms": 1142.60
}
```

---

## Running Automated Tests & Benchmarks

Run the integration benchmark suite:

```bash
pytest tests/test_audit.py -v -s
```

Or execute directly with the Python runtime:

```bash
python tests/test_audit.py
```

---

## Repository Structure

```
reposentinel/
├── .env.example
├── .gitignore
├── requirements.txt
├── run.py                 # Single entry-point script to verify/train models, seed vector DB, and launch the API
├── explanation.txt        # Deep-dive file-by-file, function-by-function architecture explanation
├── README.md              # Project documentation with quickstart, architecture diagram, and curl commands
├── backend/
│   ├── __init__.py
│   ├── main.py            # FastAPI app and endpoint routing
│   ├── config.py          # Environment configuration (Pydantic / dotenv)
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py # Python AST NodeVisitor extracting 8 structural complexity metrics
│   │   └── train_triage.py      # Generates synthetic commit AST dataset, trains XGBoost, saves model artifacts
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vector_store.py      # ChromaDB client initialization and query logic
│   │   └── seed_rules.json      # Structured CVE/CWE security policies (SQLi, RCE, Secrets, ReDoS)
│   └── agent/
│       ├── __init__.py
│       ├── tools.py             # Custom LangChain tools (@tool) for RAG querying and AST syntax checking
│       └── agent_engine.py      # create_tool_calling_agent and AgentExecutor configuration
└── tests/
    ├── __init__.py
    └── test_audit.py      # Comprehensive tests sending low-risk and high-risk payloads to the API
```
