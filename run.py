"""Single entry-point script to verify/train models, seed vector DB, and launch the RepoSentinel API."""

import os
import sys
from pathlib import Path
import uvicorn

# Add current workspace directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import get_settings
from backend.ml.train_triage import train_and_save_model
from backend.rag.vector_store import get_vector_store


def bootstrap_environment():
    """Verify and initialize machine learning models and vector store."""
    settings = get_settings()
    print("=" * 60)
    print("      RepoSentinel: Autonomous Code Review Engine       ")
    print("=" * 60)

    # 1. Verify / Train ML Triage Model
    model_path = Path(settings.MODEL_PATH)
    columns_path = Path(settings.FEATURE_COLUMNS_PATH)

    if not model_path.exists() or not columns_path.exists():
        print(f"[*] ML model artifacts missing at '{model_path}'. Initiating synthetic dataset generation & training...")
        train_and_save_model(
            model_output_path=str(model_path),
            columns_output_path=str(columns_path)
        )
        print("[+] ML triage model trained and saved successfully.")
    else:
        print(f"[+] Found pre-trained ML triage model at '{model_path}'.")

    # 2. Verify ChromaDB Collection Seeding
    print("[*] Initializing ChromaDB vector store and verifying security policies...")
    vector_store = get_vector_store()
    count = vector_store.collection.count()
    print(f"[+] ChromaDB ready. Active security rules in vector store: {count}")

    # 3. Check Gemini API key status
    if settings.is_api_key_configured:
        print(f"[+] Google Gemini LLM agent enabled with model '{settings.GEMINI_MODEL}'.")
    else:
        print("[!] Note: GOOGLE_API_KEY is not configured in .env.")
        print("    High-risk commits will trigger RAG-grounded security advisories.")
        print("    Set your GOOGLE_API_KEY in .env to enable the autonomous LLM remediation loop.")

    print("=" * 60)


def main():
    """Run bootstrap and start Uvicorn server."""
    bootstrap_environment()
    settings = get_settings()

    print(f"[*] Starting RepoSentinel API on http://{settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
