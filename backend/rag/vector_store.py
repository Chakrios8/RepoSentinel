"""ChromaDB vector store initialization, rule ingestion, and semantic retrieval for RepoSentinel."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from backend.config import get_settings


class SecurityVectorStore:
    """Manages ChromaDB collection containing security policies, CWE rules, and remediation patterns."""

    COLLECTION_NAME = "sec_guidelines"

    def __init__(self, persist_dir: Optional[str] = None, seed_path: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.persist_dir = Path(persist_dir or self.settings.CHROMA_PERSIST_DIR)
        self.seed_path = Path(seed_path or self.settings.SEED_RULES_PATH)

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SentenceTransformer embedding function (all-MiniLM-L6-v2)
        try:
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception:
            # Fallback to default Chroma embedding function if model download or initialization encounters issues
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # Ensure seed rules are ingested
        self._ensure_seed_rules_ingested()

    def _ensure_seed_rules_ingested(self) -> None:
        """Seed the vector store with initial CWE rules if collection is currently empty."""
        if self.collection.count() > 0:
            return

        if not self.seed_path.exists():
            return

        with open(self.seed_path, "r", encoding="utf-8") as f:
            rules: List[Dict[str, Any]] = json.load(f)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for rule in rules:
            rule_id = rule.get("id", rule.get("cwe_id", "UNKNOWN"))
            name = rule.get("name", "")
            cwe_id = rule.get("cwe_id", "")
            severity = rule.get("severity", "MEDIUM")
            desc = rule.get("description", "")
            patterns = ", ".join(rule.get("patterns_detected", []))
            remediation = rule.get("remediation_guidelines", "")
            secure_ex = rule.get("secure_example", "")
            insecure_ex = rule.get("insecure_example", "")
            tags = ", ".join(rule.get("tags", []))

            # Build comprehensive text for dense embedding retrieval
            doc_text = (
                f"Rule: {rule_id} ({cwe_id}) - {name}\n"
                f"Severity: {severity}\n"
                f"Tags: {tags}\n"
                f"Description: {desc}\n"
                f"Vulnerable Patterns: {patterns}\n"
                f"Insecure Example: {insecure_ex}\n"
                f"Remediation Guidelines: {remediation}\n"
                f"Secure Example: {secure_ex}"
            )

            ids.append(rule_id)
            documents.append(doc_text)
            metadatas.append({
                "cwe_id": cwe_id,
                "name": name,
                "severity": severity,
                "category": rule.get("category", "General"),
            })

        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Successfully seeded ChromaDB collection '{self.COLLECTION_NAME}' with {len(ids)} rules.")

    def query_rules(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Perform semantic similarity search over security policies."""
        if not query_text.strip():
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )

        formatted: List[Dict[str, Any]] = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return formatted

        docs = results["documents"][0]
        ids = results["ids"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

        for doc_id, doc_text, meta, dist in zip(ids, docs, metadatas, distances):
            formatted.append({
                "id": doc_id,
                "document": doc_text,
                "metadata": meta,
                "distance": dist,
            })

        return formatted

    def format_rules_for_agent(self, query_text: str, top_k: int = 2) -> str:
        """Helper to format retrieved guidelines into a human- and agent-readable string."""
        rules = self.query_rules(query_text=query_text, top_k=top_k)
        if not rules:
            return "No specific security policy matched the query."

        sections = []
        for i, item in enumerate(rules, 1):
            sections.append(f"--- MATCH {i} [ID: {item['id']}] ---\n{item['document']}")

        return "\n\n".join(sections)


_vector_store_instance: Optional[SecurityVectorStore] = None


def get_vector_store() -> SecurityVectorStore:
    """Singleton provider for SecurityVectorStore."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = SecurityVectorStore()
    return _vector_store_instance
