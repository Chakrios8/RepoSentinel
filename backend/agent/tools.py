"""LangChain tools for RepoSentinel ReAct agent: RAG security KB query and AST syntax validator."""

import ast
import re
from langchain_core.tools import tool
from backend.rag.vector_store import get_vector_store


def strip_markdown_fences(code_str: str) -> str:
    """Strip markdown code fence backticks (```python ... ```) if present."""
    text = code_str.strip()
    # Match markdown fences like ```python\n ... \n``` or ``` ... ```
    fence_pattern = r"^```(?:[a-zA-Z0-9_-]+)?\r?\n([\s\S]*?)\r?\n```$"
    match = re.match(fence_pattern, text)
    if match:
        return match.group(1).strip()

    # Fallback line-by-line strip if trailing/leading fences exist
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


@tool
def query_security_kb(query: str) -> str:
    """Query the RepoSentinel ChromaDB vector store for relevant CWE/CVE policies, security rules, and remediation patterns.
    
    Args:
        query: Security keywords, code snippet symptoms, or CWE names (e.g., 'SQL injection execute', 'eval dynamic code', 'hardcoded secret').
        
    Returns:
        Structured security guidelines, vulnerable patterns, and secure remediation code examples.
    """
    try:
        store = get_vector_store()
        result = store.format_rules_for_agent(query_text=query, top_k=2)
        return result
    except Exception as exc:
        return f"Warning: Unable to query ChromaDB vector store: {str(exc)}"


@tool
def validate_code_syntax(candidate_code: str) -> str:
    """Validate that candidate Python code or remediation patch parses without syntax errors using Python's native AST parser.
    
    Args:
        candidate_code: The Python source code patch to validate (can contain or omit markdown backticks).
        
    Returns:
        'VALIDATION SUCCESS' if syntax is valid, or 'VALIDATION ERROR: <syntax details>' with exact line number and error message if invalid.
    """
    cleaned_code = strip_markdown_fences(candidate_code)
    if not cleaned_code.strip():
        return "VALIDATION ERROR: Candidate code is empty."

    try:
        ast.parse(cleaned_code)
        return "VALIDATION SUCCESS: AST parsed cleanly with 0 syntax errors."
    except SyntaxError as err:
        line = err.lineno or 0
        col = err.offset or 0
        msg = err.msg or "Invalid syntax"
        text_line = (err.text or "").strip()
        return (
            f"VALIDATION ERROR: Syntax error on line {line}, column {col}: {msg}. "
            f"Problem line: '{text_line}'. Please correct this syntax error and validate again."
        )
    except Exception as err:
        return f"VALIDATION ERROR: Unexpected error while parsing code: {str(err)}"
