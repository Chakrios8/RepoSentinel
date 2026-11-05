"""Autonomous security remediation agent engine powered by Google Gemini and AST verification."""

import os
from typing import Any, Dict, Optional
from google import genai
from google.genai import types

from backend.config import get_settings
from backend.agent.tools import query_security_kb, validate_code_syntax
from backend.rag.vector_store import get_vector_store

SYSTEM_PROMPT = """You are RepoSentinel, an autonomous security auditor and remediation engineer.
Your task is to analyze code flagged by our static AST triage filter, identify security vulnerabilities (such as CWE-89 SQLi, CWE-95 RCE, CWE-798 Hardcoded Secrets, CWE-400 ReDoS), and produce a verified, production-grade security patch.

MANDATORY EXECUTION WORKFLOW:
1. QUERY SECURITY KB: Call `query_security_kb` with the specific vulnerability pattern or suspected CWE to retrieve official security policies and safe patterns.
2. SYNTHESIZE REMEDIATION: Formulate a minimal, robust security patch that completely neutralizes the vulnerability while preserving original functionality.
3. VALIDATE SYNTAX: Call `validate_code_syntax` on your candidate Python patch. If the output is "VALIDATION ERROR", inspect the exact line/column reported, correct the code, and re-run `validate_code_syntax` until it returns "VALIDATION SUCCESS".
4. FINAL REPORT: In your final answer, provide:
   - Vulnerability Identified (CWE ID and description)
   - Root Cause Analysis
   - Verified Secure Patch (inside ```python ... ``` fences)
   - Explanation of Fix & Best Practices
"""


class SecurityAgentRunner:
    """Executes the autonomous security audit loop with Google Gemini and compiler AST tools."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _get_api_key(self) -> Optional[str]:
        """Retrieve cleaned Gemini API key from settings or environment."""
        key = self.settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key or key.strip() == "your_gemini_api_key_here":
            return None
        return key.strip()

    def _query_security_kb_wrapper(self, query: str) -> str:
        """Expose vector store query tool to the agent."""
        return query_security_kb.invoke({"query": query})

    def _validate_code_syntax_wrapper(self, candidate_code: str) -> str:
        """Expose AST compiler validator tool to the agent."""
        return validate_code_syntax.invoke({"candidate_code": candidate_code})

    def run_remediation(
        self,
        code: str,
        file_path: str = "snippet.py",
        ast_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the security agent or return a knowledge-base guided fallback if API key is not configured."""
        metrics_repr = str(ast_metrics or {})
        api_key = self._get_api_key()

        # If API key is not configured yet, provide rich RAG-grounded advisory fallback
        if not api_key:
            query_hint = "security vulnerability remediation"
            if ast_metrics:
                if ast_metrics.get("has_raw_sql_indicators"):
                    query_hint = "SQL Injection CWE-89 parameterized query cursor.execute"
                elif ast_metrics.get("has_eval_or_exec"):
                    query_hint = "Dynamic code execution eval exec CWE-95 ast.literal_eval"
                elif "api_key" in code.lower() or "secret" in code.lower() or "password" in code.lower():
                    query_hint = "Hardcoded credentials secrets CWE-798"
                elif "re." in code or "regex" in code.lower():
                    query_hint = "ReDoS regular expression CWE-400"

            vector_store = get_vector_store()
            kb_policy = vector_store.format_rules_for_agent(query_hint, top_k=1)

            advisory = (
                "### [RepoSentinel Security Alert - High Risk Detected]\n\n"
                "AST Triage flagged this code as high-risk and triggered the autonomous remediation workflow.\n\n"
                "**Status:** Autonomous Gemini agentic remediation loop is paused pending Google Gemini API key configuration.\n\n"
                "**Action Required:**\n"
                "1. Add your Google Gemini API key to `.env` (`GOOGLE_API_KEY=your_key`).\n"
                "2. Restart or re-query to enable dynamic multi-step patch generation and AST compiler validation.\n\n"
                "**Relevant Security Policy (Retrieved from ChromaDB Vector Store):**\n\n"
                f"{kb_policy}"
            )
            return {
                "success": True,
                "remediation_report": advisory,
                "agent_status": "api_key_pending",
            }

        try:
            client = genai.Client(api_key=api_key)

            # Build tools list
            tools = [self._query_security_kb_wrapper, self._validate_code_syntax_wrapper]

            # Initialize chat session with system instruction and tool calling
            chat = client.chats.create(
                model=self.settings.GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=tools,
                    temperature=0.1,
                )
            )

            prompt = (
                f"Target File: {file_path}\n"
                f"AST Complexity & Threat Metrics: {metrics_repr}\n\n"
                f"Vulnerable Code to Audit and Remediate:\n"
                f"```python\n{code}\n```"
            )

            response = chat.send_message(prompt)
            output_text = response.text or ""

            return {
                "success": True,
                "remediation_report": output_text,
                "agent_status": "completed",
            }
        except Exception as exc:
            # Fallback to direct RAG retrieval if transient network or API issue occurs
            vector_store = get_vector_store()
            kb_policy = vector_store.format_rules_for_agent("security vulnerability remediation CWE-89 CWE-95", top_k=1)
            return {
                "success": False,
                "remediation_report": f"Agent encountered: {str(exc)}\n\nRetrieved Security Policy:\n{kb_policy}",
                "agent_status": "error",
            }


_agent_runner_instance: Optional[SecurityAgentRunner] = None


def get_agent_runner() -> SecurityAgentRunner:
    """Singleton provider for SecurityAgentRunner."""
    global _agent_runner_instance
    if _agent_runner_instance is None:
        _agent_runner_instance = SecurityAgentRunner()
    return _agent_runner_instance
