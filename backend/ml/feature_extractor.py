"""AST-based code feature extractor for RepoSentinel triage classifier."""

import ast
from typing import Any, Dict


class ASTMetricsVisitor(ast.NodeVisitor):
    """Visitor to compute structural complexity and security indicator metrics from a Python AST."""

    NESTING_CONSTRUCTS = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
    )

    BRANCHING_CONSTRUCTS = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
    )

    DANGEROUS_CALLS = {"eval", "exec", "__import__"}
    SQL_INDICATOR_METHODS = {"execute", "executemany", "raw", "cursor"}

    def __init__(self) -> None:
        self.num_functions: int = 0
        self.max_nesting_depth: int = 0
        self.current_depth: int = 0
        self.num_branches: int = 0
        self.num_imports: int = 0
        self.num_external_calls: int = 0
        self.has_eval_or_exec: int = 0
        self.has_raw_sql_indicators: int = 0

    def generic_visit(self, node: ast.AST) -> None:
        is_nesting = isinstance(node, self.NESTING_CONSTRUCTS)
        if is_nesting:
            self.current_depth += 1
            if self.current_depth > self.max_nesting_depth:
                self.max_nesting_depth = self.current_depth

        super().generic_visit(node)

        if is_nesting:
            self.current_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.num_functions += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.num_functions += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.num_branches += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.num_branches += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.num_branches += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.num_branches += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.num_branches += 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.num_imports += len(node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.num_imports += len(node.names)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.num_external_calls += 1

        # Check for dangerous dynamic code execution
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DANGEROUS_CALLS:
                self.has_eval_or_exec = 1
            elif node.func.id.lower() in self.SQL_INDICATOR_METHODS:
                self.has_raw_sql_indicators = 1
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr.lower() in self.SQL_INDICATOR_METHODS:
                self.has_raw_sql_indicators = 1
            if node.func.attr in self.DANGEROUS_CALLS:
                self.has_eval_or_exec = 1

        self.generic_visit(node)


def extract_ast_features(code_str: str) -> Dict[str, Any]:
    """Extract 8 structural complexity and security metrics from Python source code.
    
    Robustly handles malformed or unparseable inputs with sensible fallback values.
    
    Returns:
        dict with keys:
            - lines_of_code
            - num_functions
            - max_nesting_depth
            - num_branches
            - num_imports
            - num_external_calls
            - has_eval_or_exec
            - has_raw_sql_indicators
    """
    if not isinstance(code_str, str):
        code_str = str(code_str or "")

    raw_lines = [line.strip() for line in code_str.splitlines()]
    non_empty_lines = [line for line in raw_lines if line and not line.startswith("#")]
    loc = len(non_empty_lines)

    fallback_metrics: Dict[str, Any] = {
        "lines_of_code": loc,
        "num_functions": 0,
        "max_nesting_depth": 0,
        "num_branches": 0,
        "num_imports": 0,
        "num_external_calls": 0,
        "has_eval_or_exec": 0,
        "has_raw_sql_indicators": 0,
    }

    if not code_str.strip():
        return fallback_metrics

    try:
        tree = ast.parse(code_str)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        # Fallback for unparseable diffs/malformed code
        # Check textually for dangerous keywords as heuristic fallback
        lower_code = code_str.lower()
        if any(kw in lower_code for kw in ("eval(", "exec(", "__import__(")):
            fallback_metrics["has_eval_or_exec"] = 1
        if any(kw in lower_code for kw in (".execute(", ".raw(", ".cursor(")):
            fallback_metrics["has_raw_sql_indicators"] = 1
        return fallback_metrics

    visitor = ASTMetricsVisitor()
    visitor.visit(tree)

    return {
        "lines_of_code": loc,
        "num_functions": visitor.num_functions,
        "max_nesting_depth": visitor.max_nesting_depth,
        "num_branches": visitor.num_branches,
        "num_imports": visitor.num_imports,
        "num_external_calls": visitor.num_external_calls,
        "has_eval_or_exec": visitor.has_eval_or_exec,
        "has_raw_sql_indicators": visitor.has_raw_sql_indicators,
    }
