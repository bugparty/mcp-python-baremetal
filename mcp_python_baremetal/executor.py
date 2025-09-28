"""Python code execution engine with automatic import injection."""

import ast
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Dict, Set, Tuple

# Common library mappings for auto-import
COMMON_IMPORTS = {
    "np": "import numpy as np",
    "scipy": "import scipy",
    "sp": "import scipy as sp",
    "plt": "import matplotlib.pyplot as plt",
    "pd": "import pandas as pd",
    "torch": "import torch",
    "nn": "from torch import nn",
    "F": "import torch.nn.functional as F",
    "jax": "import jax",
    "jnp": "import jax.numpy as jnp",
    "sympy": "import sympy",
    "cp": "import cvxpy as cp",
    "cvx": "import cvxpy as cvx",
    "numba": "import numba",
    "sage": "import sage",
}


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze variable usage and existing imports."""

    def __init__(self):
        self.used_names: Set[str] = set()
        self.imported_names: Set[str] = set()
        self.from_imports: Dict[str, Set[str]] = {}

    def visit_Name(self, node):
        """Track name usage."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Track attribute access (e.g., np.array)."""
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)

    def visit_Import(self, node):
        """Track import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track from-import statements."""
        module = node.module or ""
        if module not in self.from_imports:
            self.from_imports[module] = set()
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
            self.from_imports[module].add(name)
        self.generic_visit(node)


class ExecutionResult:
    """Container for execution results."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        status_code: int = 0,
        execution_status: str = "success",
        modified_code: str = "",
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.status_code = status_code
        self.execution_status = execution_status
        self.modified_code = modified_code

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "status_code": self.status_code,
            "execution_status": self.execution_status,
            "modified_code": self.modified_code,
        }


class PythonExecutor:
    """Python code executor with automatic import injection."""

    def __init__(self):
        self.globals_dict = self._setup_environment()

    def _setup_environment(self) -> Dict[str, Any]:
        """Setup the execution environment with pre-imported libraries."""
        env = {
            "__builtins__": __builtins__,
        }

        # Pre-import common libraries to have them available
        try:
            import numpy as np

            env["np"] = np
            env["numpy"] = np
        except ImportError:
            pass

        try:
            import scipy

            env["scipy"] = scipy
        except ImportError:
            pass

        try:
            import sympy

            env["sympy"] = sympy
        except ImportError:
            pass

        try:
            import torch

            env["torch"] = torch
        except ImportError:
            pass

        try:
            import jax
            import jax.numpy as jnp

            env["jax"] = jax
            env["jnp"] = jnp
        except ImportError:
            pass

        try:
            import cvxpy as cp

            env["cp"] = cp
            env["cvxpy"] = cp
        except ImportError:
            pass

        try:
            import numba

            env["numba"] = numba
        except ImportError:
            pass

        return env

    def analyze_imports(self, code: str) -> Tuple[Set[str], Set[str]]:
        """Analyze code to find used names and existing imports."""
        try:
            tree = ast.parse(code)
            analyzer = ImportAnalyzer()
            analyzer.visit(tree)

            # Find missing imports
            missing = analyzer.used_names - analyzer.imported_names
            # Filter to only common library names we can auto-import
            missing_libraries = missing.intersection(COMMON_IMPORTS.keys())

            return missing_libraries, analyzer.imported_names
        except SyntaxError:
            return set(), set()

    def inject_imports(self, code: str) -> str:
        """Inject missing imports into code."""
        missing_libs, existing_imports = self.analyze_imports(code)

        if not missing_libs:
            return code

        # Build import statements
        import_statements = []
        for lib in missing_libs:
            if lib in COMMON_IMPORTS:
                import_statements.append(COMMON_IMPORTS[lib])

        if import_statements:
            # Add imports at the beginning
            imports = "\n".join(import_statements) + "\n\n"
            return imports + code

        return code

    def execute(self, code: str, auto_import: bool = True) -> ExecutionResult:
        """Execute Python code with optional auto-import."""
        original_code = code

        # Apply auto-import if enabled
        if auto_import:
            code = self.inject_imports(code)

        # Capture stdout and stderr
        stdout = StringIO()
        stderr = StringIO()

        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Execute the code
                exec(code, self.globals_dict)

            return ExecutionResult(
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                status_code=0,
                execution_status="success",
                modified_code=code if code != original_code else "",
            )

        except Exception:
            # If execution failed and we haven't tried auto-import yet, try it
            if not auto_import:
                try:
                    modified_code = self.inject_imports(original_code)
                    if modified_code != original_code:
                        return self.execute(modified_code, auto_import=False)
                except Exception:
                    pass

            # Return error result
            error_output = stderr.getvalue()
            if not error_output:
                error_output = traceback.format_exc()

            return ExecutionResult(
                stdout=stdout.getvalue(),
                stderr=error_output,
                status_code=1,
                execution_status="error",
                modified_code=code if code != original_code else "",
            )

    def reset_environment(self):
        """Reset the execution environment."""
        self.globals_dict = self._setup_environment()
