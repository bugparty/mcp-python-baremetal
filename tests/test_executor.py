"""Tests for the Python executor."""

from mcp_python_baremetal.executor import ImportAnalyzer, PythonExecutor


class TestImportAnalyzer:
    """Test the AST-based import analyzer."""

    def test_simple_name_usage(self):
        """Test detection of simple name usage."""
        code = "print(np.array([1, 2, 3]))"
        analyzer = ImportAnalyzer()
        tree = __import__("ast").parse(code)
        analyzer.visit(tree)

        assert "np" in analyzer.used_names
        assert "print" in analyzer.used_names
        assert "array" not in analyzer.used_names  # This is an attribute, not a name

    def test_existing_imports(self):
        """Test detection of existing imports."""
        code = """
import numpy as np
from scipy import stats
import torch

result = np.array([1, 2, 3])
"""
        analyzer = ImportAnalyzer()
        tree = __import__("ast").parse(code)
        analyzer.visit(tree)

        assert "np" in analyzer.imported_names
        assert "stats" in analyzer.imported_names
        assert "torch" in analyzer.imported_names

    def test_attribute_access(self):
        """Test detection of attribute access."""
        code = "result = torch.tensor([1, 2, 3]).cuda()"
        analyzer = ImportAnalyzer()
        tree = __import__("ast").parse(code)
        analyzer.visit(tree)

        assert "torch" in analyzer.used_names


class TestPythonExecutor:
    """Test the Python executor."""

    def setup_method(self):
        """Setup for each test."""
        self.executor = PythonExecutor()

    def test_simple_execution(self):
        """Test simple code execution without imports."""
        code = "print('Hello, World!')"
        result = self.executor.execute(code)

        assert result.execution_status == "success"
        assert result.status_code == 0
        assert "Hello, World!" in result.stdout
        assert result.stderr == ""

    def test_execution_with_error(self):
        """Test execution with error."""
        code = "print(undefined_variable)"
        result = self.executor.execute(code)

        assert result.execution_status == "error"
        assert result.status_code == 1
        assert "NameError" in result.stderr

    def test_auto_import_numpy(self):
        """Test automatic numpy import."""
        code = "print(np.array([1, 2, 3]))"
        result = self.executor.execute(code, auto_import=True)

        # Should succeed with auto-import
        assert result.execution_status == "success"
        assert result.status_code == 0
        assert "[1 2 3]" in result.stdout
        assert "import numpy as np" in result.modified_code

    def test_multiple_auto_imports(self):
        """Test multiple automatic imports."""
        code = """
data = np.random.randn(10)
tensor = torch.tensor(data)
print("Data shape:", data.shape)
print("Tensor:", tensor)
"""
        result = self.executor.execute(code, auto_import=True)

        assert result.execution_status == "success"
        assert result.status_code == 0
        assert "import numpy as np" in result.modified_code
        assert "import torch" in result.modified_code

    def test_no_unnecessary_imports(self):
        """Test that existing imports are not duplicated."""
        code = """
import numpy as np
print(np.array([1, 2, 3]))
"""
        result = self.executor.execute(code, auto_import=True)

        assert result.execution_status == "success"
        assert result.status_code == 0
        # Should not modify code since import already exists
        assert result.modified_code == ""

    def test_import_analysis(self):
        """Test import analysis functionality."""
        code = "result = np.array([1, 2, 3]) + torch.tensor([4, 5, 6])"
        missing, existing = self.executor.analyze_imports(code)

        assert "np" in missing
        assert "torch" in missing
        assert len(existing) == 0  # No existing imports

    def test_inject_imports(self):
        """Test import injection."""
        code = "print(np.array([1, 2, 3]))"
        modified = self.executor.inject_imports(code)

        assert "import numpy as np" in modified
        assert code in modified  # Original code should be preserved

    def test_environment_reset(self):
        """Test environment reset."""
        # Set a variable
        code1 = "test_var = 42"
        self.executor.execute(code1)

        # Check it exists
        code2 = "print(test_var)"
        result2 = self.executor.execute(code2)
        assert "42" in result2.stdout

        # Reset environment
        self.executor.reset_environment()

        # Variable should be gone
        result3 = self.executor.execute(code2)
        assert result3.execution_status == "error"
        assert "NameError" in result3.stderr

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = "print('unclosed string"
        result = self.executor.execute(code)

        assert result.execution_status == "error"
        assert result.status_code == 1
        # Should have some error information
        assert result.stderr != ""

    def test_persistent_environment(self):
        """Test that variables persist between executions."""
        # Set a variable
        code1 = "x = 10"
        result1 = self.executor.execute(code1)
        assert result1.execution_status == "success"

        # Use the variable
        code2 = "print(x * 2)"
        result2 = self.executor.execute(code2)
        assert result2.execution_status == "success"
        assert "20" in result2.stdout
