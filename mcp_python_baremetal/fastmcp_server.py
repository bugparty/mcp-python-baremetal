"""FastMCP server implementation for Python code execution with auto-import."""

import logging
from typing import Any, Dict

from fastmcp import FastMCP

from .executor import PythonExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_server() -> FastMCP:
    """Create and configure the FastMCP server with Python execution tools."""

    # Server instructions for LLM integration
    server_instructions = """
    This MCP server provides Python code execution capabilities with intelligent 
    automatic import injection. It can execute Python code, analyze imports, 
    and automatically inject missing common library imports like numpy, torch, 
    jax, scipy, etc. The execution environment persists between calls until reset.
    
    Key features:
    - Execute Python code with auto-import of common libraries
    - Analyze code for missing imports  
    - Reset execution environment
    - Support for mathematical and ML libraries
    """

    # Initialize the FastMCP server
    mcp = FastMCP(name="Python Baremetal Executor", instructions=server_instructions)

    # Initialize the Python executor
    executor = PythonExecutor()

    @mcp.tool()
    async def execute_python(
        code: str, auto_import: bool = True, reset_env: bool = False
    ) -> Dict[str, Any]:
        """
        Execute Python code with automatic import injection.

        This tool executes Python code with intelligent auto-import capabilities.
        It can automatically detect and inject missing imports for common libraries
        like numpy (np), torch, jax, scipy, sympy, etc.

        Args:
            code: Python code to execute
            auto_import: Whether to automatically inject missing imports (default: True)
            reset_env: Whether to reset the execution environment before running
                      (default: False)

        Returns:
            Dictionary containing execution results with stdout, stderr, status,
            and any auto-imports added
        """
        logger.info(
            f"Executing Python code (auto_import={auto_import}, reset_env={reset_env})"
        )

        if reset_env:
            executor.reset_environment()
            logger.info("Execution environment reset")

        # Execute the code
        result = executor.execute(code, auto_import=auto_import)

        response = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "status_code": result.status_code,
            "execution_status": result.execution_status,
        }

        # Add modified code if auto-imports were added
        if result.modified_code:
            response["modified_code"] = result.modified_code
            response["auto_imports_added"] = True
        else:
            response["auto_imports_added"] = False

        logger.info(f"Execution completed with status: {result.execution_status}")
        return response

    @mcp.tool()
    async def analyze_imports(code: str) -> Dict[str, Any]:
        """
        Analyze Python code to identify missing imports and show what would be
        auto-imported.

        This tool analyzes code without executing it to show what imports are missing
        and what would be automatically added by the auto-import system.

        Args:
            code: Python code to analyze

        Returns:
            Dictionary with missing libraries, existing imports, and preview of
            modified code
        """
        logger.info("Analyzing code imports")

        missing_libs, existing_imports = executor.analyze_imports(code)
        modified_code = executor.inject_imports(code)

        return {
            "missing_libraries": list(missing_libs),
            "existing_imports": list(existing_imports),
            "would_modify": modified_code != code,
            "modified_code": modified_code if modified_code != code else None,
        }

    @mcp.tool()
    async def reset_environment() -> Dict[str, str]:
        """
        Reset the Python execution environment.

        This tool clears all variables and state from the Python execution environment,
        starting fresh for subsequent code executions.

        Returns:
            Confirmation message that the environment was reset
        """
        logger.info("Resetting Python execution environment")
        executor.reset_environment()
        return {"message": "Python execution environment has been reset"}

    @mcp.tool()
    async def get_available_libraries() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available libraries and their auto-import mappings.

        This tool returns information about which libraries are available for
        auto-import and their corresponding import statements.

        Returns:
            Dictionary of library aliases and their import information
        """
        from .executor import COMMON_IMPORTS

        logger.info("Getting available library information")

        # Test which libraries are actually available
        available_libs = {}
        for alias, import_stmt in COMMON_IMPORTS.items():
            try:
                exec(import_stmt, {})
                available_libs[alias] = {
                    "import_statement": import_stmt,
                    "available": True,
                }
            except ImportError:
                available_libs[alias] = {
                    "import_statement": import_stmt,
                    "available": False,
                }

        available_count = sum(1 for lib in available_libs.values() if lib["available"])

        return {
            "libraries": available_libs,
            "total_mappings": len(COMMON_IMPORTS),
            "available_count": available_count,
        }

    return mcp


def create_fastmcp_server():
    """Create the FastMCP server instance."""
    return create_server()


def main():
    """Main function to start the FastMCP server."""
    logger.info("Starting FastMCP Python Baremetal server")

    # Create the MCP server
    server = create_server()

    logger.info("Starting server on 0.0.0.0:8000 with SSE transport")

    try:
        # Use FastMCP's built-in run method with SSE transport
        server.run(transport="sse", host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
