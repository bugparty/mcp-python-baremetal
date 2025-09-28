"""Command-line interface for MCP Python Baremetal."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

from .executor import PythonExecutor
from .fastmcp_server import create_fastmcp_server
from .http_server import HTTPServer
from .server import MCPPythonServer


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """MCP Python Baremetal - A simple Python execution MCP server."""
    pass


@cli.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio"]),
    default="stdio",
    help="Transport protocol for MCP server",
)
def serve(transport: str):
    """Start the MCP server."""
    click.echo(f"Starting MCP Python Baremetal server with {transport} transport...")

    async def run_server():
        server = MCPPythonServer()
        await server.run(transport_type=transport)

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        click.echo("\nServer stopped.")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the HTTP server to")
@click.option("--port", default=8000, type=int, help="Port to bind the HTTP server to")
def http(host: str, port: int):
    """Start the HTTP/SSE server (legacy FastAPI)."""
    click.echo(f"Starting HTTP server on {host}:{port}...")

    async def run_http():
        server = HTTPServer()
        await server.run(host=host, port=port)

    try:
        asyncio.run(run_http())
    except KeyboardInterrupt:
        click.echo("\nHTTP server stopped.")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the FastMCP server to")
@click.option(
    "--port", default=8000, type=int, help="Port to bind the FastMCP server to"
)
def fastmcp(host: str, port: int):
    """Start the FastMCP server (recommended for ChatGPT integration)."""
    click.echo(f"Starting FastMCP server on {host}:{port}...")

    try:
        server = create_fastmcp_server()
        server.run(transport="sse", host=host, port=port)
    except KeyboardInterrupt:
        click.echo("\nFastMCP server stopped.")


@cli.command()
@click.argument("code", required=False)
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True, path_type=Path),
    help="Execute Python code from file",
)
@click.option(
    "--auto-import/--no-auto-import",
    default=True,
    help="Enable/disable automatic import injection",
)
@click.option(
    "--analyze-only", is_flag=True, help="Only analyze imports without executing"
)
@click.option("--reset", is_flag=True, help="Reset environment before execution")
def execute(
    code: Optional[str],
    file: Optional[Path],
    auto_import: bool,
    analyze_only: bool,
    reset: bool,
):
    """Execute Python code directly."""

    if not code and not file:
        click.echo("Error: Must provide either code or --file option", err=True)
        sys.exit(1)

    if file:
        code = file.read_text()
        click.echo(f"Executing code from {file}...")

    executor = PythonExecutor()

    if reset:
        executor.reset_environment()
        click.echo("Environment reset.")

    if analyze_only:
        # Just analyze imports
        missing_libs, existing_imports = executor.analyze_imports(code)

        click.echo("=== Import Analysis ===")
        if existing_imports:
            click.echo(f"Existing imports: {', '.join(sorted(existing_imports))}")

        if missing_libs:
            click.echo(f"Missing libraries: {', '.join(sorted(missing_libs))}")

            # Show what would be added
            from .executor import COMMON_IMPORTS

            imports_to_add = []
            for lib in missing_libs:
                if lib in COMMON_IMPORTS:
                    imports_to_add.append(COMMON_IMPORTS[lib])

            if imports_to_add:
                click.echo("\nWould add imports:")
                for imp in imports_to_add:
                    click.echo(f"  {imp}")
        else:
            click.echo("No missing imports detected.")

        # Show modified code
        modified_code = executor.inject_imports(code)
        if modified_code != code:
            click.echo("\n=== Modified Code ===")
            click.echo(modified_code)

        return

    # Execute code
    click.echo("=== Execution ===")
    result = executor.execute(code, auto_import=auto_import)

    if result.modified_code:
        click.echo("Auto-imports were added:")
        click.echo(result.modified_code)
        click.echo("=" * 50)

    if result.stdout:
        click.echo("Output:")
        click.echo(result.stdout)

    if result.stderr:
        click.echo("Errors:", err=True)
        click.echo(result.stderr, err=True)

    click.echo(f"Status: {result.execution_status} (code: {result.status_code})")

    if result.status_code != 0:
        sys.exit(result.status_code)


@cli.command()
def libraries():
    """List available libraries and their import mappings."""
    from .executor import COMMON_IMPORTS

    click.echo("=== Available Library Mappings ===")

    # Test availability
    available_count = 0

    for alias, import_stmt in sorted(COMMON_IMPORTS.items()):
        try:
            exec(import_stmt, {})
            status = "✓ Available"
            available_count += 1
        except ImportError:
            status = "✗ Not installed"

        click.echo(f"{alias:10} -> {import_stmt:30} [{status}]")

    click.echo(f"\nTotal mappings: {len(COMMON_IMPORTS)}")
    click.echo(f"Available: {available_count}")
    click.echo(f"Missing: {len(COMMON_IMPORTS) - available_count}")


@cli.command()
def demo():
    """Run a demonstration of auto-import functionality."""
    demo_codes = [
        # NumPy demo
        """
# This code uses np without importing numpy
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Mean:", np.mean(arr))
""",
        # Multiple libraries demo
        """
# This uses multiple libraries without imports
data = np.random.randn(100)
plt.hist(data)
plt.title("Random Data Histogram")
plt.show()
""",
        # PyTorch demo
        """
# PyTorch without imports
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = torch.dot(x, y)
print("Dot product:", z)
""",
    ]

    executor = PythonExecutor()

    for i, code in enumerate(demo_codes, 1):
        click.echo(f"\n=== Demo {i} ===")
        click.echo("Original code:")
        click.echo(code.strip())

        # Analyze
        missing_libs, _ = executor.analyze_imports(code)
        if missing_libs:
            click.echo(
                f"\nMissing libraries detected: {', '.join(sorted(missing_libs))}"
            )

        # Execute with auto-import
        click.echo("\nExecuting with auto-import...")
        result = executor.execute(code, auto_import=True)

        if result.modified_code:
            click.echo("Modified code:")
            click.echo(result.modified_code)

        if result.stdout:
            click.echo("Output:")
            click.echo(result.stdout)

        if result.stderr:
            click.echo("Errors:")
            click.echo(result.stderr)

        click.echo(f"Status: {result.execution_status}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
