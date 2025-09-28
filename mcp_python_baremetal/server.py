"""MCP server implementation for Python code execution."""

from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool

from .executor import PythonExecutor


class MCPPythonServer:
    """MCP server for Python code execution."""

    def __init__(self):
        self.server = Server("python-baremetal")
        self.executor = PythonExecutor()
        self._setup_tools()

    def _setup_tools(self):
        """Setup MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="execute_python",
                    description="Execute Python code with automatic import injection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute",
                            },
                            "auto_import": {
                                "type": "boolean",
                                "description": (
                                    "Whether to automatically inject missing imports"
                                ),
                                "default": True,
                            },
                            "reset_env": {
                                "type": "boolean",
                                "description": (
                                    "Whether to reset the execution environment "
                                    "before running"
                                ),
                                "default": False,
                            },
                        },
                        "required": ["code"],
                    },
                ),
                Tool(
                    name="reset_environment",
                    description="Reset the Python execution environment",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
                Tool(
                    name="analyze_imports",
                    description=(
                        "Analyze Python code to show what imports would be added"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to analyze",
                            }
                        },
                        "required": ["code"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""

            if name == "execute_python":
                code = arguments["code"]
                auto_import = arguments.get("auto_import", True)
                reset_env = arguments.get("reset_env", False)

                if reset_env:
                    self.executor.reset_environment()

                result = self.executor.execute(code, auto_import=auto_import)

                # Format response
                response_parts = []

                if result.stdout:
                    response_parts.append(f"**Output:**\n```\n{result.stdout}\n```")

                if result.stderr:
                    response_parts.append(f"**Errors:**\n```\n{result.stderr}\n```")

                if result.modified_code:
                    response_parts.append(
                        f"**Modified Code (auto-imports added):**\n"
                        f"```python\n{result.modified_code}\n```"
                    )

                response_parts.append(
                    f"**Status:** {result.execution_status} "
                    f"(code: {result.status_code})"
                )

                if not response_parts:
                    response_parts.append("Code executed successfully with no output.")

                return [TextContent(type="text", text="\n\n".join(response_parts))]

            elif name == "reset_environment":
                self.executor.reset_environment()
                return [
                    TextContent(
                        type="text", text="Python execution environment has been reset."
                    )
                ]

            elif name == "analyze_imports":
                code = arguments["code"]
                missing_libs, existing_imports = self.executor.analyze_imports(code)

                response_parts = []
                if existing_imports:
                    response_parts.append(
                        f"**Existing imports:** {', '.join(sorted(existing_imports))}"
                    )

                if missing_libs:
                    from .executor import COMMON_IMPORTS

                    imports_to_add = []
                    for lib in missing_libs:
                        if lib in COMMON_IMPORTS:
                            imports_to_add.append(COMMON_IMPORTS[lib])

                    response_parts.append(
                        f"**Missing libraries detected:** "
                        f"{', '.join(sorted(missing_libs))}"
                    )
                    if imports_to_add:
                        response_parts.append(
                            "**Would add imports:**\n```python\n"
                            + "\n".join(imports_to_add)
                            + "\n```"
                        )
                else:
                    response_parts.append("**No missing imports detected.**")

                modified_code = self.executor.inject_imports(code)
                if modified_code != code:
                    response_parts.append(
                        f"**Code with auto-imports:**\n```python\n{modified_code}\n```"
                    )

                return [TextContent(type="text", text="\n\n".join(response_parts))]

            else:
                raise ValueError(f"Unknown tool: {name}")

    async def run(self, transport_type: str = "stdio"):
        """Run the MCP server."""
        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server
            from mcp.server import NotificationOptions

            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="python-baremetal",
                        server_version="0.1.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(), experimental_capabilities={}
                        ),
                    ),
                )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")


async def main():
    """Main entry point for MCP server."""
    server = MCPPythonServer()
    await server.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
