"""HTTP/SSE server for ChatGPT developer compatibility."""

import json
import asyncio
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sse_starlette.sse import EventSourceResponse

from .executor import PythonExecutor, ExecutionResult


class CodeExecutionRequest(BaseModel):
    """Request model for code execution."""
    code: str
    auto_import: bool = True
    reset_env: bool = False


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis."""
    code: str


class ExecutionResponse(BaseModel):
    """Response model for code execution."""
    stdout: str
    stderr: str
    status_code: int
    execution_status: str
    modified_code: Optional[str] = None


class HTTPServer:
    """HTTP/SSE server for Python code execution."""
    
    def __init__(self):
        self.executor = PythonExecutor()
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            print("Starting Python Baremetal HTTP Server...")
            yield
            # Shutdown
            print("Shutting down Python Baremetal HTTP Server...")
            
        app = FastAPI(
            title="MCP Python Baremetal Server",
            description="HTTP/SSE interface for Python code execution with auto-import",
            version="0.1.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "mcp-python-baremetal"}
            
        # Execute Python code
        @app.post("/execute", response_model=ExecutionResponse)
        async def execute_code(request: CodeExecutionRequest):
            """Execute Python code."""
            try:
                if request.reset_env:
                    self.executor.reset_environment()
                    
                result = self.executor.execute(
                    request.code, 
                    auto_import=request.auto_import
                )
                
                return ExecutionResponse(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    status_code=result.status_code,
                    execution_status=result.execution_status,
                    modified_code=result.modified_code if result.modified_code else None
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Execution failed: {str(e)}"
                )
                
        # Execute with SSE streaming
        @app.post("/execute/stream")
        async def execute_code_stream(request: CodeExecutionRequest):
            """Execute Python code with SSE streaming."""
            
            async def event_generator():
                try:
                    # Send start event
                    yield {
                        "event": "start",
                        "data": json.dumps({
                            "status": "executing",
                            "code": request.code
                        })
                    }
                    
                    if request.reset_env:
                        self.executor.reset_environment()
                        yield {
                            "event": "info", 
                            "data": json.dumps({"message": "Environment reset"})
                        }
                    
                    # Execute code
                    result = self.executor.execute(
                        request.code,
                        auto_import=request.auto_import
                    )
                    
                    # Send progress events
                    if result.modified_code:
                        yield {
                            "event": "import_injection",
                            "data": json.dumps({
                                "message": "Auto-imports added",
                                "modified_code": result.modified_code
                            })
                        }
                    
                    if result.stdout:
                        yield {
                            "event": "stdout",
                            "data": json.dumps({"output": result.stdout})
                        }
                        
                    if result.stderr:
                        yield {
                            "event": "stderr", 
                            "data": json.dumps({"error": result.stderr})
                        }
                    
                    # Send completion event
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "status_code": result.status_code,
                            "execution_status": result.execution_status,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "modified_code": result.modified_code
                        })
                    }
                    
                except Exception as e:
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "error": str(e),
                            "status": "failed"
                        })
                    }
            
            return EventSourceResponse(event_generator())
            
        # Analyze code imports
        @app.post("/analyze")
        async def analyze_code(request: CodeAnalysisRequest):
            """Analyze code for import requirements."""
            try:
                missing_libs, existing_imports = self.executor.analyze_imports(request.code)
                modified_code = self.executor.inject_imports(request.code)
                
                return {
                    "missing_libraries": list(missing_libs),
                    "existing_imports": list(existing_imports),
                    "modified_code": modified_code if modified_code != request.code else None,
                    "would_modify": modified_code != request.code
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {str(e)}"
                )
                
        # Reset environment
        @app.post("/reset")
        async def reset_environment():
            """Reset the Python execution environment."""
            try:
                self.executor.reset_environment()
                return {"status": "success", "message": "Environment reset"}
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Reset failed: {str(e)}"
                )
                
        # Get available libraries
        @app.get("/libraries")
        async def get_libraries():
            """Get list of available libraries and their import mappings."""
            from .executor import COMMON_IMPORTS
            
            # Test which libraries are actually available
            available_libs = {}
            for alias, import_stmt in COMMON_IMPORTS.items():
                try:
                    exec(import_stmt, {})
                    available_libs[alias] = {
                        "import_statement": import_stmt,
                        "available": True
                    }
                except ImportError:
                    available_libs[alias] = {
                        "import_statement": import_stmt,
                        "available": False
                    }
                    
            return {
                "common_imports": available_libs,
                "total_mappings": len(COMMON_IMPORTS)
            }
            
        return app
        
    async def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the HTTP server."""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        )
        server = uvicorn.Server(config)
        await server.serve()


def run_http_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the HTTP server (synchronous entry point)."""
    http_server = HTTPServer()
    uvicorn.run(
        http_server.app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    run_http_server()