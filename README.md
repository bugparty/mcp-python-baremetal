# MCP Python Baremetal

A simple, "baremetal" Python execution MCP server that supports both HTTP/SSE (ChatGPT developer compatible) and command-line (Claude compatible) interfaces. 

## Features

- **Multiple Interfaces**: 
  - MCP server with stdio transport (Claude compatible)
  - HTTP/SSE API server (ChatGPT developer compatible)  
  - Direct CLI execution
- **Automatic Import Injection**: Uses AST analysis to detect missing imports and automatically adds common ones like `import numpy as np`
- **Pre-installed Math Libraries**: Includes NumPy, SciPy, SymPy, JAX, PyTorch (CPU), Numba, and CVXPy
- **No Sandboxing**: Direct execution environment - fast and simple
- **Persistent Environment**: Variables persist between executions until reset

## Requirements

- **Python**: 3.10 or higher
- **uv**: Modern Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

## Installation

### For End Users

> **Note**: This package is not yet published to PyPI. For now, please use the development installation method below.

Once published, you will be able to install with:

```bash
# Using uv tool (recommended)
uv tool install mcp-python-baremetal

# Or with pip
pip install mcp-python-baremetal
```

### For Development

**Prerequisites**: Python 3.10+ and [uv](https://docs.astral.sh/uv/) installed.

```bash
# Clone the repository
git clone https://github.com/bugparty/mcp-python-baremetal
cd mcp-python-baremetal

# Create virtual environment and install dependencies
uv sync

# Install development dependencies (optional, for testing and linting)
uv sync --extra dev
```

## Usage

### Development Usage

When developing, use `uv run` to run commands in the project's virtual environment:

```bash
# All commands should be prefixed with 'uv run' during development
uv run mcp-python-baremetal --help
```

### As MCP Server (for Claude)

```bash
# Development
uv run mcp-python-baremetal serve

# Production (after installation)
mcp-python-baremetal serve
```

This starts the MCP server with stdio transport that Claude can connect to.

### As HTTP/SSE Server (for ChatGPT Developer)

#### FastMCP Server (Recommended)

```bash
# Development
uv run mcp-python-baremetal fastmcp --host 0.0.0.0 --port 8000

# Production (after installation)
mcp-python-baremetal fastmcp --host 0.0.0.0 --port 8000
```

This starts a FastMCP server with SSE transport optimized for ChatGPT integration.

#### Legacy FastAPI Server

```bash
# Development
uv run mcp-python-baremetal http --host 0.0.0.0 --port 8000

# Production (after installation)
mcp-python-baremetal http --host 0.0.0.0 --port 8000
```

This starts an HTTP server with the following endpoints:

- `POST /execute` - Execute Python code
- `POST /execute/stream` - Execute with SSE streaming  
- `POST /analyze` - Analyze code for import requirements
- `POST /reset` - Reset the execution environment
- `GET /libraries` - List available libraries
- `GET /health` - Health check

### Direct CLI Execution

```bash
# Development examples
uv run mcp-python-baremetal execute "print(np.array([1,2,3]))"
uv run mcp-python-baremetal execute --file script.py
uv run mcp-python-baremetal execute --analyze-only "result = np.array([1,2,3]) + torch.ones(3)"
uv run mcp-python-baremetal execute --no-auto-import "import numpy as np; print(np.array([1,2,3]))"

# Production examples (after installation)
mcp-python-baremetal execute "print(np.array([1,2,3]))"
mcp-python-baremetal execute --file script.py
mcp-python-baremetal execute --analyze-only "result = np.array([1,2,3]) + torch.ones(3)"
mcp-python-baremetal execute --no-auto-import "import numpy as np; print(np.array([1,2,3]))"
```

### Other CLI Commands

```bash
# Development
uv run mcp-python-baremetal libraries  # List available libraries
uv run mcp-python-baremetal demo       # Run demo showcasing auto-import

# Production (after installation)
mcp-python-baremetal libraries
mcp-python-baremetal demo
```

## Auto-Import Feature

The server automatically detects when your code uses common library aliases without importing them and adds the appropriate imports:

**Input:**
```python
arr = np.array([1, 2, 3, 4, 5])
tensor = torch.tensor(arr)
result = torch.dot(tensor, tensor)
print("Result:", result)
```

**Automatically becomes:**
```python
import numpy as np
import torch

arr = np.array([1, 2, 3, 4, 5])
tensor = torch.tensor(arr)  
result = torch.dot(tensor, tensor)
print("Result:", result)
```

### Supported Auto-Imports

| Alias | Import Statement |
|-------|------------------|
| `np` | `import numpy as np` |
| `scipy` | `import scipy` |
| `sp` | `import scipy as sp` |
| `plt` | `import matplotlib.pyplot as plt` |
| `pd` | `import pandas as pd` |
| `torch` | `import torch` |
| `nn` | `from torch import nn` |
| `F` | `import torch.nn.functional as F` |
| `jax` | `import jax` |
| `jnp` | `import jax.numpy as jnp` |
| `sympy` | `import sympy` |
| `cp` | `import cvxpy as cp` |
| `cvx` | `import cvxpy as cvx` |
| `numba` | `import numba` |

## HTTP API Examples

### Execute Code

```bash
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(np.array([1,2,3]))",
    "auto_import": true
  }'
```

### Execute with SSE Streaming

```bash
curl -X POST "http://localhost:8000/execute/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "code": "for i in range(3): print(f\"Step {i}\")"
  }'
```

### Analyze Code

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"code": "result = np.array([1,2,3]) + torch.ones(3)"}'
```

## Python API

```python
from mcp_python_baremetal.executor import PythonExecutor

executor = PythonExecutor()

# Execute code with auto-import
result = executor.execute("print(np.array([1,2,3]))", auto_import=True)
print(f"Output: {result.stdout}")
print(f"Status: {result.execution_status}")

# Analyze imports
missing, existing = executor.analyze_imports("result = np.mean([1,2,3])")
print(f"Missing: {missing}")  # {'np'}

# Reset environment  
executor.reset_environment()
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/bugparty/mcp-python-baremetal
cd mcp-python-baremetal

# Create virtual environment and install dependencies
uv sync

# Install development dependencies (for testing, linting, etc.)
uv sync --extra dev
```

### Development Workflow

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=mcp_python_baremetal

# Code formatting and linting
uv run ruff check                 # Check for issues
uv run ruff check --fix          # Fix auto-fixable issues
uv run ruff format               # Format code

# Type checking
uv run mypy mcp_python_baremetal/

# Security check
uv run bandit -r mcp_python_baremetal/

# Run the application in development mode
uv run mcp-python-baremetal execute "print('Hello, World!')"
```

### Project Structure

```
mcp-python-baremetal/
├── mcp_python_baremetal/     # Main package
│   ├── __init__.py
│   ├── cli.py               # Command-line interface
│   ├── executor.py          # Python code execution engine
│   ├── fastmcp_server.py    # FastMCP server implementation
│   ├── http_server.py       # HTTP/SSE server implementation
│   └── server.py            # MCP server implementation
├── tests/                   # Test suite
├── examples/                # Usage examples
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
└── README.md               # This file
```

### Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv sync

# Remove a dependency
uv remove package-name
```

### Virtual Environment Management

The project uses `uv` for dependency management. The virtual environment is automatically created in `.venv/` when you run `uv sync`.

```bash
# Activate virtual environment manually (optional)
source .venv/bin/activate

# Run commands without uv run (when activated)
mcp-python-baremetal execute "print('Hello!')"

# Deactivate (when done)
deactivate
```

## Quick Reference

### Common Development Commands

| Task | Command |
|------|---------|
| Setup project | `uv sync` |
| Install dev dependencies | `uv sync --extra dev` |
| Run application | `uv run mcp-python-baremetal execute "code"` |
| Start MCP server | `uv run mcp-python-baremetal serve` |
| Start HTTP server | `uv run mcp-python-baremetal fastmcp --port 8000` |
| Run tests | `uv run pytest` |
| Format code | `uv run ruff format` |
| Check code | `uv run ruff check` |
| Type checking | `uv run mypy mcp_python_baremetal/` |
| Add dependency | `uv add package-name` |
| Update dependencies | `uv sync` |

### Troubleshooting

**Q: `uv sync` fails with dependency conflicts**  
A: Make sure you have Python 3.10+ installed. Check with `python --version`.

**Q: Commands hang or seem to freeze**  
A: Try running with timeout: `timeout 30 uv run mcp-python-baremetal execute "code"`

**Q: Import errors in development**  
A: Make sure you've run `uv sync` to install all dependencies.

**Q: Want to use system Python instead of virtual environment**  
A: You can still use `pip install -e .` for development, but `uv` is recommended.

## License

MIT License - see LICENSE file for details.