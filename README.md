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

## Installation

Using uvx (recommended):

```bash
uvx install mcp-python-baremetal
```

Or with pip:

```bash
pip install mcp-python-baremetal
```

## Usage

### As MCP Server (for Claude)

```bash
mcp-python-baremetal serve
```

This starts the MCP server with stdio transport that Claude can connect to.

### As HTTP/SSE Server (for ChatGPT Developer)

```bash
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
# Execute code directly
mcp-python-baremetal execute "print(np.array([1,2,3]))"

# Execute from file
mcp-python-baremetal execute --file script.py

# Analyze imports only
mcp-python-baremetal execute --analyze-only "result = np.array([1,2,3]) + torch.ones(3)"

# Disable auto-import
mcp-python-baremetal execute --no-auto-import "import numpy as np; print(np.array([1,2,3]))"
```

### Other CLI Commands

```bash
# List available libraries and their status
mcp-python-baremetal libraries

# Run demo showcasing auto-import
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

```bash
git clone https://github.com/bugparty/mcp-python-baremetal
cd mcp-python-baremetal

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## License

MIT License - see LICENSE file for details.