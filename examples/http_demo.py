#!/usr/bin/env python3
"""Demo script showing HTTP API usage."""

import json
import httpx
import asyncio

async def demo_http_api():
    """Demonstrate the HTTP API functionality."""
    base_url = "http://localhost:8002"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("🚀 MCP Python Baremetal HTTP API Demo")
        
        # Health check
        print("1. Health Check")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")
            print("   Make sure server is running: mcp-python-baremetal http --port 8002")
            return
        
        # Execute with auto-import
        print("2. Execute with Auto-Import")
        payload = {"code": "arr = np.array([1,2,3]); print(arr)", "auto_import": True}
        response = await client.post(f"{base_url}/execute", json=payload)
        result = response.json()
        print(f"   Status: {result[\"execution_status\"]}")
        print(f"   Output: {result[\"stdout\"].strip()}")

if __name__ == "__main__":
    print("Run: mcp-python-baremetal http --port 8002")
    asyncio.run(demo_http_api())
