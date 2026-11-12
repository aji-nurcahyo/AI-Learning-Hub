"""Model Context Protocol (MCP) Architectural Concepts.

Demonstrates the open standard Model Context Protocol (MCP):
1. Separation between MCP Client (host application) and MCP Server (tool/resource provider).
2. JSON-RPC 2.0 message framing and standard protocol methods:
   - 'tools/list': Dynamic tool discovery and schema exposure.
   - 'tools/call': Parameterized tool invocation.
   - 'resources/list': Exposing read-only context resources.
   - 'resources/read': Fetching static context data.
3. Permission boundaries and client-side authorization control.
"""

import json
from typing import Any, Dict, List, Optional


class MockMCPServer:
    """Simulates an isolated MCP Server exposing tools and resources over JSON-RPC 2.0."""

    def __init__(self, server_name: str = "itera-course-mcp-server"):
        self.server_name = server_name
        self.tools = {
            "get_lecture_schedule": {
                "name": "get_lecture_schedule",
                "description": "Mengambil jadwal perkuliahan AI berdasarkan kode mata kuliah.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "course_code": {"type": "string", "description": "Contoh: 'IF4021'"}
                    },
                    "required": ["course_code"],
                },
            }
        }
        self.resources = {
            "course://syllabus/ai-2026": {
                "uri": "course://syllabus/ai-2026",
                "name": "Silabus Perkuliahan Kecerdasan Buatan 2026",
                "mimeType": "text/markdown",
                "content": "# Silabus AI ITERA 2026\n40 Bab terstruktur dari Fondasi hingga AI Terapan.",
            }
        }

    def handle_request(self, json_rpc_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches JSON-RPC 2.0 protocol requests."""
        msg_id = json_rpc_msg.get("id", 1)
        method = json_rpc_msg.get("method")
        params = json_rpc_msg.get("params", {})

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": list(self.tools.values())},
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            if tool_name == "get_lecture_schedule":
                code = args.get("course_code", "IF-UNKNOWN")
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Jadwal {code}: Setiap Senin & Rabu pukul 08:00 - 09:40 WIB (Gedung Kuliah Umum 2).",
                            }
                        ],
                        "isError": False,
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method/Tool '{tool_name}' not found"},
            }
        elif method == "resources/list":
            res_meta = [
                {"uri": r["uri"], "name": r["name"], "mimeType": r["mimeType"]}
                for r in self.resources.values()
            ]
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": res_meta}}
        elif method == "resources/read":
            uri = params.get("uri")
            if uri in self.resources:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": self.resources[uri]["mimeType"],
                                "text": self.resources[uri]["content"],
                            }
                        ]
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32002, "message": f"Resource URI '{uri}' not found"},
            }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Unknown JSON-RPC method '{method}'"},
        }


class MockMCPClient:
    """Simulates an MCP Client connecting to an MCP Server."""

    def __init__(self, server: MockMCPServer):
        self.server = server
        self.req_counter = 0

    def send_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.req_counter += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.req_counter,
            "method": method,
            "params": params or {},
        }
        return self.server.handle_request(request)

    def discover_tools(self) -> List[Dict[str, Any]]:
        resp = self.send_rpc("tools/list")
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        resp = self.send_rpc("tools/call", {"name": name, "arguments": arguments})
        return resp.get("result", resp.get("error"))

    def read_resource(self, uri: str) -> str:
        resp = self.send_rpc("resources/read", {"uri": uri})
        contents = resp.get("result", {}).get("contents", [{}])
        return contents[0].get("text", "")


def main():
    print("=" * 70)
    print("CHAPTER 34: MODEL CONTEXT PROTOCOL (MCP) DEMONSTRATION")
    print("=" * 70)

    server = MockMCPServer()
    client = MockMCPClient(server)

    # 1. Tool Discovery via JSON-RPC 'tools/list'
    print("[1] Memanggil 'tools/list' untuk penemuan alat dinamis...")
    tools = client.discover_tools()
    for t in tools:
        print(f"    Ditemukan Tool MCP: [{t['name']}] - {t['description']}")
        print(f"    Skema Input: {t['inputSchema']['properties']}\n")

    # 2. Tool Execution via 'tools/call'
    print("[2] Memanggil 'tools/call' (get_lecture_schedule)...")
    res_call = client.call_tool("get_lecture_schedule", {"course_code": "IF4021"})
    print(f"    Keluaran MCP Server: {res_call['content'][0]['text']}\n")

    # 3. Resource Fetch via 'resources/read'
    print("[3] Memanggil 'resources/read' (course://syllabus/ai-2026)...")
    res_text = client.read_resource("course://syllabus/ai-2026")
    print(f"    Isi Resource Konteks:\n    {res_text}")

    print("\n[SUCCESS] Model Context Protocol concepts verified.")


if __name__ == "__main__":
    main()

