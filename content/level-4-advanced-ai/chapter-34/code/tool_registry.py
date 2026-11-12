"""Tool Registry for AI Agents.

Implements a central registry managing discoverable tools:
1. Tool metadata: name, description, risk_level (READ_ONLY vs STATE_CHANGING).
2. JSON Schema definitions for parameters.
3. Callable implementation binding.
4. Tool discovery and capability listing.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ToolRiskLevel(Enum):
    READ_ONLY = "read_only"
    STATE_CHANGING = "state_changing"


class ToolDefinition:
    """Represents a registered tool with its execution schema and risk profile."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
        risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.risk_level = risk_level

    def to_schema(self) -> Dict[str, Any]:
        """Returns JSON schema definition for LLM tool declaration."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "risk_level": self.risk_level.value,
            },
        }

    def execute(self, **kwargs) -> Any:
        return self.handler(**kwargs)


class ToolRegistry:
    """Manages registration, discovery, and lookup of agent tools."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
        risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY,
    ):
        tool = ToolDefinition(name, description, parameters, handler, risk_level)
        self._tools[name] = tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def is_state_changing(self, name: str) -> bool:
        tool = self.get_tool(name)
        return tool.risk_level == ToolRiskLevel.STATE_CHANGING if tool else False

    def __len__(self) -> int:
        return len(self._tools)


# Sample Tool Implementations (Safe, Local, Read-Only)
def local_calculator(expression: str) -> float:
    """Evaluates basic arithmetic expressions safely using an allowed token parser."""
    import re

    # Strict whitelist: digits, spaces, basic operators +, -, *, /, (, )
    if not re.match(r"^[\d\s\+\-\*\/\(\)\.]+$", expression):
        raise ValueError(f"Ekspresi mengandung karakter berbahaya atau tidak diizinkan: '{expression}'")

    # Safe evaluation of mathematical tokens
    # Note: Using ast or simple restricted eval for arithmetic
    clean_expr = expression.strip()
    result = eval(clean_expr, {"__builtins__": None}, {})
    return float(result)


def search_knowledge_base(query: str, max_results: int = 2) -> List[Dict[str, str]]:
    """Simulated local knowledge base retrieval tool (Read-Only)."""
    # Deterministic mock response for educational testing
    return [
        {
            "doc_id": "doc_rag_01",
            "title": "Pengantar RAG",
            "snippet": f"Hasil pencarian untuk '{query}': RAG menggabungkan model bahasa parametrik dengan basis data eksternal.",
        }
    ]


def delete_cache_entry(entry_id: str) -> str:
    """Simulated state-changing cache deletion tool (State-Changing, High-Risk)."""
    return f"SUCCESS: Cache entry '{entry_id}' berhasil dihapus dari penyimpanan lokal."


def build_default_registry() -> ToolRegistry:
    """Builds and returns the standard reference tool registry."""
    registry = ToolRegistry()

    # Tool 1: Calculator (Read-Only)
    registry.register_tool(
        name="calculator",
        description="Melakukan operasi matematika aritmatika dasar (penjumlahan, pengurangan, perkalian, pembagian).",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Ekspresi aritmatika yang akan dihitung, contoh: '15 * 4 + 20'",
                }
            },
            "required": ["expression"],
        },
        handler=local_calculator,
        risk_level=ToolRiskLevel.READ_ONLY,
    )

    # Tool 2: Knowledge Base Search (Read-Only)
    registry.register_tool(
        name="search_knowledge_base",
        description="Mencari artikel materi kuliah dan konsep AI dalam basis pengetahuan lokal.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Kata kunci pencarian konsep kecerdasan buatan.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Jumlah dokumen maksimal yang ingin diambil.",
                },
            },
            "required": ["query"],
        },
        handler=search_knowledge_base,
        risk_level=ToolRiskLevel.READ_ONLY,
    )

    # Tool 3: Delete Cache (State-Changing)
    registry.register_tool(
        name="delete_cache_entry",
        description="Menghapus berkas cache sistem (Operasi permanen yang memerlukan izin persetujuan manusia).",
        parameters={
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "ID entri cache yang akan dihapus.",
                }
            },
            "required": ["entry_id"],
        },
        handler=delete_cache_entry,
        risk_level=ToolRiskLevel.STATE_CHANGING,
    )

    return registry


def main():
    print("=" * 70)
    print("CHAPTER 34: AGENT TOOL REGISTRY")
    print("=" * 70)

    reg = build_default_registry()
    print(f"Total tools terdaftar: {len(reg)}\n")

    print("Daftar Tool dan Profil Risiko:")
    for t in reg.list_tools():
        fn = t["function"]
        print(f"- Tool: [{fn['name']}] ({fn['risk_level'].upper()})")
        print(f"  Deskripsi: {fn['description']}")
        print(f"  Parameter: {list(fn['parameters']['properties'].keys())}\n")

    # Test Execution
    calc = reg.get_tool("calculator")
    ans = calc.execute(expression="100 / 4 + 15")
    print(f"Uji Eksekusi Tool 'calculator': 100 / 4 + 15 = {ans}")

    print("\n[SUCCESS] Tool Registry operational with risk-level classification.")


if __name__ == "__main__":
    main()

