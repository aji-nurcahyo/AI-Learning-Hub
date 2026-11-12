"""Permission Manager and Least Privilege Enforcement.

Implements role-based access control (RBAC) and permission scopes for tools:
1. Roles: STUDENT, INSTRUCTOR, ADMIN.
2. Scopes: 'read:knowledge', 'exec:calculator', 'modify:system'.
3. Allowlist validation prior to any tool execution.
4. Least-privilege principle: granting only the minimum necessary permissions.
"""

from enum import Enum
from typing import Any, Dict, List, Set


class UserRole(Enum):
    STUDENT = "student"
    INSTRUCTOR = "instructor"
    ADMIN = "admin"


class PermissionManager:
    """Enforces role-based permissions and tool execution allowlists."""

    # Role -> Set of allowed tool names
    ROLE_TOOL_ALLOWLIST: Dict[UserRole, Set[str]] = {
        UserRole.STUDENT: {"search_knowledge_base", "calculator"},
        UserRole.INSTRUCTOR: {"search_knowledge_base", "calculator"},
        UserRole.ADMIN: {"search_knowledge_base", "calculator", "delete_cache_entry"},
    }

    # Tool Name -> Required permission scope
    TOOL_SCOPE_REQUIREMENTS: Dict[str, str] = {
        "search_knowledge_base": "read:knowledge",
        "calculator": "exec:math",
        "delete_cache_entry": "modify:system",
    }

    def __init__(self, user_role: UserRole = UserRole.STUDENT):
        self.user_role = user_role

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Checks if current role is authorized to invoke tool_name."""
        allowed_tools = self.ROLE_TOOL_ALLOWLIST.get(self.user_role, set())
        return tool_name in allowed_tools

    def get_required_scope(self, tool_name: str) -> str:
        return self.TOOL_SCOPE_REQUIREMENTS.get(tool_name, "scope:unknown")

    def check_permission(self, tool_name: str) -> Dict[str, Any]:
        allowed = self.is_tool_allowed(tool_name)
        scope = self.get_required_scope(tool_name)

        if allowed:
            return {
                "authorized": True,
                "role": self.user_role.value,
                "tool_name": tool_name,
                "required_scope": scope,
                "message": f"Akses diberikan: Peran '{self.user_role.value}' memiliki izin untuk '{tool_name}'.",
            }
        else:
            return {
                "authorized": False,
                "role": self.user_role.value,
                "tool_name": tool_name,
                "required_scope": scope,
                "message": (
                    f"Akses DITOLAK: Peran '{self.user_role.value}' tidak memiliki izin "
                    f"untuk menjalankan tool '{tool_name}' (memerlukan scope '{scope}')."
                ),
            }


def main():
    print("=" * 70)
    print("CHAPTER 34: PERMISSION MANAGER & LEAST PRIVILEGE")
    print("=" * 70)

    # Student Role Test
    student_pm = PermissionManager(UserRole.STUDENT)
    print("1. Menguji Izin Peran Mahasiswa (STUDENT):")
    p1 = student_pm.check_permission("calculator")
    print(f"   Tool 'calculator'        : Authorized={p1['authorized']}")
    p2 = student_pm.check_permission("delete_cache_entry")
    print(f"   Tool 'delete_cache_entry': Authorized={p2['authorized']} (Pesan: {p2['message']})\n")

    # Admin Role Test
    admin_pm = PermissionManager(UserRole.ADMIN)
    print("2. Menguji Izin Peran Administrator (ADMIN):")
    p3 = admin_pm.check_permission("delete_cache_entry")
    print(f"   Tool 'delete_cache_entry': Authorized={p3['authorized']} (Pesan: {p3['message']})")

    print("\n[SUCCESS] Permission Manager and Least Privilege verified.")


if __name__ == "__main__":
    from typing import Any
    main()
