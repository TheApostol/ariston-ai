"""Multi-tenant Auth + RBAC — Phase 4 / Ariston AI."""
from vinci_core.auth.rbac import RBACManager, Tenant, APIKey, rbac_manager, require_permission

__all__ = ["RBACManager", "Tenant", "APIKey", "rbac_manager", "require_permission"]
