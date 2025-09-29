from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from orchestrator import sandbox_test_plan
from settings import STATIC_ANALYSIS_CMDS


class Verifier:
    """Run automated checks for a plan patch."""

    def run(self, plan_id: str, *, timeout_seconds: int = 300) -> Dict[str, Any]:
        extra = [cmd for cmd in (STATIC_ANALYSIS_CMDS or []) if cmd]
        result = sandbox_test_plan(plan_id, timeout_seconds=timeout_seconds, extra_commands=extra)
        tests = result.get("tests") or {}
        status = "passed" if tests.get("returncode") == 0 else "failed"

        summary_parts: List[str] = []
        if tests:
            rc = tests.get("returncode")
            summary_parts.append(f"tests rc={rc}")
        extra_checks = result.get("extra_checks") or []
        for chk in extra_checks:
            name = chk.get("name")
            rc = chk.get("returncode")
            status_str = chk.get("status") or ("passed" if rc == 0 else "failed")
            summary_parts.append(f"{name}:{status_str}")

        summary = "; ".join(summary_parts)
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": status,
            "summary": summary,
            "details": result,
        }
