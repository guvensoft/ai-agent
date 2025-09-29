from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from unidiff import PatchSet, UnidiffParseError

from settings import LLM_MODEL
from llm_utils import chat_once


class PatchValidationError(RuntimeError):
    """Raised when an implementer response fails structural validation."""


class Implementer:
    """Generate unified diffs that obey a previously approved plan."""

    SYSTEM_PROMPT = """You are an expert software engineer implementing an approved plan.
Return a unified diff that can be applied with `git apply`. Follow these rules:
1. Modify only the files explicitly listed under `plan.files`.
2. Output **diffs only** (no Markdown code fences or explanations).
3. Include file headers (diff --git/---/+++) and well-formed hunks.
4. Preserve existing code style and imports; prefer minimal edits.
5. If no changes are required, output an empty string.
"""

    def __init__(self, model: str = LLM_MODEL):
        self.model = model

    def _build_messages(
        self,
        plan_record: Dict[str, Any],
        *,
        feedback: Optional[str] = None,
        previous_patch: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        plan_json = plan_record.get("plan") or plan_record.get("planner", {}).get("plan")
        if plan_json is None:
            raise ValueError("Plan data missing from record")
        context_text = (
            plan_record.get("planner", {}).get("context", {}).get("text")
            or plan_record.get("context", {}).get("text")
            or ""
        )
        allowed_files = [item.get("path") for item in plan_json.get("files", []) if item.get("path")]
        plan_dump = json.dumps(plan_json, ensure_ascii=False, indent=2)

        user_parts = [
            "Approved plan JSON:",
            plan_dump,
            "\nRepository context snippets:",
            context_text or "<no context>",
            "\nInstructions:",
            "- Produce a unified diff touching only the allowed files.",
            "- Respect the step ordering and intent described in the plan.",
        ]
        if allowed_files:
            user_parts.append("- Allowed files: " + ", ".join(allowed_files))
        if previous_patch:
            user_parts.append("- Previous patch (for reference, do not repeat verbatim):")
            user_parts.append(previous_patch)
        if feedback:
            user_parts.append("- Address the following feedback or test failures:")
            user_parts.append(feedback)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ]
        return messages

    def _validate_patch(self, patch_text: str, allowed_files: List[str]) -> Dict[str, Any]:
        patch = patch_text.strip()
        if not patch:
            raise PatchValidationError("Implementer returned an empty patch")
        if "diff --git" not in patch:
            raise PatchValidationError("Patch must start with 'diff --git' blocks")
        try:
            ps = PatchSet(patch.splitlines(keepends=True))
        except UnidiffParseError as exc:
            raise PatchValidationError(f"Patch could not be parsed: {exc}") from exc

        touched = []
        for pf in ps:
            target = (pf.target_file or pf.path or pf.source_file or "").replace("b/", "").replace("a/", "")
            if allowed_files and target not in allowed_files:
                raise PatchValidationError(f"Patch modifies '{target}' which is outside the approved file list")
            touched.append(target)
        if not touched:
            raise PatchValidationError("Patch contains no file modifications")
        return {"files": sorted(set(touched)), "patch_text": patch}

    def generate_patch(
        self,
        plan_record: Dict[str, Any],
        *,
        feedback: Optional[str] = None,
        previous_patch: Optional[str] = None,
    ) -> Dict[str, Any]:
        plan_json = plan_record.get("plan") or plan_record.get("planner", {}).get("plan")
        if plan_json is None:
            raise ValueError("Plan data missing from record")
        allowed_files = [item.get("path") for item in plan_json.get("files", []) if item.get("path")]

        messages = self._build_messages(
            plan_record,
            feedback=feedback,
            previous_patch=previous_patch,
        )
        raw_patch = chat_once(self.model, messages, stream=False)
        validation = self._validate_patch(raw_patch, allowed_files)
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "raw_response": raw_patch,
            "files": validation["files"],
            "patch": validation["patch_text"],
            "feedback": feedback,
        }
