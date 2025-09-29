from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from datetime import datetime

from rag_core import two_stage_retrieval, build_context_block
from settings import LLM_MODEL
from llm_utils import chat_once


@dataclass
class PlanContext:
    text: str
    chunks: List[Dict[str, Any]]


class Planner:
    """Planner component that prepares machine-readable plans."""

    SYSTEM_PROMPT = """You are an expert planning assistant helping an autonomous code agent.
You must read the user's request and repository context, then reply with **valid JSON only**.
Follow this JSON schema:
{
  "objective": str,                      # succinct goal statement
  "rationale": str,                      # why this change is needed
  "scope": {
    "in_scope": [str],                   # bullet items describing covered work
    "out_of_scope": [str]                # explicit exclusions to avoid scope creep
  },
  "files": [
    {
      "path": str,                       # repository relative path
      "reason": str                      # why the file must be edited or read
    }
  ],
  "steps": [
    {
      "id": str,                         # short identifier e.g. "S1"
      "description": str,                # actionable instructions
      "depends_on": [str]                # list of step ids this step relies on
    }
  ],
  "tests": {
    "strategy": str,                     # how to validate the change manually or automatically
    "commands": [str]                    # shell commands to run (if any)
  },
  "risks": [
    {
      "description": str,                # what could go wrong
      "mitigation": str                  # how to prevent or detect the risk
    }
  ]
}
Rules:
- Always emit well-formed JSON without Markdown fences or commentary.
- Include every top-level key even if you need to supply an empty list.
- Prefer relative file paths exactly as they appear in the repository tree.
- Limit each "steps" item to a focused action suitable for automation.
"""

    def __init__(self, model: str = LLM_MODEL):
        self.model = model

    def _retrieve(self, request_text: str, top_k: int = 8) -> PlanContext:
        retrieved = two_stage_retrieval(request_text, top_k=top_k)
        context_text = build_context_block(retrieved)
        chunk_entries: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(retrieved):
            meta = chunk.meta or {}
            chunk_id = meta.get("key") or meta.get("id")
            if not chunk_id:
                src = meta.get("source") or meta.get("path") or "?"
                start = meta.get("start_line", "?")
                end = meta.get("end_line", "?")
                chunk_id = f"{src}:{start}-{end}"
            chunk_entries.append(
                {
                    "chunk_id": chunk_id,
                    "source": meta.get("source") or meta.get("path"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "score": chunk.score,
                    "kind": meta.get("kind"),
                }
            )
        return PlanContext(text=context_text, chunks=chunk_entries)

    def _parse_plan(self, raw_text: str) -> Dict[str, Any]:
        raw_text = raw_text.strip()
        if not raw_text:
            raise ValueError("Planner returned empty response")
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # try to find JSON block inside the text
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            snippet = raw_text[start : end + 1]
            return json.loads(snippet)

    def create_plan(self, request_text: str) -> Dict[str, Any]:
        """Return structured plan data and provenance info."""
        context = self._retrieve(request_text)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "User request:\n"
                    f"{request_text}\n\n"
                    "Repository context:\n"
                    f"{context.text}\n"
                ),
            },
        ]
        raw_plan = chat_once(self.model, messages, stream=False)
        plan_data = self._parse_plan(raw_plan)
        plan_data["schema_version"] = "1.0"
        return {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "raw_plan": raw_plan,
            "plan": plan_data,
            "context": {
                "text": context.text,
                "chunks": context.chunks,
            },
        }
