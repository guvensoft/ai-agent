from __future__ import annotations

import json
import shlex
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from git import Actor, Repo
from unidiff import PatchSet

from implementer import Implementer, PatchValidationError
from planner import Planner
from settings import (
    GIT_COMMIT_AUTHOR,
    GIT_COMMIT_MESSAGE,
    MEMORY_DIR,
    REPO_ROOT,
    TEST_CMD,
)

PLANS_DIR = MEMORY_DIR / "plans"
PLANS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _plan_path(plan_id: str) -> Path:
    return PLANS_DIR / f"{plan_id}.json"


def _load_plan(plan_id: str) -> Dict[str, Any]:
    path = _plan_path(plan_id)
    if not path.exists():
        raise FileNotFoundError(f"Plan not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_plan(plan_id: str, data: Dict[str, Any]) -> None:
    path = _plan_path(plan_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Planning stage
# ---------------------------------------------------------------------------

def make_plan(request_text: str) -> Dict[str, Any]:
    planner = Planner()
    planner_result = planner.create_plan(request_text)

    plan_id = f"plan_{uuid.uuid4().hex[:8]}"
    record: Dict[str, Any] = {
        "plan_id": plan_id,
        "request": request_text,
        "created_at": planner_result.get("created_at", _now_iso()),
        "planner": planner_result,
        "plan": planner_result.get("plan", {}),
        "implementation": {"attempts": [], "final_patch": None},
        "verification": {"attempts": [], "status": "pending"},
        "logs": [],
    }
    _save_plan(plan_id, record)

    return {
        "plan_id": plan_id,
        "plan": record["plan"],
        "plan_raw": planner_result.get("raw_plan"),
        "context": planner_result.get("context", {}),
        "created_at": record["created_at"],
        "request": request_text,
    }


# ---------------------------------------------------------------------------
# Implementation stage
# ---------------------------------------------------------------------------

def _ensure_impl_section(record: Dict[str, Any]) -> Dict[str, Any]:
    impl = record.setdefault("implementation", {})
    impl.setdefault("attempts", [])
    impl.setdefault("final_patch", None)
    return impl


def implement_plan(plan_id: str, *, feedback: Optional[str] = None) -> Dict[str, Any]:
    record = _load_plan(plan_id)
    impl_section = _ensure_impl_section(record)
    implementer = Implementer()
    previous_patch = impl_section.get("final_patch")

    try:
        attempt = implementer.generate_patch(
            {
                "plan": record.get("plan"),
                "planner": record.get("planner"),
            },
            feedback=feedback,
            previous_patch=previous_patch,
        )
    except PatchValidationError as exc:
        error_entry = {
            "timestamp": _now_iso(),
            "error": str(exc),
            "feedback": feedback,
        }
        impl_section.setdefault("attempts", []).append(error_entry)
        record["implementation"] = impl_section
        _save_plan(plan_id, record)
        raise

    impl_section.setdefault("attempts", []).append(attempt)
    impl_section["final_patch"] = attempt["patch"]
    record["implementation"] = impl_section
    record["patch"] = attempt["patch"]
    _save_plan(plan_id, record)

    return {
        "plan_id": plan_id,
        "attempt": attempt,
        "files": attempt["files"],
        "patch": attempt["patch"],
    }


# ---------------------------------------------------------------------------
# Verification stage
# ---------------------------------------------------------------------------

def verify_plan(plan_id: str, *, auto_fix: bool = True, max_rounds: int = 3) -> Dict[str, Any]:
    from verifier import Verifier  # Lazy import to avoid circular dependency

    record = _load_plan(plan_id)
    impl_section = _ensure_impl_section(record)
    if not impl_section.get("final_patch"):
        raise RuntimeError("Plan has no generated patch. Run implement_plan first.")

    verification = record.setdefault("verification", {"attempts": [], "status": "pending"})
    attempts: List[Dict[str, Any]] = verification.setdefault("attempts", [])

    verifier = Verifier()
    rounds = 0
    final_status = verification.get("status", "pending")

    while rounds < max_rounds:
        rounds += 1
        result = verifier.run(plan_id)
        result["round"] = len(attempts) + 1
        attempts.append(result)
        final_status = result.get("status") or "failed"
        verification["status"] = final_status
        record["verification"] = verification
        _save_plan(plan_id, record)

        if final_status == "passed":
            break
        if not auto_fix or rounds >= max_rounds:
            break

        # Attempt self-correction by invoking implementer with feedback
        details = result.get("details", {})
        tests = details.get("tests", {}) if isinstance(details, dict) else {}
        feedback_parts = [result.get("summary") or ""]
        stderr = tests.get("stderr") or ""
        if stderr:
            feedback_parts.append(f"Test stderr:\n{stderr}")
        stdout = tests.get("stdout") or ""
        if stdout:
            feedback_parts.append(f"Test stdout (truncated):\n{stdout[-2000:]}")
        extra_checks = details.get("extra_checks") if isinstance(details, dict) else []
        for chk in extra_checks or []:
            feedback_parts.append(
                f"Check {chk.get('name')}: status={chk.get('status')} rc={chk.get('returncode')} stderr={chk.get('stderr')}"
            )
        feedback = "\n\n".join([part for part in feedback_parts if part])
        if not feedback:
            break
        try:
            implement_plan(plan_id, feedback=feedback)
            record = _load_plan(plan_id)
            impl_section = _ensure_impl_section(record)
            verification = record.setdefault("verification", verification)
            attempts = verification.setdefault("attempts", attempts)
        except PatchValidationError:
            break

    return {"plan_id": plan_id, "status": final_status, "attempts": attempts}


# ---------------------------------------------------------------------------
# Patch utilities and application helpers
# ---------------------------------------------------------------------------

def parse_patch_hunks(patch_text: str) -> List[Dict[str, Any]]:
    lines = patch_text.splitlines()
    blocks: List[List[str]] = []
    current: List[str] = []
    for ln in lines:
        if ln.startswith("diff --git"):
            if current:
                blocks.append(current)
            current = [ln]
        else:
            if current:
                current.append(ln)
    if current:
        blocks.append(current)

    parsed: List[Dict[str, Any]] = []
    for block in blocks:
        target = None
        for ln in block:
            if ln.startswith("+++ "):
                target = ln.split(" ", 1)[1].strip()
                if target.startswith("b/"):
                    target = target[2:]
                break
        if not target and block:
            header_parts = block[0].split()
            if len(header_parts) >= 4:
                target = header_parts[3]
                if target.startswith("b/"):
                    target = target[2:]
        hunks: List[str] = []
        hunk_lines: List[str] = []
        for ln in block:
            if ln.startswith("@@"):
                if hunk_lines:
                    hunks.append("\n".join(hunk_lines))
                hunk_lines = [ln]
            else:
                if hunk_lines:
                    hunk_lines.append(ln)
        if hunk_lines:
            hunks.append("\n".join(hunk_lines))
        parsed.append({"file": target or "unknown", "hunks": hunks})
    return parsed


def get_plan_hunks(plan_id: str) -> List[Dict[str, Any]]:
    record = _load_plan(plan_id)
    patch = record.get("implementation", {}).get("final_patch") or record.get("patch") or ""
    if not patch:
        return []
    return parse_patch_hunks(patch)


def make_plan_files(plan_id: str) -> Dict[str, Any]:
    record = _load_plan(plan_id)
    plan = record.get("plan", {})
    files = [item.get("path") for item in plan.get("files", []) if item.get("path")]
    return {"plan_id": plan_id, "files": files}


def _write_tmp_patch(patch_text: str, tmp_path: Path) -> Path:
    normalized = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in normalized.split("\n")]
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def apply_patch_text(patch_text: str, wd: str = str(REPO_ROOT)) -> Dict[str, Any]:
    tmp = Path(wd) / ".agent_patch.tmp"
    attempts: List[Dict[str, Any]] = []

    def _run(args: List[str]) -> Dict[str, Any]:
        proc = subprocess.run(args, cwd=wd, capture_output=True, text=True)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "args": args,
        }

    try:
        _write_tmp_patch(patch_text, tmp)
        attempts.append(_run(["git", "apply", "--whitespace=fix", str(tmp)]))
        if attempts[-1]["returncode"] == 0:
            return {"ok": True, "attempts": attempts}

        # Try creating missing files/directories then retry
        try:
            ps = PatchSet(tmp.read_text(encoding="utf-8").splitlines(True))
            for pf in ps:
                target = pf.target_file or pf.path
                if not target:
                    continue
                target = str(target)
                if target.startswith("a/") or target.startswith("b/"):
                    target = target[2:]
                target_path = Path(wd) / target
                if not target_path.exists():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if not pf.is_added_file:
                        target_path.write_text("", encoding="utf-8")
        except Exception:
            pass

        attempts.append(_run(["git", "apply", "--whitespace=fix", str(tmp)]))
        if attempts[-1]["returncode"] == 0:
            return {"ok": True, "attempts": attempts, "note": "created_missing_files"}

        # Serialize via unidiff to fix format oddities
        try:
            ps = PatchSet(tmp.read_text(encoding="utf-8").splitlines(True))
            tmp.write_text(str(ps), encoding="utf-8")
            attempts.append(_run(["git", "apply", "--whitespace=fix", str(tmp)]))
            if attempts[-1]["returncode"] == 0:
                return {"ok": True, "attempts": attempts, "note": "repaired_with_unidiff"}
        except Exception as exc:
            attempts.append({"returncode": -1, "stdout": "", "stderr": f"unidiff parse failed: {exc}"})

        # Final fallback: git apply --reject
        attempts.append(_run(["git", "apply", "--reject", "--whitespace=fix", str(tmp)]))
        rejected: List[Dict[str, Any]] = []
        for rej_file in Path(wd).rglob("*.rej"):
            try:
                rejected.append({"path": str(rej_file.relative_to(wd)), "content": rej_file.read_text(encoding="utf-8")})
            except Exception:
                rejected.append({"path": str(rej_file), "content": "<unreadable>"})
        return {"ok": False, "attempts": attempts, "rejected": rejected}
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Git utilities
# ---------------------------------------------------------------------------

def commit_changes(message_suffix: str = "") -> str:
    repo = Repo(REPO_ROOT)
    author = Actor(GIT_COMMIT_AUTHOR[0], GIT_COMMIT_AUTHOR[1])
    repo.git.add(all=True)
    repo.index.commit(
        GIT_COMMIT_MESSAGE + (f" {message_suffix}" if message_suffix else ""),
        author=author,
        committer=author,
    )
    return repo.head.commit.hexsha


def run_tests() -> Dict[str, Any]:
    cmd = TEST_CMD
    if isinstance(cmd, str):
        args = shlex.split(cmd)
    else:
        args = list(cmd)
    try:
        proc = subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr": "Tests timed out."}
    except Exception as exc:
        return {"returncode": 1, "stdout": "", "stderr": str(exc)}


def apply_plan(plan_id: str) -> Dict[str, Any]:
    record = _load_plan(plan_id)
    patch = record.get("implementation", {}).get("final_patch") or record.get("patch")
    if not patch:
        raise RuntimeError("No patch stored for plan. Run implement_plan first.")
    apply_patch_text(patch)
    commit = commit_changes()
    tests = run_tests()
    return {"applied": True, "commit": commit, "tests": tests}


def apply_plan_files(plan_id: str, files: List[str]) -> Dict[str, Any]:
    record = _load_plan(plan_id)
    patch = record.get("implementation", {}).get("final_patch") or record.get("patch")
    if not patch:
        raise RuntimeError("No patch stored for plan. Run implement_plan first.")
    ps = PatchSet(patch.splitlines(True))
    out_blocks: List[str] = []
    for pf in ps:
        target = (pf.target_file or pf.path or "").replace("b/", "").replace("a/", "")
        if target not in files:
            continue
        header = f"diff --git a/{target} b/{target}\n--- a/{target}\n+++ b/{target}\n"
        hunks: List[str] = []
        for h in pf:
            h_lines = [str(h.header).rstrip()]
            for line in h:
                try:
                    h_lines.append(line.value.rstrip("\n"))
                except Exception:
                    h_lines.append(str(line))
            hunks.append("\n".join([ln for ln in h_lines if ln]))
        if hunks:
            out_blocks.append(header + "\n".join(hunks))
    if not out_blocks:
        raise RuntimeError("No matching files found in patch")
    partial_patch = "\n".join(out_blocks)
    apply_patch_text(partial_patch)
    commit = commit_changes("(partial)")
    tests = run_tests()
    return {"applied": True, "commit": commit, "tests": tests, "files": files}


def apply_plan_hunks(plan_id: str, selections: Dict[str, List[int]]) -> Dict[str, Any]:
    record = _load_plan(plan_id)
    patch = record.get("implementation", {}).get("final_patch") or record.get("patch")
    if not patch:
        raise RuntimeError("No patch stored for plan. Run implement_plan first.")
    parsed = parse_patch_hunks(patch)
    out_blocks: List[str] = []
    for block in parsed:
        path = block.get("file")
        chosen = selections.get(path, [])
        if not chosen:
            continue
        header = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        hunks = [block["hunks"][idx] for idx in chosen if 0 <= idx < len(block["hunks"]) ]
        if hunks:
            out_blocks.append(header + "\n".join(hunks))
    if not out_blocks:
        raise RuntimeError("No hunks selected")
    partial_patch = "\n".join(out_blocks)
    apply_patch_text(partial_patch)
    commit = commit_changes("(hunks)")
    tests = run_tests()
    return {"applied": True, "commit": commit, "tests": tests, "selections": selections}


def revert_commit(commit_sha: str) -> Dict[str, Any]:
    repo = Repo(REPO_ROOT)
    if repo.is_dirty(untracked_files=True):
        raise RuntimeError("Working tree is dirty. Commit or stash changes first.")
    repo.git.revert("--no-commit", commit_sha)
    author = Actor(GIT_COMMIT_AUTHOR[0], GIT_COMMIT_AUTHOR[1])
    repo.git.add(all=True)
    repo.index.commit(f"revert(agent): {commit_sha}", author=author, committer=author)
    return {"reverted": True, "new_commit": repo.head.commit.hexsha}
