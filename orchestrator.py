# orchestrator.py
from pathlib import Path
import json
import traceback
import tempfile
import shutil
import subprocess
import os
from typing import Dict, Any
from settings import REPO_ROOT, GIT_COMMIT_AUTHOR, GIT_COMMIT_MESSAGE, CHROMA_DIR
from dev_workflow import PLANS_DIR, apply_patch_text
from git import Repo, InvalidGitRepositoryError



# Helper: run tests in given working dir
def run_tests_in_dir(wd: str, timeout: int = 300) -> Dict[str, Any]:
    """
    Runs tests using settings.TEST_CMD if present, otherwise 'pytest -q'.
    Returns dict with returncode, stdout, stderr.
    """
    from settings import TEST_CMD  # lazy import; add TEST_CMD to settings if you want custom
    cmd = TEST_CMD if hasattr(__import__("settings"), "TEST_CMD") else "pytest -q"
    if isinstance(cmd, str):
        cmd_list = cmd.split()  # simple split
    else:
        cmd_list = cmd
    try:
        res = subprocess.run(cmd_list, cwd=wd, capture_output=True, text=True, timeout=timeout)
        return {"returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}
    except subprocess.TimeoutExpired as e:
        return {"returncode": -1, "stdout": e.stdout or "", "stderr": f"TimeoutExpired: {e}"}
    except Exception as e:
        return {"returncode": -2, "stdout": "", "stderr": str(e)}

def _apply_patch_via_git_apply(patch_text: str, wd: str) -> Dict[str, Any]:
    """
    Writes patch_text to temp file and runs 'git apply <tmp>' in wd.
    Returns {"ok": bool, "stderr": ..., "stdout": ...}
    """
    tmp = Path(wd) / ".agent_tmp_patch.patch"
    tmp.write_text(patch_text, encoding="utf-8")
    try:
        proc = subprocess.run(["git", "apply", str(tmp)], cwd=wd, capture_output=True, text=True)
        ok = proc.returncode == 0
        return {"ok": ok, "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

def sandbox_test_plan(plan_id: str, timeout_seconds: int = 300) -> Dict[str, Any]:
    """
    Sandbox-run the plan's patch and tests.
    If repo at REPO_ROOT is a git repo, use a new branch there and git apply (no commit), run tests, then restore.
    If not a git repo, copy files to a temp dir, init git there, apply patch and run tests, then cleanup.
    Returns dictionary with applied flag, tests results, and errors/traces.
    """
    plan_path = Path(PLANS_DIR) / f"{plan_id}.json"
    if not plan_path.exists():
        return {"plan_id": plan_id, "applied": False, "error": "Plan not found: " + str(plan_path)}

    plan_obj = json.loads(plan_path.read_text(encoding="utf-8"))
    patch_text = plan_obj.get("patch", "")
    if not patch_text:
        return {"plan_id": plan_id, "applied": False, "error": "Plan has no patch."}

    # Try to use real repo if available
    try:
        repo = Repo(REPO_ROOT)
    except InvalidGitRepositoryError:
        repo = None
    except Exception as e:
        return {"plan_id": plan_id, "applied": False, "error": f"Git repo check failed: {e}", "traceback": traceback.format_exc()}

    if repo is not None:
        # Repo exists: ensure clean work tree
        try:
            if repo.is_dirty(untracked_files=True):
                return {"plan_id": plan_id, "applied": False, "error": "Repo dirty. Please commit/stash changes before sandbox testing."}
            orig_branch = None
            try:
                orig_branch = repo.active_branch.name
            except Exception:
                orig_branch = repo.head.commit.hexsha
            sandbox_branch = f"agent-sandbox-{plan_id}"
            # delete if exists
            if sandbox_branch in [h.name for h in repo.branches]:
                repo.git.branch("-D", sandbox_branch)
            repo.git.checkout("-b", sandbox_branch)
            apply_res = apply_patch_text(patch_text, str(REPO_ROOT))
            if not apply_res.get("ok"):
                # restore branch and return error
                try:
                    repo.git.checkout(orig_branch)
                    if sandbox_branch in [h.name for h in repo.branches]:
                        repo.git.branch("-D", sandbox_branch)
                except Exception:
                    pass
                return {"plan_id": plan_id, "applied": False, "error": "git apply failed", "apply": apply_res}
            # run tests (no commit)
            test_res = run_tests_in_dir(str(REPO_ROOT), timeout=timeout_seconds)
            # restore branch and remove sandbox branch
            try:
                repo.git.checkout(orig_branch)
                if sandbox_branch in [h.name for h in repo.branches]:
                    repo.git.branch("-D", sandbox_branch)
            except Exception:
                pass
            return {"plan_id": plan_id, "applied": True, "tests": test_res}
        except Exception as e:
            return {"plan_id": plan_id, "applied": False, "error": str(e), "traceback": traceback.format_exc()}
    else:
        # No git repo — create temp copy and init new git repo there
        tmpd = tempfile.mkdtemp(prefix="agent_sandbox_")
        try:
            # copy project files into tmp (skip .git and venv)
            def _ignore(p, names):
                ignore_list = set()
                for n in names:
                    ln = n.lower()
                    if ln == ".git" or ln.startswith(".venv") or ln == "venv" or "site-packages" in ln:
                        ignore_list.add(n)
                return ignore_list
            # copytree expects target not exists; using copy manually
            for item in Path(REPO_ROOT).iterdir():
                if item.name in (".git",) or item.name.lower().startswith(".venv") or item.name == "venv":
                    continue
                dest = Path(tmpd) / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "venv"))
                else:
                    shutil.copy2(item, dest)
            # init git in tmpd
            subprocess.run(["git", "init"], cwd=tmpd, check=True, capture_output=True)
            # configure user if needed
            subprocess.run(["git", "config", "user.email", "ai-agent@local"], cwd=tmpd)
            subprocess.run(["git", "config", "user.name", "ai-agent"], cwd=tmpd)
            # initial commit
            subprocess.run(["git", "add", "."], cwd=tmpd)
            subprocess.run(["git", "commit", "-m", "agent: sandbox init"], cwd=tmpd, capture_output=True)
            # apply patch
            apply_res = apply_patch_text(patch_text, tmpd)
            if not apply_res.get("ok"):
                return {"plan_id": plan_id, "applied": False, "error": "git apply failed in temp repo", "apply": apply_res}
            # run tests
            test_res = run_tests_in_dir(tmpd, timeout=timeout_seconds)
            return {"plan_id": plan_id, "applied": True, "tests": test_res}
        except Exception as e:
            return {"plan_id": plan_id, "applied": False, "error": str(e), "traceback": traceback.format_exc()}
        finally:
            try:
                shutil.rmtree(tmpd)
            except Exception:
                pass



# --- Backwards-compatible helpers for app.py imports ---

# import dev_make_plan (create plan + patch)
try:
    from dev_workflow import make_plan as dev_make_plan
except Exception:
    dev_make_plan = None

def create_plan(request_text: str) -> Dict[str, Any]:
    """
    Wrapper around dev_workflow.make_plan to provide same API as older orchestrator.
    Returns whatever dev_make_plan returns (plan_id, plan, patch_preview, ...)
    """
    if dev_make_plan is None:
        raise RuntimeError("dev_workflow.make_plan bulunamadı; dev_workflow modülünü kontrol et.")
    return dev_make_plan(request_text)

def get_plan(plan_id: str) -> Dict[str, Any]:
    """
    Read saved plan JSON from PLANS_DIR and return parsed object.
    """
    p = Path(PLANS_DIR) / f"{plan_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"Plan bulunamadı: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
