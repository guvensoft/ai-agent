import json, subprocess, shlex
from pathlib import Path
from typing import Dict, Any
from unidiff import PatchSet
from settings import MEMORY_DIR, LLM_MODEL, GIT_COMMIT_AUTHOR, GIT_COMMIT_MESSAGE, REPO_ROOT, TEST_CMD
from agent.utils.ollama_helpers import ollama_chat_once
from rag_core import get_retriever, to_retrieved_chunks, build_context_block, rerank
import ollama
from git import Repo, Actor


PLANS_DIR = MEMORY_DIR / "plans"
PLANS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_FOR_PLAN = "You are a senior engineer. Make a plan using repo context."
SYSTEM_FOR_DIFF = "Produce a unified diff (patch) implementing the plan. Output only the patch."

def revert_commit(commit_sha: str):
    """
    Verilen commit SHA'sını geri alır (git revert --no-commit).
    Çalışma dizininin temiz olması gerekir.
    Dönen dict: {"reverted": True, "new_commit": "<sha>"}
    """
    repo = Repo(REPO_ROOT)
    if repo.is_dirty(untracked_files=True):
        raise RuntimeError("Çalışma dizini temiz değil. Lütfen değişiklikleri stash/commit edin.")
    # no-commit revert (değişiklikleri çalışma dizinine uygular, commit etmez)
    repo.git.revert("--no-commit", commit_sha)
    author = Actor(GIT_COMMIT_AUTHOR[0], GIT_COMMIT_AUTHOR[1])
    repo.git.add(all=True)
    repo.index.commit(f"revert(agent): {commit_sha}", author=author, committer=author)
    return {"reverted": True, "new_commit": repo.head.commit.hexsha}


def _ollama_chat(system: str, user: str):
    """
    Safe wrapper around the Ollama chat API. Builds messages and uses
    `ollama_chat_once` to obtain a single assistant response.
    """
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return ollama_chat_once(LLM_MODEL, msgs, stream=False)

def make_plan(request_text: str):
    retriever = get_retriever(filters={"kind":{"$eq":"code"}}, k=12, collection_name="repo")
    docs = retriever.invoke(request_text)
    docs = rerank(request_text, docs, top_k=8)
    ctx = build_context_block(to_retrieved_chunks(docs))
    plan_text = _ollama_chat(SYSTEM_FOR_PLAN + (ctx or ""), f"Request:\n{request_text}")
    patch_text = _ollama_chat(SYSTEM_FOR_DIFF + (ctx or ""), f"PLAN:\n{plan_text}\nPatch:")
    plan_id = f"plan_{abs(hash(plan_text))}"
    (PLANS_DIR / f"{plan_id}.json").write_text(json.dumps({"request": request_text, "plan": plan_text, "patch": patch_text}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"plan_id": plan_id, "plan": plan_text, "patch_preview": patch_text[:4000], "patch": patch_text}

def make_plan_files(plan_id: str):
    obj = json.loads((PLANS_DIR / f"{plan_id}.json").read_text(encoding="utf-8"))
    ps = PatchSet(obj["patch"].splitlines(keepends=True))
    files = []
    for p in ps:
        path = str(p.target_file or p.path).replace("b/","").replace("a/","")
        if path not in files:
            files.append(path)
    return {"plan_id": plan_id, "files": files}

def parse_patch_hunks(patch_text: str):
    # Split by 'diff --git' blocks
    lines = patch_text.splitlines(keepends=False)
    blocks = []
    cur = []
    for ln in lines:
        if ln.startswith("diff --git"):
            if cur:
                blocks.append(cur)
            cur = [ln]
        else:
            if cur:
                cur.append(ln)
            else:
                # skip leading lines until first diff
                continue
    if cur:
        blocks.append(cur)
    result = []
    for block in blocks:
        # determine file path from '+++ b/...' line if present
        target = None
        for ln in block:
            if ln.startswith("+++ "):
                target = ln.split(" ",1)[1].strip()
                if target.startswith("b/"):
                    target = target[2:]
                break
        if not target:
            # fallback: try to parse from diff --git line
            header = block[0]
            parts = header.split()
            if len(parts) >= 3:
                a = parts[2]
                if a.startswith("a/"):
                    target = a[2:]
                else:
                    target = a
        # Now split hunks by lines starting with @@
        hunks = []
        cur_hunk = []
        for ln in block:
            if ln.startswith("@@"):
                if cur_hunk:
                    hunks.append("\n".join(cur_hunk))
                cur_hunk = [ln]
            else:
                if cur_hunk:
                    cur_hunk.append(ln)
        if cur_hunk:
            hunks.append("\n".join(cur_hunk))
        result.append({"file": target or "unknown", "hunks": hunks})
    return result

def get_plan_hunks(plan_id: str):
    obj = json.loads((PLANS_DIR / f"{plan_id}.json").read_text(encoding="utf-8"))
    patch = obj.get("patch","")
    return parse_patch_hunks(patch)

def _write_tmp_patch(patch_text: str, tmp_path: Path):
    # normalize line endings to LF and ensure trailing newline
    text = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    # do NOT strip meaningful leading spaces; only rstrip trailing spaces
    lines = [ln.rstrip() for ln in text.split("\n")]
    norm = "\n".join(lines) + "\n"
    tmp_path.write_text(norm, encoding="utf-8")
    return tmp_path

def _try_git_apply(args, wd: str) -> Dict[str, Any]:
    proc = subprocess.run(args, cwd=wd, capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "args": args}

 
def _write_tmp_patch(patch_text: str, tmp_path: Path):
    # normalize line endings to LF and strip trailing spaces (only trailing)
    text = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    norm = "\n".join(lines) + "\n"
    tmp_path.write_text(norm, encoding="utf-8")
    return tmp_path

def _try_git_apply(args, wd: str) -> Dict[str, Any]:
    proc = subprocess.run(args, cwd=wd, capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "args": args}






def _write_tmp_patch_normalized(patch_text: str, tmp_path: Path):
    # normalize CRLF->LF, strip trailing spaces only, keep leading spaces
    text = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    norm = "\n".join(lines) + "\n"
    tmp_path.write_text(norm, encoding="utf-8")
    return tmp_path

def _run_git_apply(args, wd: str) -> Dict[str, Any]:
    proc = subprocess.run(args, cwd=wd, capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "args": args}

def apply_patch_text(patch_text: str, wd: str) -> Dict[str, Any]:
    """
    Çok aşamalı patch apply. Döndürür:
      { "ok": bool, "attempts": [...], "repaired_with_unidiff": bool?, "rejected": [...] }
    """
    tmp = Path(wd) / ".agent_tmp_patch.patch"
    attempts = []

    # 1) Normalize + git apply --whitespace=fix
    _write_tmp_patch_normalized(patch_text, tmp)
    attempts.append(_run_git_apply(["git", "apply", "--whitespace=fix", str(tmp)], wd))
    if attempts[-1]["returncode"] == 0:
        tmp.unlink(missing_ok=True)
        return {"ok": True, "attempts": attempts}

    # 2) Eğer 'No such file or directory' türü hata varsa, parse patch ve oluşturulması gereken path'leri yarat
    stderr = attempts[-1]["stderr"] or ""
    if "No such file or directory" in stderr or "error: " in stderr:
        try:
            ps = PatchSet(tmp.read_text(encoding="utf-8").splitlines(True))
            # for each patched file, ensure parent dirs exist; if file missing and patch is not an 'added file', create empty file
            for pf in ps:
                # target file path (prefer b/ style)
                target = pf.target_file or pf.path
                if not target:
                    continue
                # strip possible a/ or b/ prefix
                tname = str(target)
                if tname.startswith("a/") or tname.startswith("b/"):
                    tname = tname[2:]
                fpath = Path(wd) / tname
                if not fpath.exists():
                    # if patch is an "added" file (pf.is_added_file), git apply can create it; otherwise create empty
                    try:
                        if not pf.is_added_file:
                            fpath.parent.mkdir(parents=True, exist_ok=True)
                            fpath.write_text("", encoding="utf-8")
                    except Exception:
                        pass
        except Exception:
            # parse hatası; geç
            pass

        # tekrar dene
        attempts.append(_run_git_apply(["git", "apply", "--whitespace=fix", str(tmp)], wd))
        if attempts[-1]["returncode"] == 0:
            tmp.unlink(missing_ok=True)
            return {"ok": True, "attempts": attempts, "note": "created_missing_files_before_apply"}

    # 3) unidiff ile reserialize (bazı biçim hatalarını düzeltir)
    try:
        ps = PatchSet(tmp.read_text(encoding="utf-8").splitlines(True))
        if len(ps) > 0:
            repaired = str(ps)
            tmp.write_text(repaired, encoding="utf-8")
            attempts.append(_run_git_apply(["git", "apply", "--whitespace=fix", str(tmp)], wd))
            if attempts[-1]["returncode"] == 0:
                tmp.unlink(missing_ok=True)
                return {"ok": True, "attempts": attempts, "repaired_with_unidiff": True}
    except Exception as e:
        attempts.append({"returncode": -1, "stdout": "", "stderr": f"unidiff parse failed: {e}"})

    # 4) son çare: --reject ile uygula ve .rej dosyalarını topla
    attempts.append(_run_git_apply(["git", "apply", "--reject", "--whitespace=fix", str(tmp)], wd))
    rejected = []
    try:
        for p in Path(wd).rglob("*.rej"):
            try:
                rejected.append({"path": str(p.relative_to(wd)), "content": p.read_text(encoding="utf-8")})
            except Exception:
                rejected.append({"path": str(p.relative_to(wd)), "content": "<read error>"})
    except Exception:
        pass

    tmp.unlink(missing_ok=True)
    return {"ok": False, "attempts": attempts, "rejected": rejected}

def commit_changes(message_suffix=""):
    repo = Repo(REPO_ROOT)
    author = Actor(GIT_COMMIT_AUTHOR[0], GIT_COMMIT_AUTHOR[1])
    repo.git.add(all=True)
    repo.index.commit(GIT_COMMIT_MESSAGE + (" " + message_suffix if message_suffix else ""), author=author, committer=author)
    return repo.head.commit.hexsha

def run_tests():
    # run TEST_CMD from settings
    try:
        # Use shell for convenience; be careful in production.
        p = subprocess.run(TEST_CMD, shell=True, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr": "Tests timed out."}
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": str(e)}

def apply_plan(plan_id: str):
    obj = json.loads((PLANS_DIR / f"{plan_id}.json").read_text(encoding="utf-8"))
    apply_patch_text(obj["patch"])
    commit = commit_changes()
    test_res = run_tests()
    return {"applied": True, "commit": commit, "tests": test_res}

def apply_plan_files(plan_id: str, files):
    obj = json.loads((PLANS_DIR / f"{plan_id}.json").read_text(encoding="utf-8"))
    ps = PatchSet(obj["patch"].splitlines(keepends=True))
    out_blocks = []
    for p in ps:
        path = str(p.target_file or p.path).replace("b/","").replace("a/","")
        if path in files:
            # recompose full diff block for this file
            header = f"diff --git a/{path} b/{path}\n"
            header += f"--- a/{path}\n+++ b/{path}\n"
            hunks = []
            for h in p:
                # each h is a Hunk; build text
                h_lines = []
                h_lines.append(str(h.header).rstrip() if hasattr(h, 'header') else '')
                for ln in h:
                    try:
                        h_lines.append(ln.value.rstrip('\n'))
                    except Exception:
                        pass
                hunks.append("\n".join([l for l in h_lines if l]))
            if hunks:
                out_blocks.append(header + "\n".join(hunks))
    if not out_blocks:
        raise RuntimeError("No matching files/hunks found.")
    partial = "\n".join(out_blocks)
    apply_patch_text(partial)
    commit = commit_changes("(partial)")
    test_res = run_tests()
    return {"applied": True, "commit": commit, "tests": test_res, "files": files}

def apply_plan_hunks(plan_id: str, selections):
    # selections: dict { "path": [hunk_index, ...], ... }
    obj = json.loads((PLANS_DIR / f"{plan_id}.json").read_text(encoding="utf-8"))
    parsed = parse_patch_hunks(obj.get("patch",""))
    out_blocks = []
    for file_block in parsed:
        path = file_block.get("file")
        chosen = selections.get(path, [])
        if not chosen:
            continue
        header = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        hunks = []
        for idx in chosen:
            if idx < 0 or idx >= len(file_block["hunks"]):
                continue
            hunks.append(file_block["hunks"][idx])
        if hunks:
            out_blocks.append(header + "\n".join(hunks))
    if not out_blocks:
        raise RuntimeError("No hunks selected.")
    partial = "\n".join(out_blocks)
    apply_patch_text(partial)
    commit = commit_changes("(hunks)")
    test_res = run_tests()
    return {"applied": True, "commit": commit, "tests": test_res, "selections": selections}

def make_plan_files(plan_id: str):
    # keep compatibility
    return make_plan_files(plan_id)
