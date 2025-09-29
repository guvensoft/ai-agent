# repo_ingest.py
"""
Repo indeksleyici - AST tabanlı chunking (Python için fonksiyon/class/method düzeyi).
Diğer uzantılar için fallback: paragraf parçalama.
Chroma collection 'repo' içine ekler ve metadata'ları güvenli hale getirir.
"""
from pathlib import Path
import ast
import os
import json
from typing import Iterable, List, Dict, Any, Optional
from settings import REPO_ROOT, INCLUDE_GLOBS, EXCLUDE_GLOBS, RESPECT_GITIGNORE, CHUNK_SIZE, CHUNK_OVERLAP
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern
from rag_core import get_vectorstore
from tqdm import tqdm

# try langchain-community helper
try:
    from langchain_community.vectorstores.utils import filter_complex_metadata
    _HAS_FILTER_UTIL = True
except Exception:
    filter_complex_metadata = None
    _HAS_FILTER_UTIL = False

# Safety: don't index virtualenv / site-packages
_ADDITIONAL_EXCLUDES = ["\\.venv", "venv", "site-packages", ".venv", "__pycache__"]

# Maximum file size to index (1 MB)
MAX_FILE_SIZE = 1_000_000

def load_gitignore(root: Path) -> PathSpec:
    gi = root / ".gitignore"
    if not gi.exists():
        return PathSpec.from_lines(GitWildMatchPattern, [])
    return PathSpec.from_lines(GitWildMatchPattern, gi.read_text().splitlines())

def iter_files(root: Path) -> Iterable[Path]:
    """
    Iterate files matching INCLUDE_GLOBS, excluding EXCLUDE_GLOBS, .gitignore matches,
    and additional excludes like virtualenv or site-packages.
    """
    spec = load_gitignore(root) if RESPECT_GITIGNORE else PathSpec.from_lines(GitWildMatchPattern, [])
    include_set = set()
    for pat in INCLUDE_GLOBS:
        include_set.update(root.glob(pat))
    ex_paths = set()
    for pat in EXCLUDE_GLOBS:
        ex_paths.update(root.glob(pat))
    for p in sorted(include_set):
        try:
            rel = p.relative_to(root)
        except Exception:
            rel = p
        rel_str = str(rel).replace("\\", "/")
        # skip if matches gitignore
        if RESPECT_GITIGNORE and spec.match_file(rel_str):
            continue
        # skip explicit exclude globs
        if any(str(p).replace("\\", "/").startswith(str(x).replace("\\", "/")) for x in ex_paths):
            continue
        # skip venv / site-packages heuristics
        low = str(p).lower()
        if any(ex in low for ex in _ADDITIONAL_EXCLUDES):
            continue
        if p.is_file() and p.stat().st_size < MAX_FILE_SIZE:
            yield p

# --- Helpers to extract code spans from file by lineno ---
def get_lines_for_span(path: Path, start: int, end: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        start_idx = max(0, (start or 1) - 1)
        end_idx = min(len(lines), end or len(lines))
        return "\n".join(lines[start_idx:end_idx])
    except Exception:
        return ""

# --- AST visitor to collect functions/classes & calls ---
class SymbolCollector(ast.NodeVisitor):
    def __init__(self):
        self.symbols = []  # list of dicts
        self._parent_stack = []

    def visit_Module(self, node: ast.Module):
        self._parent_stack.append("<module>")
        self.generic_visit(node)
        self._parent_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None) or start
        fullname = ".".join([*self._parent_stack, node.name]).replace("<module>.", "")
        sym = {
            "name": fullname,
            "kind": "class",
            "start_lineno": start,
            "end_lineno": end,
            "calls": self._collect_calls(node),
        }
        self.symbols.append(sym)
        self._parent_stack.append(node.name)
        self.generic_visit(node)
        self._parent_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._add_function(node, kind="function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._add_function(node, kind="async_function")
        self.generic_visit(node)

    def _add_function(self, node, kind="function"):
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None) or start
        # build qualified name from parent stack if possible
        qual = ".".join([*self._parent_stack, node.name]).replace("<module>.", "")
        sym = {
            "name": qual,
            "kind": kind,
            "start_lineno": start,
            "end_lineno": end,
            "decorators": [ast.unparse(d) if hasattr(ast, "unparse") else "" for d in getattr(node, "decorator_list", [])],
            "calls": self._collect_calls(node),
        }
        self.symbols.append(sym)

    def _collect_calls(self, node):
        calls = set()
        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, call_node):
                fn = call_node.func
                name = None
                if isinstance(fn, ast.Name):
                    name = fn.id
                elif isinstance(fn, ast.Attribute):
                    try:
                        name = ast.unparse(fn)
                    except Exception:
                        name = getattr(fn, "attr", None)
                if name:
                    calls.add(name)
                self.generic_visit(call_node)
        cv = CallVisitor()
        cv.visit(node)
        return sorted(calls)

# --- metadata sanitizer ---
def _sanitize_metadatas(metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use filter_complex_metadata if available; otherwise JSON-serialize complex values.
    Ensures each metadata value is primitive or a JSON string.
    """
    if _HAS_FILTER_UTIL and filter_complex_metadata is not None:
        try:
            return filter_complex_metadata(metadatas)
        except Exception:
            # fallback to manual sanitation
            pass

    sanitized = []
    for m in metadatas:
        nm: Dict[str, Any] = {}
        for k, v in (m or {}).items():
            if v is None or isinstance(v, (str, int, float, bool)):
                nm[k] = v
            else:
                # convert lists/dicts/other to compact JSON string
                try:
                    nm[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    nm[k] = str(v)
        sanitized.append(nm)
    return sanitized

# --- indexing functions ---
def index_python_file(fp: Path, vs) -> int:
    """
    Parse python file via AST, extract functions/classes as chunks with metadata,
    and upsert into vectorstore.
    """
    text = fp.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except Exception as e:
        # fallback: whole-file as single chunk
        meta = {"source": str(fp), "symbol": None, "kind": "file", "filename": fp.name}
        metas = _sanitize_metadatas([meta])
        vs.add_texts([text], metadatas=metas, ids=[f"{fp}:file"])
        return 1

    collector = SymbolCollector()
    collector.visit(tree)

    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    added = 0

    for sym in collector.symbols:
        start = sym.get("start_lineno") or 1
        end = sym.get("end_lineno") or start
        code = get_lines_for_span(fp, start, end)
        if not code.strip():
            continue
        meta = {
            "source": str(fp),
            "symbol": sym.get("name"),
            "kind": sym.get("kind"),
            "start_line": start,
            "end_line": end,
            "calls": sym.get("calls", []),
            "filename": fp.name,
        }
        chunks.append(code)
        metadatas.append(meta)
        ids.append(f"{fp}:{meta['symbol']}:{start}-{end}")
        added += 1

    # if no symbols found (eg. scripts), fallback to paragraph splitting
    if not chunks:
        lines = text.splitlines()
        window = 200
        step = 120
        idx = 0
        while idx < len(lines):
            seg = "\n".join(lines[idx: idx + window])
            meta = {"source": str(fp), "symbol": None, "kind": "file_chunk", "start_line": idx + 1, "end_line": min(len(lines), idx + window), "filename": fp.name}
            chunks.append(seg)
            metadatas.append(meta)
            ids.append(f"{fp}:chunk:{idx}")
            idx += step
        added += len(chunks)

    if chunks:
        safe_metas = _sanitize_metadatas(metadatas)
        vs.add_texts(chunks, metadatas=safe_metas, ids=ids)
    return added

def index_other_file(fp: Path, vs) -> int:
    """
    Fallback indexing for non-Python files: split by paragraphs and add.
    """
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return 0
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        metas = []
        ids = []
        for i, p in enumerate(paras):
            if len(p) < 30:
                continue
            meta = {"source": str(fp), "symbol": None, "kind": "other", "chunk": i, "filename": fp.name}
            chunks.append(p)
            metas.append(meta)
            ids.append(f"{fp}:p{i}")
        if chunks:
            safe_metas = _sanitize_metadatas(metas)
            vs.add_texts(chunks, metadatas=safe_metas, ids=ids)
            return len(chunks)
    except Exception as e:
        print(f"[Other index error] {fp}: {e}")
    return 0

def main():
    vs = get_vectorstore(collection_name="repo")
    total = 0
    for fp in tqdm(iter_files(REPO_ROOT), desc="Repo indexing (AST chunks)"):
        try:
            if fp.suffix.lower() == ".py":
                added = index_python_file(fp, vs)
            else:
                added = index_other_file(fp, vs)
            total += added
        except Exception as e:
            print(f"[Index error] {fp}: {e}")
    print(f"Indexing done. Added items: {total}")

if __name__ == "__main__":
    main()
