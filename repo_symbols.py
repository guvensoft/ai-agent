# repo_symbols.py
"""
Symbol index & call-graph extractor.

- Çıktı: memory/symbols.json (map: symbols, call_graph)
- Ayrıca Chroma collection 'symbols' içine sembol kod parçalarını yazar (semantik arama için).
- Python odaklıdır; diğer dosya türleri atlanır.
"""
from pathlib import Path
import json
import ast
from collections import defaultdict
from typing import Dict, List, Any
from tqdm import tqdm

# Project helpers (kendi repo_ingest.py'de tanımlı iter_files/get_lines_for_span vs. kullanılır)
try:
    from repo_ingest import iter_files, get_lines_for_span
except Exception:
    # fallback: very small iter_files
    from settings import REPO_ROOT, INCLUDE_GLOBS, EXCLUDE_GLOBS, RESPECT_GITIGNORE
    from pathspec import PathSpec
    from pathspec.patterns.gitwildmatch import GitWildMatchPattern

    def load_gitignore(root: Path) -> PathSpec:
        gi = root / ".gitignore"
        if not gi.exists():
            return PathSpec.from_lines(GitWildMatchPattern, [])
        return PathSpec.from_lines(GitWildMatchPattern, gi.read_text().splitlines())

    def iter_files(root: Path = Path(".")) -> List[Path]:
        spec = load_gitignore(root) if RESPECT_GITIGNORE else PathSpec.from_lines(GitWildMatchPattern, [])
        include_set = set()
        for pat in INCLUDE_GLOBS:
            include_set.update(root.glob(pat))
        for p in sorted(include_set):
            if RESPECT_GITIGNORE and spec.match_file(str(p.relative_to(root)).replace("\\", "/")):
                continue
            if p.is_file():
                yield p

    def get_lines_for_span(path: Path, start: int, end: int) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            start_idx = max(0, (start or 1) - 1)
            end_idx = min(len(lines), end or len(lines))
            return "\n".join(lines[start_idx:end_idx])
        except Exception:
            return ""

# Chroma / retrieval helper
from rag_core import get_vectorstore

MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
SYMBOLS_JSON = MEMORY_DIR / "symbols.json"

def _extract_symbols_from_py(fp: Path):
    """
    AST parse: returns list of symbol dicts:
    { name, kind, start_line, end_line, calls (list), docstring, code }
    """
    text = fp.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except Exception:
        return []

    symbols = []
    parent_stack = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            qual = ".".join([*parent_stack, node.name]).strip(".")
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            code = get_lines_for_span(fp, start or 1, end or start or 1)
            doc = ast.get_docstring(node) or ""
            calls = self._collect_calls(node)
            symbols.append({
                "name": qual,
                "kind": "class",
                "path": str(fp),
                "start_line": start,
                "end_line": end,
                "calls": calls,
                "docstring": doc,
                "code": code
            })
            parent_stack.append(node.name)
            self.generic_visit(node)
            parent_stack.pop()

        def visit_FunctionDef(self, node):
            qual = ".".join([*parent_stack, node.name]).strip(".")
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            code = get_lines_for_span(fp, start or 1, end or start or 1)
            doc = ast.get_docstring(node) or ""
            calls = self._collect_calls(node)
            symbols.append({
                "name": qual,
                "kind": "function",
                "path": str(fp),
                "start_line": start,
                "end_line": end,
                "calls": calls,
                "docstring": doc,
                "code": code
            })
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def _collect_calls(self, node):
            s = set()
            class CV(ast.NodeVisitor):
                def visit_Call(self, cnode):
                    fn = cnode.func
                    name = None
                    if isinstance(fn, ast.Name):
                        name = fn.id
                    elif isinstance(fn, ast.Attribute):
                        try:
                            name = ast.unparse(fn)
                        except Exception:
                            name = getattr(fn, "attr", None)
                    if name:
                        s.add(name)
                    self.generic_visit(cnode)
            cv = CV()
            cv.visit(node)
            return sorted(s)

    v = Visitor()
    v.visit(tree)
    return symbols

def build_symbol_index(collection_to_write: str = "symbols"):
    """
    Tarama yapar, sembol datasını oluşturur, call_graph çıkarır ve memory/symbols.json'a kaydeder.
    Ayrıca Chroma 'symbols' koleksiyonuna symbol.code metinleri ile upsert yapar.
    """
    symbols_all = []  # list of dicts
    for fp in tqdm(list(iter_files(Path("."))), desc="Symbol extraction"):
        try:
            if fp.suffix.lower() != ".py":
                continue
            syms = _extract_symbols_from_py(fp)
            symbols_all.extend(syms)
        except Exception as e:
            print(f"[symbol extract error] {fp}: {e}")

    # Build lookup: name -> symbol object (last wins)
    name_to_sym: Dict[str, Dict[str, Any]] = {}
    for s in symbols_all:
        # Use path-qualified name to reduce collisions: path::name
        key = f"{s['path']}::{s['name']}"
        name_to_sym[key] = s

    # Build call graph (callees & callers) with best-effort name matching
    callees_map: Dict[str, List[str]] = defaultdict(list)
    callers_map: Dict[str, List[str]] = defaultdict(list)

    # Create small helper to match simple names to candidates
    simple_to_keys = defaultdict(list)
    for k, s in name_to_sym.items():
        simple = s["name"].split(".")[-1]
        simple_to_keys[simple].append(k)

    for src_k, s in name_to_sym.items():
        for cal in s.get("calls", []):
            # try to match cal to known symbol keys by simple name first
            cand_keys = simple_to_keys.get(cal, [])
            if not cand_keys:
                # try exact match on qualified names
                exacts = [k for k in name_to_sym.keys() if k.endswith(f"::{cal}") or k.endswith(f"::{cal}")]
                cand_keys = exacts
            for tgt in cand_keys:
                callees_map[src_k].append(tgt)
                callers_map[tgt].append(src_k)

    # Normalize lists (unique)
    for k in list(callees_map.keys()):
        callees_map[k] = sorted(list(set(callees_map[k])))
    for k in list(callers_map.keys()):
        callers_map[k] = sorted(list(set(callers_map[k])))

    # Compose output
    out = {
        "symbols": name_to_sym,     # mapping key -> symbol object
        "callees": dict(callees_map),
        "callers": dict(callers_map),
        "meta": {"count_symbols": len(name_to_sym)}
    }

    # save JSON
    SYMBOLS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Symbol index written: {SYMBOLS_JSON} (symbols: {len(name_to_sym)})")

    # Also upsert to vectorstore for retrieval (one document per symbol)
    vs = get_vectorstore(collection_name=collection_to_write)
    docs = []
    metas = []
    ids = []
    for k, s in name_to_sym.items():
        text = s.get("docstring") or s.get("code") or ""
        if not text.strip():
            text = s.get("code")[:512] if s.get("code") else ""
        meta = {
            "key": k,
            "name": s.get("name"),
            "path": s.get("path"),
            "kind": s.get("kind")
        }
        docs.append(text)
        metas.append(meta)
        ids.append(k)
    if docs:
        # sanitize metas similar to repo_ingest (avoid complex types)
        safe_metas = []
        for m in metas:
            nm = {}
            for a, b in m.items():
                if b is None or isinstance(b, (str, int, float, bool)):
                    nm[a] = b
                else:
                    try:
                        nm[a] = json.dumps(b, ensure_ascii=False)
                    except Exception:
                        nm[a] = str(b)
            safe_metas.append(nm)
        vs.add_texts(docs, metadatas=safe_metas, ids=ids)

    return out

if __name__ == "__main__":
    idx = build_symbol_index()
    print("Done. Symbols:", idx["meta"]["count_symbols"])
