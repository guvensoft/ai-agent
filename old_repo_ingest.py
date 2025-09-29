from pathlib import Path
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern
from rag_core import get_vectorstore
from settings import REPO_ROOT, INCLUDE_GLOBS, EXCLUDE_GLOBS, RESPECT_GITIGNORE, CHUNK_SIZE, CHUNK_OVERLAP
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from tqdm import tqdm

LANG_MAP = { ".py": Language.PYTHON, ".md": Language.MARKDOWN }

def load_gitignore(root: Path) -> PathSpec:
    gi = root / ".gitignore"
    if not gi.exists():
        return PathSpec.from_lines(GitWildMatchPattern, [])
    return PathSpec.from_lines(GitWildMatchPattern, gi.read_text().splitlines())

def iter_files(root: Path):
    spec = load_gitignore(root) if RESPECT_GITIGNORE else PathSpec.from_lines(GitWildMatchPattern, [])
    include = set()
    for pat in INCLUDE_GLOBS:
        include.update(root.glob(pat))
    ex_paths = set()
    for pat in EXCLUDE_GLOBS:
        ex_paths.update(root.glob(pat))
    for p in sorted(include):
        rel = p.relative_to(root)
        if RESPECT_GITIGNORE and spec.match_file(str(rel).replace("\\","/")):
            continue
        if p.is_file() and p.stat().st_size < 1_000_000:
            yield p

def code_splitter(ext: str):
    lang = LANG_MAP.get(ext.lower())
    if lang:
        return RecursiveCharacterTextSplitter.from_language(language=lang, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def main():
    vs = get_vectorstore(collection_name="repo")
    added = 0
    for fp in tqdm(iter_files(REPO_ROOT), desc="Indexing repo"):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            if not text.strip(): continue
            splitter = code_splitter(fp.suffix)
            chunks = splitter.split_text(text)
            metadatas = []
            ids = []
            for i, ch in enumerate(chunks):
                metadatas.append({"source": str(fp), "chunk": i, "ext": fp.suffix.lower(), "kind": "code"})
                ids.append(f"{fp}:{i}")
            if chunks:
                vs.add_texts(chunks, metadatas=metadatas, ids=ids)
                added += len(chunks)
        except Exception as e:
            print("Err:", e)
    print("Done. Added:", added)

if __name__ == '__main__':
    main()
