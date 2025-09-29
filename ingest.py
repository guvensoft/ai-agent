import os
from pathlib import Path
from settings import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_core import get_vectorstore

def load_text(fp: Path) -> str:
    ext = fp.suffix.lower()
    if ext in [".txt", ".md"]:
        return fp.read_text(encoding="utf-8", errors="ignore")
    elif ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(fp))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        from unstructured.partition.auto import partition
        els = partition(filename=str(fp))
        return "\n".join(getattr(e, 'text', '') for e in els if getattr(e, 'text', ''))

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def main():
    vs = get_vectorstore(collection_name="docs")
    files = []
    for root, _, fns in os.walk(DATA_DIR):
        for fn in fns:
            files.append(Path(root)/fn)
    added = 0
    for fp in files:
        try:
            txt = load_text(fp)
            if not txt.strip(): continue
            chunks = chunk_text(txt)
            metadatas = [{"source": str(fp), "chunk": i, "kind": "doc"} for i, _ in enumerate(chunks)]
            ids = [f"{fp}:{i}" for i,_ in enumerate(chunks)]
            if chunks:
                vs.add_texts(chunks, metadatas=metadatas, ids=ids)
                added += len(chunks)
        except Exception as e:
            print("Error:", e)
    print("Done. Added:", added)

if __name__ == '__main__':
    main()
