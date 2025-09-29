# rag_core.py
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import threading

# Embedding / rerank modeller
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from settings import (
    CHROMA_DIR, EMBEDDING_MODEL, TOP_K, RERANK_MODEL, USE_RERANK, MAX_TOKENS_CONTEXT
)

# --- lazy singletons (thread-safe-ish) ---
_embedder = None
_reranker = None
_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()

def get_lc_embeddings():
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})

def get_embedder():
    global _embedder
    with _embed_lock:
        if _embedder is None:
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
        return _embedder

def get_reranker():
    """Return a CrossEncoder instance if USE_RERANK True; else None."""
    global _reranker
    if not USE_RERANK:
        return None
    with _rerank_lock:
        if _reranker is None:
            _reranker = CrossEncoder(RERANK_MODEL)
        return _reranker

# --- Vectorstore wrapper (Chroma via langchain-chroma) ---
def get_vectorstore(collection_name: str = "docs"):
    embeddings = get_lc_embeddings()
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vs

def get_retriever(filters: Optional[Dict] = None, k: int = TOP_K, collection_name: str = "docs"):
    vs = get_vectorstore(collection_name=collection_name)
    search_kwargs = {"k": k}
    if filters:
        search_kwargs["filter"] = filters
    return vs.as_retriever(search_kwargs=search_kwargs)

# --- small dataclass for returned chunks ---
@dataclass
class RetrievedChunk:
    text: str
    meta: Dict
    score: Optional[float] = None

def _token_estimate(s: str) -> int:
    return max(1, len(s) // 4)

def build_context_block(chunks: List[RetrievedChunk]) -> str:
    acc = []
    total = 0
    for c in chunks:
        src = c.meta.get("source", c.meta.get("path", "?"))
        ch = c.meta.get("chunk", c.meta.get("start_line", "?"))
        header = f"[{src} :: {ch}]\n"
        block = header + c.text.strip() + "\n"
        t = _token_estimate(block)
        if total + t > MAX_TOKENS_CONTEXT:
            break
        acc.append(block)
        total += t
    if not acc:
        return ""
    return "\n--- CONTEXT ---\n" + "\n".join(acc)

def to_retrieved_chunks(docs: List[Document]) -> List[RetrievedChunk]:
    out = []
    for d in docs:
        meta = d.metadata or {}
        # score might be in metadata or not; try several keys
        score = meta.get("_distance") or meta.get("score") or None
        out.append(RetrievedChunk(text=d.page_content, meta=meta, score=score))
    return out

# --- rerank using CrossEncoder (pairs: (query, doc_text)) ---
def rerank_cross_encoder(query: str, docs: List[Document], top_k: int) -> List[Document]:
    """
    Re-rank a list of langchain Documents by relevance to 'query' using a CrossEncoder.
    If no cross-encoder available, returns docs[:top_k].
    """
    rr = get_reranker()
    if rr is None or not docs:
        return docs[:top_k]
    pairs = [(query, d.page_content) for d in docs]
    scores = rr.predict(pairs).tolist()
    order = np.argsort(scores)[::-1]
    ranked = [docs[i] for i in order]
    return ranked[:top_k]

# --- two-stage retrieval: repo + symbols (or any two collections) ---
def two_stage_retrieval(query: str,
                        top_k: int = TOP_K,
                        repo_k: int = None,
                        symbols_k: int = None,
                        repo_collection: str = "repo",
                        symbols_collection: str = "symbols") -> List[RetrievedChunk]:
    """
    Two-stage retrieval:
      1) retrieve repo_k docs from 'repo' collection (chunk-level),
      2) retrieve symbols_k docs from 'symbols' collection (symbol-level),
      3) combine and rerank via cross-encoder, return top_k as RetrievedChunk list.
    Defaults: repo_k = top_k*2, symbols_k = top_k*2 if not provided.
    """
    if repo_k is None:
        repo_k = max(top_k * 2, TOP_K)
    if symbols_k is None:
        symbols_k = max(top_k * 2, TOP_K)

    docs_combined = []

    # 1) repo retrieval
    try:
        retr_repo = get_retriever(k=repo_k, collection_name=repo_collection)
        docs_repo = retr_repo.get_relevant_documents(query)
        docs_combined.extend(docs_repo)
    except Exception as e:
        # fallback: empty
        docs_repo = []

    # 2) symbols retrieval
    try:
        retr_sym = get_retriever(k=symbols_k, collection_name=symbols_collection)
        docs_sym = retr_sym.get_relevant_documents(query)
        docs_combined.extend(docs_sym)
    except Exception as e:
        docs_sym = []

    # 3) dedupe by content (simple)
    seen = set()
    unique_docs = []
    for d in docs_combined:
        key = (d.page_content[:200].strip() if d.page_content else "") + "::" + str(d.metadata)
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(d)

    # 4) rerank and select top_k
    ranked = rerank_cross_encoder(query, unique_docs, top_k=top_k)
    # convert to RetrievedChunk
    chunks = to_retrieved_chunks(ranked)
    return chunks

# --- convenience wrapper for backward compatibility ---
def query_similar(query: str, top_k: int = TOP_K, two_stage: bool = True) -> List[Dict]:
    """
    Returns list of dicts: {"text":..., "meta":..., "score":...}.
    If two_stage True and symbols collection exists, uses two_stage_retrieval, otherwise get_retriever on 'repo'.
    """
    if two_stage:
        docs = two_stage_retrieval(query, top_k=top_k)
    else:
        retr = get_retriever(k=top_k, collection_name="repo")
        docs = to_retrieved_chunks(retr.get_relevant_documents(query))

    out = []
    for c in docs:
        out.append({"text": c.text, "meta": c.meta, "score": c.score})
    return out

# --- Compatibility wrapper for older imports (rerank) ---
def rerank(query: str, docs: List[Document], top_k: int) -> List[Document]:
    """
    Backwards-compatible wrapper used by older code that expects 'rerank'.
    Uses cross-encoder reranker if enabled, otherwise returns first top_k docs.
    """
    try:
        return rerank_cross_encoder(query, docs, top_k)
    except Exception:
        # fallback: simple slice
        return docs[:top_k]
