# test_retrieval.py
from rag_core import two_stage_retrieval, query_similar
q = "Where is the health check endpoint defined?"
chunks = two_stage_retrieval(q, top_k=6)
for i, c in enumerate(chunks):
    print("----", i, "----")
    print("META:", c.meta)
    print(c.text[:400])
    print()
