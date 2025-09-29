from pathlib import Path
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_store")
MEMORY_DIR = Path("memory")
for _d in (DATA_DIR, CHROMA_DIR, MEMORY_DIR):
    Path(_d).mkdir(exist_ok=True, parents=True)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"#"mixedbread-ai/mxbai-rerank-large-v1"
USE_RERANK = True

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K = 6
MAX_TOKENS_CONTEXT = 3000

LLM_MODEL = "qwen2.5-coder:3b"
AGENT_MAX_STEPS = 6

REPO_ROOT = Path(".").resolve()
RESPECT_GITIGNORE = True
INCLUDE_GLOBS = ["**/*.py","**/*.md"]
EXCLUDE_GLOBS = [
    "node_modules/**",
    "dist/**",
    "build/**",
    ".git/**",
    ".venv/**",
    "**/site-packages/**",
    "**/*.min.*"
    ]

GIT_COMMIT_AUTHOR = ("ai-agent","ai-agent@local")
GIT_COMMIT_MESSAGE = "feat(agent): applied changes by agent"

# Test command to run after applying patches. Adjust to your project's test command if needed.
TEST_CMD = "pytest -q"
