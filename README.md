AI Dev Agent (local) - Hunk-level apply + test-run

New endpoints:
- POST /dev/plan -> create machine-readable plan (JSON only)
- POST /dev/implement -> run implementer and generate unified diff patch
- POST /dev/verify -> execute sandboxed tests & static analysis with auto-fix loop
- POST /dev/plan/hunks -> returns parsed hunks per file for a plan_id
- POST /dev/apply/hunks -> apply selected hunks; runs tests and returns results

UI:
- streamlit run ui.py shows Patch Preview, Hunk selection and applies selected hunks then runs tests.

Security:
- apply endpoints require a clean git working tree (no unstaged changes).
- Test command is configurable via settings.TEST_CMD (default: "pytest -q").

Usage summary:
1) pip install -r requirements.txt
2) ollama pull qwen2.5-coder:3b
3) uvicorn app:app --reload --port 8000
4) streamlit run ui.py
5) Create plan -> run implementer -> verify -> apply selected hunks/files
