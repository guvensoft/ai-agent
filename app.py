import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import ollama
from rag_core import get_retriever, to_retrieved_chunks, build_context_block, rerank
from settings import LLM_MODEL, TOP_K
from agent.agent_main import build_agent
from repo_ingest import main as ingest_repo
from dev_workflow import (
    make_plan,
    make_plan_files,
    apply_plan,
    apply_plan_files,
    revert_commit,
    get_plan_hunks,
    apply_plan_hunks,
    implement_plan,
    verify_plan,
)
from orchestrator import sandbox_test_plan, create_plan, get_plan
from implementer import PatchValidationError

app = FastAPI(title="Repo Doc Chat + Dev Agent", version="0.5.0")
_agent = None


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[List[str]]] = None
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = TOP_K

@app.post("/repo/ingest")
def repo_ingest():
    ingest_repo()
    return {"status":"ok"}

class PlanReq(BaseModel):
    request: str


@app.post("/dev/plan")
def dev_plan(req: PlanReq):
    res = make_plan(req.request)
    return JSONResponse(res)


class ImplementReq(BaseModel):
    plan_id: str
    feedback: Optional[str] = None


@app.post("/dev/implement")
def dev_implement(req: ImplementReq):
    try:
        res = implement_plan(req.plan_id, feedback=req.feedback)
        return JSONResponse(res)
    except PatchValidationError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


class SandboxReq(BaseModel):
    plan_id: str

@app.post("/dev/sandbox_test")
def dev_sandbox_test(req: SandboxReq):
    try:
        res = sandbox_test_plan(req.plan_id)
        return JSONResponse(res)
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=400)

class PlanPreviewReq(BaseModel):
    request: str

@app.post("/dev/plan/preview")
def dev_plan_preview(req: PlanPreviewReq):
    res = create_plan(req.request)
    # dönen res içinde plan_id, plan, patch_preview, patch var
    # biz sadece preview dönebiliriz:
    return JSONResponse({"plan_id": res["plan_id"], "plan": res["plan"], "patch_preview": res["patch_preview"]})


class FilesReq(BaseModel):
    plan_id: str

@app.post("/dev/plan/files")
def dev_plan_files(req: FilesReq):
    res = make_plan_files(req.plan_id)
    return JSONResponse(res)

class HunkReq(BaseModel):
    plan_id: str

@app.post("/dev/plan/hunks")
def dev_plan_hunks(req: HunkReq):
    res = get_plan_hunks(req.plan_id)
    return JSONResponse(res)

class ApplyHunksReq(BaseModel):
    plan_id: str
    selections: Dict[str, List[int]]

@app.post("/dev/apply/hunks")
def dev_apply_hunks(req: ApplyHunksReq):
    res = apply_plan_hunks(req.plan_id, req.selections)
    return JSONResponse(res)

class ApplyReq(BaseModel):
    plan_id: str

@app.post("/dev/apply")
def dev_apply(req: ApplyReq):
    res = apply_plan(req.plan_id)
    return JSONResponse(res)


class VerifyReq(BaseModel):
    plan_id: str
    auto_fix: Optional[bool] = True
    max_rounds: Optional[int] = 3


@app.post("/dev/verify")
def dev_verify(req: VerifyReq):
    res = verify_plan(req.plan_id, auto_fix=req.auto_fix, max_rounds=req.max_rounds)
    return JSONResponse(res)

class ApplyFilesReq(BaseModel):
    plan_id: str
    files: List[str]

@app.post("/dev/apply/files")
def dev_apply_files(req: ApplyFilesReq):
    res = apply_plan_files(req.plan_id, req.files)
    return JSONResponse(res)

class RevertReq(BaseModel):
    commit: str

@app.post("/dev/revert")
def dev_revert(req: RevertReq):
    res = revert_commit(req.commit)
    return JSONResponse(res)

class AgentRequest(BaseModel):
    input: str

@app.post("/agent")
def run_agent(req: AgentRequest):
    global _agent
    if _agent is None:
        _agent = build_agent()
    result = _agent.invoke({"input": req.input})
    out = result.get("output", "") or result.get("answer","")
    return JSONResponse({"answer": out, "model": LLM_MODEL})
