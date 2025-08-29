# src/web/main.py
#  $ uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8080
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

import firebase_admin
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from firebase_admin import credentials, firestore
from pydantic import BaseModel

from .deps import get_settings
from .init_model import load_ckip_model
from .routers import health, verifier, answerer, rag_mode

# ── 確保本地目錄存在，避免檔案操作錯誤 ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent  # …/FactGraph/src/web
# 應該往上兩層才是專案根目錄 …/home/karca5103/dev/FactGraph
PROJECT_ROOT = BASE_DIR.parent.parent
for mode in ("verifier", "answerer", "rag_mode"):
    (PROJECT_ROOT / "data" / "interim" / mode /
     "user-input").mkdir(parents=True, exist_ok=True)

# ── Firebase Key 路徑 & CORS 設定 ─────────────────────────────────────────────────
KEY_PATH = BASE_DIR / "key" / "factgraph-38be7-firebase-adminsdk-fbsvc-20b7fbb9a4.json"
origins = [
    "https://factgraph-38be7.web.app",
    "http://localhost:8080",
    "http://localhost:5173",
]

# ── FastAPI App 初始化 ─────────────────────────────────────────────────────────
app = FastAPI(title="FactGraph API", version="0.1.0", docs_url="/")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router, prefix="/api")
app.include_router(verifier.router, prefix="/api")
app.include_router(answerer.router, prefix="/api")
app.include_router(rag_mode.router, prefix="/api")

# ── Firebase Admin SDK 初始化 (Admin SDK 不受安全規則限制) ─────────────────────────────
cred = credentials.Certificate(str(KEY_PATH))
firebase_admin.initialize_app(cred)
db = firestore.client()

# 用於內部同步呼叫
sync_client = TestClient(app)


# ── Pydantic 定義 ────────────────────────────────────────────────────────────
class JobCreate(BaseModel):
    url: str
    mode: Literal["writing", "question", "rag"]
    date: str  # YYYY/MM/DD


class JobOut(BaseModel):
    id: str
    status: str


# ── 背景任務：呼叫同步 endpoint，並將答案與知識寫入 Firestore ─────────────────────────
def process_task(job_id: str, url: str, mode: str, date: str):
    doc_ref = db.collection("url-results").document(job_id)
    doc_ref.update({"status": "RUNNING"})
    try:
        files = {"file": ("user.txt", url, "text/plain")}
        if mode == "writing":
            # 保持原行為
            resp = sync_client.post(
                "/api/verifier/query", files=files, data={"date": date})
            body = resp.json()
            wa, wk = body.get("judge_result", ""), body.get("news_kg", "")
            update_fields = {
                "writingAnswer": wa,
                "writingKnowledge": wk,
                "questionAnswer": None,
                "questionKnowledge": None,
                "ragAnswer": None,
                "last_error": None,
            }

        elif mode == "question":
            # 保持原行為
            resp = sync_client.post(
                "/api/answerer/query", files=files, data={"date": date})
            body = resp.json()
            qa = body.get("user_judge_result") or body.get("result") or ""
            qk = body.get("user_news_kg") or body.get("news_kg") or ""
            update_fields = {
                "questionAnswer": qa,
                "questionKnowledge": qk,
                "writingAnswer": None,
                "writingKnowledge": None,
                "ragAnswer": None,
                "last_error": None,
            }

        else:  # mode == "rag"
            # ── 這裡是唯一改動重點 ────────────────────────────────────────
            # 呼叫同步 RAG 端點（rag_mode.py 會在產出為空/錯誤時回 500）:contentReference[oaicite:2]{index=2}
            resp = sync_client.post("/api/rag_mode/query", files=files)

            # 非 200 → 標記 FAILED + last_error
            if resp.status_code != 200:
                err_msg = ""
                try:
                    j = resp.json()
                    err_msg = j.get("detail") or str(j)
                except Exception:
                    err_msg = (resp.text or "").strip()
                doc_ref.update({
                    "ragAnswer": None,
                    "writingAnswer": None,
                    "writingKnowledge": None,
                    "questionAnswer": None,
                    "questionKnowledge": None,
                    "last_error": f"RAG endpoint {resp.status_code}: {err_msg}"[:1500],
                })
                doc_ref.update({"status": "FAILED"})
                return

            # 解析成功結果
            body = resp.json()
            ra = (body.get("result_md") or "").strip()

            # 回 200 但內容為空 → 視為錯誤（避免 DONE + 空字串）
            if not ra:
                doc_ref.update({
                    "ragAnswer": None,
                    "writingAnswer": None,
                    "writingKnowledge": None,
                    "questionAnswer": None,
                    "questionKnowledge": None,
                    "last_error": "RAG 產出為空，請檢查向量庫或 annex",
                })
                doc_ref.update({"status": "FAILED"})
                return

            # 成功：寫入 ragAnswer，清空 last_error
            update_fields = {
                "ragAnswer": ra,
                "writingAnswer": None,
                "writingKnowledge": None,
                "questionAnswer": None,
                "questionKnowledge": None,
                "last_error": None,
            }

        # 統一寫入並標記完成
        doc_ref.update(update_fields)
        doc_ref.update({"status": "DONE"})

    except Exception as e:
        print(f"[process_task] EXCEPTION job_id={job_id}: {e}")
        # 捕捉到任何未處理例外 → 標記 FAILED
        doc_ref.update(
            {"status": "FAILED", "last_error": f"{type(e).__name__}: {e}"})


# ── 建立任務 Endpoint：POST /api/tasks ───────────────────────────────────────────
@app.post("/api/tasks", response_model=JobOut)
def create_task(payload: JobCreate, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    # 多加 ragAnswer 與 last_error 的預設值，不影響其他模式（僅欄位預置）
    db.collection("url-results").document(job_id).set({
        "status": "PENDING",
        "url": payload.url,
        "mode": payload.mode,
        "date": payload.date,
        "writingAnswer": None,
        "writingKnowledge": None,
        "questionAnswer": None,
        "questionKnowledge": None,
        "ragAnswer": None,
        "last_error": None,
        "created_at": firestore.SERVER_TIMESTAMP,
    })
    background_tasks.add_task(process_task, job_id,
                              payload.url, payload.mode, payload.date)
    return JobOut(id=job_id, status="PENDING")


# ── 查詢任務狀態 Endpoint：GET /api/tasks/{job_id} ───────────────────────────────────
@app.get("/api/tasks/{job_id}", response_model=JobOut)
def get_task(job_id: str):
    doc = db.collection("url-results").document(job_id).get()
    if not doc.exists:
        raise HTTPException(404, "Job not found")
    data = doc.to_dict()
    return JobOut(id=job_id, status=data.get("status"))


# ── 啟動時 Pre-load CKIP 模型 ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("📦 預載 CKIP 模型…")
    ckip_model = load_ckip_model()
    app.state.ckip_model = ckip_model
    app.state.model_loaded = True
    print("📦 模型載入完成。")
