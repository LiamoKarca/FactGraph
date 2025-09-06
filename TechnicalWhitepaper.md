# FactGraph／「芒狗偵探」技術白皮書

角色與宗旨：以 LLM × 知識圖譜 (Neo4j) × 向量檢索 為核心的新聞查核系統。系統覆蓋資料蒐集、ETL、KG 建模、RAG 管線、前後端、雲端部署、觀測與成本控管；回答僅能引用檢索證據並以 Job ID 追溯產物鏈。

---

## 目錄
1. 設計目標與原則  
2. 整體架構總覽（Logical View）  
3. 高階資料流與元件  
4. 前端（Vue 3 + Vite + Firebase Hosting）  
5. 後端（FastAPI on Cloud Run）  
6. RAG／Verifier／Answerer 管線  
7. 資料層：爬蟲／合併／向量庫  
8. 部署（GCP：Docker、Artifact Registry、Cloud Run、Firebase）  
9. 安全與權限  
10. 觀測性、測試與品質  
11. 常見錯誤與快速修復  
12. 效能與成本盤點  
13. 版本治理與紀錄  
14. 開發者速查（實務守則）  
15. 附錄：API 介面、環境變數、資料夾結構、Cloud Run 參數建議、Firestore 訂閱指引  

---

## 1. 設計目標與原則
* **目標**：在臺灣新聞語境下，將「事實（來源）→ 表述（三元組）→ 證據檢索（KG/向量）→ 判定（一致／矛盾／缺漏）」全鏈路自動化，並提供可回溯的憑證。  
* **核心原則**  
  - Evidence only：回答必須引用 RAG 片段或 KG，禁止臆測。  
  - 可重現性：原始新聞 JSON、向量、三元組與回答均以 Job ID 管理。  
  - 低耦合：爬蟲/ETL、KG、RAG、判定器、前端、部署彼此獨立。  
  - 雲原生：容器化 + Cloud Run；前端 Firebase Hosting；狀態追蹤 Firestore。

---

## 2. 整體架構總覽（Logical View）
### 前端 (Vue 3 + Vite + Firebase Hosting)
* **主要頁面**：  
  - `components/ServicePage.vue`（查詢入口）  
  - `views/TaskStatus.vue`（任務狀態）  
  - `pages/ServiceIntro.vue`、`pages/AboutUs.vue`  
  路由在 `frontend/src/router/index.js` 統一管理。

* **狀態來源**：  
  - Firestore 即時訂閱（優先）或輪詢 `watchRagJob.js`  
  - REST API（FastAPI）

### 後端 (FastAPI on Cloud Run)
* `src/web/main.py` 初始化 FastAPI、CORS、Firebase Admin，並掛載 `health`、`verifier`、`answerer`、`rag_mode` 路由。  
* `process_task()` 以背景任務方式呼叫同步 endpoint，並將結果寫回 Firestore。

### 資料層
* `data/raw/news`：新聞爬蟲輸出。  
* `data/processed/news_merge`：合併與去重結果。  
* `data/interim/<mode>/jobs/{uuid}`：各 Job 的工作空間（以 mode 區分）。

---

## 3. 高階資料流與元件
1. **News Crawlers** → `data/raw/news/{source}/*.json`  
2. **ETL**：`news_merge/merge.py` + `dedup.py` → `data/processed/news_merge/*_DEDUP.json`  
3. **向量庫更新**：`news_merge/upload_storage.py` 依 5MB 分片上傳 OpenAI Vector Store。  
4. **Triple Extractor**：LLM 抽取三元組 → Neo4j KG。  
5. **Back-end API**：`/api/rag_mode`, `/api/verifier`, `/api/answerer`, `/api/health`。  
6. **Front-end**：ServicePage 提交任務，TaskStatus 監看結果；靜態 PDF 等資產在 `frontend/src/assets`。

---

## 4. 前端（Vue 3 + Vite + Firebase Hosting）
1. **路由與頁面**：使用 `createWebHistory`；路徑如 `/`, `/service`, `/about`, `/tasks/:id`。  
2. **即時監聽**：`watchRagJob.js` 嘗試 onSnapshot；失敗或 5 秒無首筆資料即降級輪詢。  
3. **API 客戶端**：集中封裝 `/api/*`；送出任務後立即顯示 Pending 卡片並以 Job ID 追蹤。  
4. **PDF 下載**：`AI-Fact-Graph-Plan.pdf` 可透過 `<a download>` 或 `<embed>` 呈現。  
5. **卸載清理**：所有 Firestore 監聽在 `return()` 時關閉，避免資源外洩。

---

## 5. 後端（FastAPI on Cloud Run）
1. **應用結構**：  
   - `web/main.py` 為入口；路由位於 `web/routers/*`。  
   - 管線實作於 `qa/<mode>/pipeline.py`。  
2. **RAG Mode 路由** (`rag_mode.py`)：  
   - `POST /api/rag_mode/query?job_id=`：建立 job 專屬工作目錄、寫入輸入、以環境變數傳遞參數並呼叫 pipeline，回傳 Markdown 結果。  
   - `POST /api/rag_mode/cleanup`：清除 job 原始輸入，減少磁碟堆積。  
3. **Access Log Middleware**：`access_log_middleware()` 依照 Uvicorn `AccessFormatter` 規格輸出，避免格式錯誤。  
4. **背景任務**：`process_task()` 會同步呼叫 Verifier/Answerer/RAG endpoint，並將結果與錯誤寫入 Firestore。  
5. **路徑預建**：啟動時建立 `data/interim/<mode>/user-input` 以避免檔案操作失敗。

---

## 6. RAG／Verifier／Answerer 管線
### 6.1 RAG
* `src/qa/rag_mode/pipeline.py` 為入口，依序執行 `keywords.py` → `annex_kw.py` → `gpt_rag.py`。  
* `_precheck_vs_before_gpt()` 會列出線上 Vector Store，選用最新一個並輪詢其 `file_count`，未就緒即跳過 gpt 步驟並提示稍後重試。  
* `_validate_preconditions()` 確保 `user-input` 目錄中僅有單一 `.txt`，若設置 `RAG_USER_FILE` 則改用 job 專屬輸入。  
* 每個 Job 的資料存放於 `data/interim/rag_mode/jobs/{job_id}/`：  
  - `user-input/`：原始提問或 URL。  
  - `annex/`：LLM 擴寫輸入。  
  - `retrieval/`：向量檢索結果。  
  - `synthesis/`：初稿。  
  - `final/`：最終 Markdown。

### 6.2 Verifier
* `core/paths.py` 以 `_find_repo_root()` 自動尋找專案根目錄並建立 CKIP 模型與資料路徑常數。  
* `llm/extract.py` 使用 `client.chat.completions.create` 並透過 `stream=True` 逐字輸出；若遇 `OpenAIError` 或 `APITimeoutError` 以指數退避重試。  
* 最終判定由 `llm/judge.py` 根據 Neo4j 檢索結果產生一致／矛盾／缺漏報告。

### 6.3 Answerer
* 與 Verifier 相似，會對問句進行 KG + 向量檢索，返回帶來源的簡短答案。  
* `core/utils.py` 提供日期正規化與分詞工具。

---

## 7. 資料層：爬蟲／合併／向量庫
1. **爬蟲**：`knowledge_base_operation/news_crawler/pipeline.py` 產出 `data/raw/news/{src}/*.json`。  
2. **合併與去重**：`news_merge/merge.py` 合併多來源後，`dedup.py` 依標題與內容相似度去重並移除過短文章。  
3. **向量庫更新**：`news_merge/upload_storage.py` 切片上傳並於成功後更新向量庫 ID；失敗時會保留未上傳片段以利重試。  
4. **KG 建模**：`knowledge_graph/pipeline.py` 將 `triples.jsonl` 轉為 Cypher 指令寫入 Neo4j。  
5. **路徑慣例**：所有腳本透過 `Path(__file__).resolve().parents[...]` 取得專案根目錄，避免依賴 `cwd`。

---

## 8. 部署（GCP：Docker、Artifact Registry、Cloud Run、Firebase）
1. **容器**：多階段 Dockerfile；`requirements_base.txt` 與 `requirements_app.txt` 分層安裝以觸發 build cache。  
2. **Artifact Registry**：以不可變 tag (`YYYYMMDD-HHMM`) 儲存映像並配合 Registry Cache 加速。  
3. **Cloud Run**：  
   - `concurrency`：RAG/Verifier 建議 1–4；Answerer 可 20+。  
   - `cpu_always_on` 視是否需要長時間背景作業決定。  
   - 常用環境變數：`OPENAI_API_KEY`, `DATA_ROOT=/workspace/data`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`。  
4. **Firebase Hosting**：前端由 GitHub Actions 自動部署；如需 API proxy，可在 `firebase.json` 設定 `rewrites` 指向 Cloud Run URL。

---

## 9. 安全與權限
* Cloud Run Service Account 僅授予 `Artifact Registry Reader`、`Secret Accessor`、`Logging Writer`。  
* 所有輸入在進入 LLM 前先正規化並檢查可疑字串以防 Prompt 注入。  
* 對昂貴步驟採用 token bucket 速率限制，並加入重試與退避。  
* API key 與 DB 密碼使用 Secret Manager，不寫入映像。

---

## 10. 觀測性、測試與品質
* Logging：`access_log_middleware` 產生的 Access Log 包含 client IP、HTTP 方法、路徑、版本與狀態碼。  
* Metrics：檢索命中率、引用完整度、Verifier ROC/PR。  
* Tracing：規劃導入 OpenTelemetry 以 job_id 串起管線。  
* 單元測試：Annex、檢索、Rerank、引用格式使用固定 fixture。  
* 整合測試：以小型假索引端到端檢查。  
* 災難演練：模擬 Vector Store 延遲、Neo4j 中斷、OpenAI 429。

---

## 11. 常見錯誤與快速修復
1. **ModuleNotFoundError: src**：以 `-m src.qa.rag_mode.pipeline` 啟動或設定 `PYTHONPATH`。  
2. **data/ vs src/data/ 路徑錯置**：統一透過 `PROJECT_ROOT` + 相對路徑定位。  
3. **Firestore onSnapshot primary lease**：建立新監聽前先 `unsubscribe()` 舊監聽，必要時降級輪詢。  
4. **Uvicorn Access Log ValueError**：使用內建 `AccessFormatter`。  
5. **向量庫索引未就緒**：`_precheck_vs_before_gpt()` 檢查 `file_count`，若為 0 則跳過 gpt 步驟。  
6. **Cloud Run 並發競爭**：降低 `concurrency` 或於程式內加 semaphore；重度背景工作可改用 Cloud Run Jobs。

---

## 12. 效能與成本盤點
* OpenAI：控制 context 長度與 Top k；Rerank 設門檻。  
* Firestore：讀多寫少；前端緩取結果與時間窗切分。  
* Cloud Run：CPU 秒與冷啟動是主要成本；流量穩定時可設 `min_instances=1`。  
* 向量計算：OpenAI Vector Store 為主；自管索引用於 re-rank 或備援。  
* 收費試探：每問 NT$2–10 估算成本回收。

---

## 13. 版本治理與紀錄
* 分支：`main`（後端）與 `frontend` 各自維護；`feature/*` 短期開發。  
* Commit 使用 Conventional Commits。  
* 發佈時由 GitHub Actions 自動生成 CHANGELOG。  
* Prompt 置於 `src/prompts` 版本化；A/B 測試以 `chore(prompt):` 記錄。

---

## 14. 開發者速查（實務守則）
* 以 repo 根為相對路徑：`ROOT = Path(__file__).resolve().parents[2]`。  
* Job ID 隔離中間產物；所有 pipeline 必須在自己的 job 目錄內運作。  
* RAG 若索引未就緒不得產生答案。  
* 引用必含來源（URL + 日期 + 摘要）。  
* Firestore 監聽先退舊再綁新；頁面卸載須 `unsubscribe()`。  
* Cloud Run 同一執行個體可並行處理多請求；`concurrency` 需自行控制。

---

## 15. 附錄
### A. API 介面（節選）
* **POST** `/api/rag_mode/query?job_id=...`  
  - Request：multipart `file`（文本）、`job_id` query。  
  - Response：`{"result_md": string}`  
* **POST** `/api/rag_mode/cleanup?job_id=...`  
  - Response：`{"job_id": string, "removed_files": number}`  
* **POST** `/api/verifier/query`、`/api/answerer/query` 類似。

### B. 建議環境變數
`OPENAI_API_KEY`, `DATA_ROOT=/workspace/data`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`, `FIREBASE_PROJECT` 等。

### C. 資料夾結構（節選）
```
repo/
  data/
    raw/news/{cna,pts,moi,ey}/...
    processed/news_merge/*_DEDUP.json
    interim/rag_mode/jobs/{uuid}/(user-input|annex|retrieval|synthesis|final)
    interim/verifier/user-input/...
  src/
    web/main.py
    web/routers/{health,verifier,answerer,rag_mode}.py
    qa/{rag_mode,verifier,answerer}/pipeline.py
    qa/verifier/llm/{extract.py,judge.py}
    qa/verifier/core/{paths.py,config.py,dedup.py}
    knowledge_base_operation/{news_crawler,news_merge,knowledge_graph}/...
  frontend/src/
    components/ServicePage.vue
    views/TaskStatus.vue
    pages/{ServiceIntro,AboutUs}.vue
    rag/watchRagJob.js
```

### D. Cloud Run 參數建議
`concurrency=1~4`（RAG/Verifier）、`memory=1~2GiB`、`timeout=300s`、`min_instances=0~1`、`cpu_always_on` 視需求。

### E. Firestore 訂閱指引
使用 `watchRagJob.js`：  
1. 建立監聽；5 秒內無資料或監聽失敗即降級輪詢。  
2. 監聽完成後或頁面卸載時呼叫回傳的清理函數。

---