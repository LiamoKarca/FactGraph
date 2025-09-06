# FactGraph

**FactGraph** 是一套以 **知識圖譜（Knowledge Graph）** 為基礎的事實查證與問答平台，整合 **LLM、向量檢索** 與 **RAG（Retrieval-Augmented Generation）** 技術，協助使用者快速驗證資訊並生成可追溯的證據。

---

## 目錄

* [功能與特色](#功能與特色)
* [技術架構](#技術架構)
* [環境需求](#環境需求)
* [快速開始](#快速開始)
* [資料夾結構](#資料夾結構)
* [API 概覽](#api-概覽)
* [Docker 部署](#docker-部署)
* [團隊與夥伴](#團隊與夥伴)
* [貢獻指南](#貢獻指南)
* [授權](#授權)

---

## 功能與特色

* **Verifier：文字事實查證**
  對新聞或文章抽取三元組後，比對知識圖譜並生成判斷結果及評估答案可信度。

* **Answerer：問題回答**
  解析使用者提問、抽取三元組後，從知識圖譜中搜尋相關敘述，將檢索結果生成答覆。

* **RAG Mode**
  經過長、短篇分類任務後，以關鍵字擴充原始文本、檢索向量庫後，整合給出完整答覆，並予以 LLM 自信程度進行自我評估。

* **後端 API 與前端分離**
  後端以 **FastAPI** 實作；前端採用 **Vue 3 + Vite**，並整合 **Firebase** 進行登入與資料儲存。

* **多階段資料流程**
  `data/` 目錄區分 `raw`、`interim`、`processed`，明確管理輸入與產出。

* **背景作業與 Firebase**
  後端支援背景任務，將查證結果回寫至 **Firestore**，方便非同步操作。

---

## 技術架構

```
Backend (FastAPI)                              Frontend (Vue 3 + Vite)
 ├─ /api/verifier/query                         ├─ 使用者介面
 ├─ /api/answerer/query                         └─ Firebase 認證與儲存
 └─ /api/rag_mode/query
            │
            └─ QA Pipeline 模組 (src/qa/)
               ├─ verifier/
               ├─ answerer/
               └─ rag_mode/
```
> 更多技術說明詳見 **TechnicalWhitepaper.md**

---

## 環境需求

* Python **3.12+**
* Node.js **20+**
* Firebase 服務帳戶 JSON（放在 `src/web/key/`）

---

## 快速開始

1. **取得程式碼並建立虛擬環境**

   ```bash
   git clone <repository-url> FactGraph
   cd FactGraph
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

2. **安裝後端相依套件**

   ```bash
   pip install -r requirements_base.txt
   pip install -r requirements_app.txt
   ```
   或使用 Poetry，預設（GPU，Linux x86_64）：
   ```bash
   poetry env use 3.12
   poetry install # 若只想使用 CPU 運行，請加上 --without gpu
   ```
   安裝完成後，驗證 GPU 是否啟用：
   ```bash
    poetry run python - <<'PY'
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    PY
   ```

3. **設定 Firebase 金鑰**
   將 Firebase 服務帳戶金鑰（JSON）放在：

   ```
   src/web/key/<your-service-account>.json
   ```

4. **啟動 FastAPI 伺服器**

   ```bash
   uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8080
   ```

5. **啟動前端開發環境**

   ```bash
    cd frontend
    yarn install
    yarn run dev
   ```

   預設前端服務位於 `http://localhost:5173`，並會向本地後端發送請求。

---

## 資料夾結構

```
FactGraph/
├── data/
│   ├── raw/         # 原始資料
│   ├── interim/     # 中間產物 (user-input 等)
│   └── processed/   # 處理後結果 (news_kg_*, judge_result_* 等)
├── frontend/        # Vue 3 + Vite 前端程式
├── models/          # 下載或快取的模型權重
├── src/
│   ├── qa/
│   │   ├── verifier/    # 文字事實查證流程
│   │   ├── answerer/    # 問答流程
│   │   └── rag_mode/    # RAG 管線
│   └── web/             # FastAPI App 與路由
├── Dockerfile
├── pyproject.toml
└── requirements_*.txt
```

---

## API 概覽

| Endpoint              | 方法   | 說明                            |
| --------------------- | ---- | ----------------------------- |
| `/api/health`         | GET  | 健康檢查                          |
| `/api/verifier/query` | POST | 文字事實查證：上傳 `file` 與 `date`     |
| `/api/answerer/query` | POST | 問題回答：上傳 `file` 與 `date`       |
| `/api/rag_mode/query` | POST | RAG 模式：上傳 `file`（可附 `job_id`） |

> 回傳內容包含判斷結果、知識圖譜敘述或 RAG 生成答案，具體格式可參考 `src/web/routers/` 中的實作。

### 直接使用 FastAPI Swagger UI 進行測試

```bash
uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8080
```
---

## Docker 部署

專案提供多階段 `Dockerfile`：

* **base stage**：建立虛擬環境並安裝依賴
* **runtime stage**：複製模型及程式碼，以非 root 使用者執行

> 更完整的本地建置及雲端部署流程之操作，說明文件詳見 **DEPLOYMENT.md**

---
---

## 團隊與夥伴
**學校｜系所**

> 國立屏東科技大學｜資訊管理系

**指導教授**  
> 陳灯能 老師  

**專案成員** 
```
B11156007 張宏瑋｜前端切版、美術設計
B11156057 賴彥霖｜前端 & 後端設計、整合與部署
B11156039 黃品翰｜文書處理
```
---

## 貢獻指南

1. Fork 專案並建立新分支
2. 進行修改並撰寫／更新測試
3. 提交 Pull Request，說明變更內容與動機

---

## 授權

本專案以 **MIT License** 授權。歡迎自由使用與擴充。