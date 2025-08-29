# FactGraph – 全端部署與快取最佳化 README

> 新聞事實查核平台：**Vue 3 + Vite 前端（Firebase Hosting）**／**FastAPI 後端（Docker → Cloud Run）**／**Artifact Registry 映像與 Buildx 快取**。
>
> **目標**：冷啟動快、重建快、可追溯、可回滾、成本可控。

---

## 0. 架構總覽

```
[使用者瀏覽器]
       │  HTTPS
       ▼
Firebase Hosting (CDN)
       │  /api/** 透過 rewrite → Cloud Run
       ▼
Cloud Run (FastAPI 容器)
       │  拉取映像 from Artifact Registry (同區)
       ▼
Artifact Registry (image & build cache)
```

* **區域統一**：全部採用 `asia-southeast1`（新加坡）。
* **映像快取**：`docker buildx` + Artifact Registry 快取 repo，縮短 build 時間。
* **不可變 Tag**：以時間或 Git SHA 命名，利於追蹤與回滾。

---

## 1. 前置準備

| 工具 / 服務                     | 建議版本     | 用途                   |
| --------------------------- | -------- | -------------------- |
| Docker CLI / Desktop        | 24.x↑    | 本地建置映像、啟用 Buildx     |
| Google Cloud CLI (`gcloud`) | 530.0.0↑ | 建立 Repo、部署 Cloud Run |
| Firebase CLI                | 14.x↑    | 部署 Firebase Hosting  |
| Node.js 18.x + Yarn 1.x     | —        | 前端建置                 |
| Python 3.12 + venv          | —        | 後端本地開發               |

**GCP 權限**：用來部署的帳號需具備以下角色：

* `roles/artifactregistry.admin`（初始化階段）
* `roles/run.admin`
* `roles/iam.serviceAccountUser`
* Cloud Run 預設服務帳號需有 `roles/artifactregistry.reader`

---

## 2. 專案結構（深度 2 範例）

```bash
$ tree -L 2
.
├── Dockerfile
├── .dockerignore
├── requirements_base.txt
├── requirements_app.txt
├── models/                  # CKIP 等大型模型 (~500MB)
├── data/                    # 非必要請勿打包進映像
├── src/
│   ├── web/                 # FastAPI 入口 main.py、│ │routers
│   └── qa/                  # pipeline 與其他模組
├── frontend/
│   ├── firebase.json
│   ├── vite.config.js
│   └── src/
└── scripts/                 # 可選：部署 / 清理腳本
```

---

## 3. 共用環境變數（請先貼到你的 shell）

```bash
# ---- 基本參數 ----
export PROJECT="factgraph-38be7"
export REGION="asia-southeast1"

# ---- Artifact Registry Repositories ----
export REPO_IMG="factgraph-backend"   # 最終映像儲存庫
export REPO_CACHE="buildcache"        # Buildx 快取儲存庫

# ---- 產生不可變 Tag（時間戳） ----
export TAG="$(date +%Y%m%d-%H%M)"

# ---- 完整路徑 ----
export IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO_IMG/$REPO_IMG:$TAG"
export CACHE_REF="$REGION-docker.pkg.dev/$PROJECT/$REPO_CACHE/backend"
```

> 之後每次 build 前只要重新 export `TAG`（或寫在 script）即可。

---

## 4. 一次性初始化（沒做過才跑）

### 4.1 建立 Artifact Registry Repos（快取＆映像）

```bash
gcloud artifacts repositories create "$REPO_CACHE" \
  --repository-format=docker \
  --location="$REGION"

gcloud artifacts repositories create "$REPO_IMG" \
  --repository-format=docker \
  --location="$REGION"
```

> 已存在會報錯，無視即可。

### 4.2 Docker 登入 AR

```bash
gcloud auth configure-docker "$REGION-docker.pkg.dev"
```

### 4.3 建立並啟用 buildx builder（只需一次）

```bash
docker buildx create --name factgraphbx --driver docker-container --use
docker buildx inspect --bootstrap
```

---

## 5. 首次「暖快取」（完整建置並寫入快取）

> 目的：把 pip / 模型等大層 layer 存進快取 repo，未來 build 時直接命中。

```bash
docker buildx build \
  --file Dockerfile \
  --platform linux/amd64 \
  --cache-from=type=registry,ref="$CACHE_REF" \
  --cache-to=type=registry,ref="$CACHE_REF",mode=max \
  --tag "$IMAGE" \
  --push \
  --progress=plain .
```

完成後：

* AR 的 `buildcache/backend` 已保存所有中間層。
* AR 的 `factgraph-backend` 儲存最新映像（tag = `$TAG`）。

## 快速檢查
```bash
export REGION="asia-southeast1"
export PROJECT="factgraph-38be7"
export REPO_IMG="factgraph-backend"
gcloud artifacts docker images list \
  "$REGION-docker.pkg.dev/$PROJECT/$REPO_IMG" \
  --include-tags \
  --format="table(TAGS,DIGEST,CREATE_TIME)"
```
---

## 6. 日常開發迭代流程

### 6.1 Build + Push（同時讀/寫快取）

```bash
export TAG="$(date +%Y%m%d-%H%M)"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO_IMG/$REPO_IMG:$TAG"

docker buildx build \
  --file Dockerfile \
  --platform linux/amd64 \
  --cache-from=type=registry,ref="$CACHE_REF" \
  --cache-to=type=registry,ref="$CACHE_REF",mode=min \
  --tag "$IMAGE" \
  --push \
  --progress=plain .
```

> **不要用 `--output=type=docker`**，會多花數分鐘 export/unpack。
>
> 若需要本機測試：

```bash
docker pull "$IMAGE"
docker run --rm -p 8080:8080 "$IMAGE"
```

---

## 7. 部署到 Cloud Run
### 7.1 尋找最新的 tag & 單筆確認 (分開執行)
```bash
gcloud artifacts docker tags list \
  "${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}" \
  --sort-by=~UPDATE_TIME \
  --limit=20 \
  --format="table(UPDATE_TIME, TAG, DIGEST)"

gcloud artifacts docker images describe \
  "${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}:${TAG}" \
  --format="json(image_summary.digest,image_summary.create_time)"
```
### 7.2.A TAG 及 IMAGE 設置
```bash
# 先補上 tag
export TAG="<補上查詢到最新的 tag>"

# 組出完整 IMAGE
export IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO_IMG/$REPO_IMG:$TAG"

# 確認一下
echo "$IMAGE"
```

### 7.2.B 如何補 TAG ? (如果之前的步驟忘記上 TAG)
```bash
# 先查最新的 DIGEST
gcloud artifacts docker images list   "$REGION-docker.pkg.dev/$PROJECT/$REPO_IMG"   --include-tags   --filter='TAGS!=""'   --sort-by=~CREATE_TIME   --limit=10   --format='table(CREATE_TIME, TAGS, DIGEST)'

# 設定最新的 DIGEST
export DIGEST=<剛剛查詢到的最新 DIGEST>

# 再設定要補上的 tag
export TAG="$(date +%Y%m%d-%H%M)"

# 確認一下
echo "DIGEST=$DIGEST"
echo "TAG=$TAG"

# 最後幫 DIGEST 補上 TAG
gcloud artifacts docker tags add \
  "asia-southeast1-docker.pkg.dev/factgraph-38be7/factgraph-backend/factgraph-backend@${DIGEST}" \
  "asia-southeast1-docker.pkg.dev/factgraph-38be7/factgraph-backend/factgraph-backend:${TAG}"
```

### 7.3 執行部署
```bash
gcloud run deploy "$REPO_IMG" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --cpu=8 \
  --memory=16Gi \
  --concurrency=40 \
  --timeout=900 \
  --min-instances=1 \
  --max-instances=3 \
  --allow-unauthenticated
```

> 參數請依實測調整。`min-instances=1` 可避免冷啟動；`max-instances` 控制成本。

---

## 8. 部署後驗證與回滾

```bash
# 列出服務
gcloud run services list --platform managed --region "$REGION"

# 列出 revisions（確認新 digest）
gcloud run revisions list --service "$REPO_IMG" --region "$REGION"

# 健康檢查（假設 /health endpoint）
SERVICE_URL=$(gcloud run services describe "$REPO_IMG" --region "$REGION" --format='value(status.url)')
curl -I "$SERVICE_URL/health"
```

### 回滾至舊版

```bash
# 找到想回滾的 revision 名稱
gcloud run revisions list --service "$REPO_IMG" --region "$REGION"

# 將 100% 流量切到舊 revision
gcloud run services update-traffic "$REPO_IMG" \
  --region "$REGION" \
  --to-revisions REVISION_NAME=100
```

---

## 9. 前端：Firebase Hosting 部署

### 9.1 `frontend/firebase.json`

```jsonc
{
  "hosting": {
    "public": "dist",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "region": "asia-southeast1",
    "rewrites": [
      {
        "source": "/api/**",
        "run": {
          "serviceId": "factgraph-backend",
          "region": "asia-southeast1"
        }
      },
      { "source": "**", "destination": "/index.html" }
    ]
  }
}
```

> 若你用的是較老的 Firebase CLI 或不支援 `run`，請改用 GAE Proxy 或直接在前端寫死 Cloud Run URL。

### 9.2 建置與部署

```bash
cd frontend
yarn install       # 首次
yarn build         # 產生 dist/
firebase deploy --only hosting
```

---

## 10. FastAPI CORS 設定（後端）

`src/web/main.py`：

```python
origins = [
    "http://localhost:5173",            # Vite dev
    "http://localhost:8080",            # 其他本機測試
    "https://factgraph-38be7.web.app",  # Firebase Hosting
]
```

> 若你在其他網域測試，記得補上；或採用萬用 `"*"`（不建議正式環境）。

---

## 11. Dockerfile / .dockerignore 最佳實務

### Dockerfile（關鍵：層次穩定）

```dockerfile
# Stage: py-base
FROM python:3.12.3-slim AS py-base
WORKDIR /opt/app
COPY requirements_base.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements_base.txt

# Stage: deps
FROM py-base AS deps
WORKDIR /opt/app
COPY requirements_app.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements_app.txt

# Stage: runtime
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS runtime
WORKDIR /app
COPY --from=deps /usr/local /usr/local
COPY models/ /app/models           # 大模型層（不常改）
COPY src/    /app/src              # 程式碼層（最常改）
COPY data/   /app/data             # 若必要才放入
CMD ["python", "-m", "src.web.main"]
```

### `.dockerignore`

```gitignore
.git/
**/__pycache__/
*.pyc
venv/
.env*
# 不要把暫存資料塞進映像
/data/interim/
/data/processed/
/logs/
```

---

## 12. 疑難排解速查表

| 現象              | 可能原因                                         | 快速檢查                      | 解法                                              |
| --------------- | -------------------------------------------- | ------------------------- | ----------------------------------------------- |
| build 超慢（pip 層） | 沒寫入/讀到快取、requirements 調整                     | log 無 `CACHED`            | 加 `--cache-to`，調整 Dockerfile COPY 順序            |
| push 重傳大層       | digest 改變（層順序、時間戳）、使用 `--output=type=docker` | push log 顯示重新上傳           | build 時 `--push`，避免多次 tag/push                  |
| Cloud Run 冷啟動慢  | `min-instances=0`                            | 首次請求很慢                    | 設 `--min-instances=1`，或用 Cloud Run jobs 預熱      |
| Swagger UI 轉圈圈  | CORS 未放行、服務還沒啟動好                             | 瀏覽器 console/network       | 加入正確 origins、檢查 Cloud Run logs                  |
| `/api/...` 404  | Firebase rewrite 設錯                          | Firebase Hosting logs     | 檢查 `firebase.json` 中 `run.serviceId` / `region` |
| 回滾困難            | always 使用 `latest`                           | revisions list 找不到 digest | 改用不可變 tag，部署時指定 digest 或 tag                    |

---

## 13. 清理與成本控制

```bash
# 本地 buildx cache 狀況
docker buildx du
# 清本地 cache
docker buildx prune --all --force

# 列出遠端映像
gcloud artifacts docker images list "$REGION-docker.pkg.dev/$PROJECT/$REPO_IMG" --include-tags
# 刪除舊 digest
gcloud artifacts docker images delete \
  "$REGION-docker.pkg.dev/$PROJECT/$REPO_IMG/$REPO_IMG@sha256:<digest>" --quiet
```

> 可在 Artifact Registry 設定保留策略（Lifecycle Policy）自動刪除舊 image / cache。

---

## 14. 延伸：不用 Docker 可以嗎？

* **Cloud Run 需要容器**。若不想自己打包 Docker，可考慮 Cloud Run Source Deploy + Buildpacks，但大檔模型仍是瓶頸。
* 模型可放 GCS，容器啟動時載入（加快 build；啟動時間可能變慢）。
* 若要更細粒度更新，可拆多個服務 / 微服務，減少單個 image 體積。
---
## 15. FAQ

**Q：為什麼我明明加了 `--cache-from` 還是沒有 CACHED？**
A：你前一次 build 沒有 `--cache-to` 寫入快取、或 Dockerfile COPY 順序導致層被破壞、或快取 repo 被清除。

**Q：一定要用不可變 tag 嗎？**
A：不是硬性規定，但沒有會很痛。debug、rollback、比對版本都麻煩。

**Q：Dockerfile 裡要不要用 `--mount=type=cache,target=/root/.cache/pip`？**
A：建議用，可加速同一次 build 或同 builder 的 pip 安裝，但跨機器仍需 registry cache。

**Q：大型模型與依賴要怎麼處理？**
A：放在前層、盡量不變。或拆成 base image；或放 GCS 啟動時下載。

**Q：為什麼 `docker push` 還是重新上傳大層？**
A：Digest 改變（層順序移動、檔案 timestamp 改變、`--output=type=docker` 重新打包）。**盡量 build 時就 `--push`。**

---

> **結束語**：以上流程能確保你每次只重建「真正改動的層」，並且部署可追溯、可回滾。若你要改 Cloud Build / GitHub Actions CI/CD、自動跑測試，也可沿用同樣的 cache 策略。