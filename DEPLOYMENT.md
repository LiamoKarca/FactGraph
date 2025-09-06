# FactGraph – 後端部署與快取最佳實務（Cloud Run + Artifact Registry）

> 後端：**FastAPI / Docker** → **Artifact Registry** → **Cloud Run**
> 目標：**可追溯（不可變 Tag）**、**快取加速（Buildx + Registry Cache）**、**部署穩定可回滾**

---

## 0) 架構與命名

```
[Dev Machine] --buildx--> Artifact Registry (images & build-cache) --> Cloud Run
```

* **區域統一**：`asia-southeast1`
* **Repository**

  * 最終映像：`factgraph-backend`
  * Buildx 快取：`buildcache`（建議獨立）
* **標籤策略**：時間戳 `YYYYMMDD-HHMM`（不可變，便於比對與回滾）

---

## 1) 需求與權限

* 工具：Docker 24+、gcloud 530+、（可選）Firebase CLI
* GCP 角色（部署帳號）：

  * `roles/artifactregistry.admin`（初始化期）
  * `roles/run.admin`
  * `roles/iam.serviceAccountUser`
  * （Cloud Run 服務帳號）`roles/artifactregistry.reader`

---

## 2) 一次性初始化（沒做過才跑）

```bash
# 2.1 建立 Repos（存在會報錯，無視即可）
export REGION="asia-southeast1"
export PROJECT="factgraph-38be7"
export REPO_IMG="factgraph-backend"
export REPO_CACHE="buildcache"

gcloud artifacts repositories create "$REPO_IMG" \
  --repository-format=docker --location="$REGION" || true

gcloud artifacts repositories create "$REPO_CACHE" \
  --repository-format=docker --location="$REGION" || true

# 2.2 Docker 登入 Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# 2.3 建立 buildx builder（僅首次）
docker buildx create --name factgraphbx --driver docker-container --use
docker buildx inspect --bootstrap
```

---

## 3) 環境變數（每次部署前先備妥）

> **務必用 ASCII 雙引號**（不要用全形 ” “）。
> 下列指令可貼上整段執行；缺值會立刻中止，避免出現 `-docker.pkg.dev///` 的錯誤。

```bash
# 3.1 基本參數（缺一不可）
export REGION=${REGION:-"asia-southeast1"}; : "${REGION:?REGION missing}"
export PROJECT=${PROJECT:-"factgraph-38be7"}; : "${PROJECT:?PROJECT missing}"
export REPO_IMG=${REPO_IMG:-"factgraph-backend"}; : "${REPO_IMG:?REPO_IMG missing}"

# 3.2 Buildx Registry Cache（可選；若不用快取就不要設定）
export CACHE_REF="${REGION}-docker.pkg.dev/${PROJECT}/buildcache/docker:${REPO_IMG}"

# 3.3 本次不可變標籤
export TAG="$(date +%Y%m%d-%H%M)"

# 3.4 完整 Image 路徑（repo 名與 image 名相同的設計）
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}:${TAG}"

# 3.5 快速自檢
echo "REGION=$REGION"
echo "PROJECT=$PROJECT"
echo "REPO_IMG=$REPO_IMG"
echo "TAG=$TAG"
echo "IMAGE=$IMAGE"
echo "CACHE_REF=${CACHE_REF:-<none>}"
```

---

## 4) Build & Push（兩種模式）

### 4.1 推薦：使用 Registry Cache（快）

```bash
docker buildx build \
  --file Dockerfile \
  --platform linux/amd64 \
  --cache-from=type=registry,ref="${CACHE_REF}" \
  --cache-to=type=registry,ref="${CACHE_REF}",mode=min \
  --tag "${IMAGE}" \
  --push \
  --progress=plain .
```

### 4.2 簡化：不使用 Cache（保證一定能推上去）

```bash
docker buildx build \
  --file Dockerfile \
  --platform linux/amd64 \
  --tag "${IMAGE}" \
  --push \
  --progress=plain .
```

> **不要**使用 `--output=type=docker`（會慢且破壞層快取）。

---

## 5) 驗證推送結果（避免「看不到新標籤」的誤會）

> `images list` 是「digest 粒度」，由 **digest 的建立時間** 排序，不一定看得出剛下的 tag。
> 使用 **tags list** 最清楚。

```bash
# 5.1 看 tag 清單（建議首選）
gcloud artifacts docker tags list \
  "${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}" \
  --sort-by=~UPDATE_TIME \
  --limit=20 \
  --format="table(UPDATE_TIME, TAG, DIGEST)"

# 5.2 單筆確認該 tag 是否存在
gcloud artifacts docker images describe \
  "${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}:${TAG}" \
  --format="json(image_summary.digest,image_summary.create_time)"

# 5.3 也可用 buildx 讀 manifest（看多平台清單）
docker buildx imagetools inspect \
  "${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}:${TAG}"
```

（如需舊式 `images list`，請拿掉 filter 以免混淆。）

---

## 6) 部署到 Cloud Run

```bash
# 若要讓 Cloud Run 永遠抓最新穩定版，也可同步更新 latest（可選）
gcloud artifacts docker tags add \
  "${IMAGE}" \
  "${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}:latest"

# 正式部署
gcloud run deploy "${REPO_IMG}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --cpu=8 \
  --memory=16Gi \
  --concurrency=40 \
  --timeout=900 \
  --min-instances=1 \
  --max-instances=3 \
  --allow-unauthenticated
```

> 參數依實測調整；`min-instances=1` 可避冷啟動，`max-instances` 控制成本。

---

## 7) 部署後驗證與回滾

```bash
# 7.1 驗證服務與修訂
gcloud run services list --platform=managed --region "${REGION}"
gcloud run revisions list --service "${REPO_IMG}" --region "${REGION}"

# 7.2 健康檢查（假設 /health）
SERVICE_URL=$(gcloud run services describe "${REPO_IMG}" --region "${REGION}" --format='value(status.url)')
curl -I "${SERVICE_URL}/health"

# 7.3 回滾至舊版（把 100% 流量切回指定 revision）
gcloud run services update-traffic "${REPO_IMG}" \
  --region "${REGION}" \
  --to-revisions REVISION_NAME=100
```

---

## 8) Firebase Hosting（前端）要點（選讀）

`frontend/firebase.json`（重點是把 `/api/**` rewrite 到 Cloud Run）：

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

部署：

```bash
cd frontend
yarn install
yarn build
firebase deploy --only hosting
```

---

## 9) Dockerfile / .dockerignore 提示

* 把「**不常變**的大層」（模型、base 依賴）放前面，程式碼層放後面 → 更容易命中快取
* pip 安裝層加 `--mount=type=cache,target=/root/.cache/pip`
* `.dockerignore` 排除 `venv/`、`__pycache__/`、暫存資料（`data/interim/`, `data/processed/`）等

---

## 10) 疑難排解——**最常見的 6 個坑**

| 現象                                             | 快速檢查                              | 原因                                         | 解法                                              |
| ---------------------------------------------- | --------------------------------- | ------------------------------------------ | ----------------------------------------------- |
| `invalid reference format: -docker.pkg.dev///` | `echo $REGION $PROJECT $REPO_IMG` | 必填環境變數空值                                   | 第 3 節加 `: "${VAR:?missing}"`，先 Echo 再 Build     |
| `registry cache exporter requires ref`         | `echo $CACHE_REF`                 | `CACHE_REF` 空值                             | 先移除 `--cache-*` 或正確設定完整 `registry/repo:tag`     |
| 新 tag 推上去但 `images list` 看不到                   | 用 **tags list**                   | `images list` 是 digest 粒度、排序看 digest 的建立時間 | 依第 5 節改用 `docker tags list` 或 `images describe` |
| Push 重傳大層很慢                                    | 看 build log 有無 `CACHED`           | 層順序或內容變動、或沒寫入快取                            | 穩定 Dockerfile 層次、使用 `--cache-to`                |
| Cloud Run 首呼叫很慢                                | `min-instances`                   | 冷啟動                                        | `--min-instances=1` 或預熱                         |
| CORS/Swagger 卡住                                | 瀏覽器 Console                       | CORS origins 漏加                            | 後端補 `origins` 或暫時 `*`（正式不建議）                    |

---

## 11) 清理與成本控管

```bash
# 本機 buildx 快取量
docker buildx du
docker buildx prune --all --force

# Artifact Registry - 列出 / 刪除
gcloud artifacts docker tags list \
  "${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}" --limit=50

gcloud artifacts docker images delete \
  "${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}@sha256:<digest>" \
  --quiet
```

> 建議在 Artifact Registry 設 **Lifecycle Policy** 自動淘汰老舊 digest / cache。

---

## 12) 一鍵部署腳本（可丟到 `scripts/deploy.sh`）

```bash
#!/usr/bin/env bash
set -euo pipefail

# === 必填 ===
REGION="${REGION:-asia-southeast1}"
PROJECT="${PROJECT:-factgraph-38be7}"
REPO_IMG="${REPO_IMG:-factgraph-backend}"

: "${REGION:?REGION missing}"
: "${PROJECT:?PROJECT missing}"
: "${REPO_IMG:?REPO_IMG missing}"

# === 可選：Registry Cache ===
CACHE_REF="${CACHE_REF:-${REGION}-docker.pkg.dev/${PROJECT}/buildcache/docker:${REPO_IMG}}"

# === Tag & Image ===
TAG="$(date +%Y%m%d-%H%M)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}:${TAG}"

echo "[info] REGION=${REGION}"
echo "[info] PROJECT=${PROJECT}"
echo "[info] REPO_IMG=${REPO_IMG}"
echo "[info] TAG=${TAG}"
echo "[info] IMAGE=${IMAGE}"
echo "[info] CACHE_REF=${CACHE_REF}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# 若想關掉快取，移除下面兩行的 ${...:+...} 包裝，或直接刪兩個參數
docker buildx build \
  --file Dockerfile \
  --platform linux/amd64 \
  ${CACHE_REF:+--cache-from=type=registry,ref="${CACHE_REF}"} \
  ${CACHE_REF:+--cache-to=type=registry,ref="${CACHE_REF}",mode=min} \
  --tag "${IMAGE}" \
  --push \
  --progress=plain .

echo "[info] Verify tag exists in AR..."
gcloud artifacts docker images describe \
  "${REGION}-docker.pkg.dev/${PROJECT}/${REPO_IMG}/${REPO_IMG}:${TAG}" \
  --format="json(image_summary.digest,image_summary.create_time)" \
  | jq '.'

echo "[info] Deploy to Cloud Run..."
gcloud run deploy "${REPO_IMG}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --cpu=8 --memory=16Gi --concurrency=40 --timeout=900 \
  --min-instances=1 --max-instances=3 \
  --allow-unauthenticated

echo "[done] URL:"
gcloud run services describe "${REPO_IMG}" --region "${REGION}" --format='value(status.url)'
```

> 使用方式：
>
> ```bash
> chmod +x scripts/deploy.sh
> REGION=asia-southeast1 PROJECT=factgraph-38be7 REPO_IMG=factgraph-backend ./scripts/deploy.sh
> ```

---

### 備註

* 你的 Repo 名與 Image 名相同（`factgraph-backend`）。若未來改為不同（例如 repo=`factgraph-backend`, image=`api`），請把 `IMAGE` 改為：
  `.../$REPO_IMG/<image-name>:$TAG`
* 所有變數與路徑，**一律先 `echo` 檢查**；90% 問題都是空值或打錯引號造成。
