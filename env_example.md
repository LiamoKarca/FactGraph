# 🧩 FactGraph `.env` 環境變數說明文件

> 本文件說明 **FactGraph / 芒狗偵探** 系統所使用的所有環境變數。
> 涵蓋：資料庫連線、模型與 API 設定、RAG 檢索參數、LlamaIndex 擴充與除錯設定。
>
> **建議使用方式：**
>
> 1. 將 `env_example.md` 搭配 `env.example` 一併存放於專案根目錄。
> 2. 複製 `env.example` 為 `.env`，再依實際環境（local / staging / production）填入。
> 3. 所有值皆以 UTF-8 編碼保存。

---

## 🧱 一、資料庫設定（Database Configuration）

| 變數名稱               | 範例值                                                     | 說明                                         |
| ------------------ | ------------------------------------------------------- | ------------------------------------------ |
| **NEO4J_URI**      | `bolt://localhost:7687`                                 | Neo4j 伺服器連線 URI。遠端部署時改為對應主機位置。             |
| **NEO4J_USER**     | `neo4j`                                                 | Neo4j 登入帳號。                                |
| **NEO4J_PASSWORD** | `abc123`                                                | Neo4j 密碼（請勿提交至公開 repo）。                    |
| **NEO4J_DATABASE** | `neo4j`                                                 | 目標資料庫，可切換為 `testing`、`experiment` 等測試用 DB。 |
| **MONGODB_URI**    | `mongodb://user:pass@localhost:27017/?authSource=admin` | MongoDB 連線字串，用於儲存查核紀錄、任務狀態與系統日誌。           |

---

## 🧠 二、模型與 API 設定（Model & LLM Configuration）

| 變數名稱                      | 範例值                                         | 說明                                  |
| ------------------------- | ------------------------------------------- | ----------------------------------- |
| **MODEL_CONFIG_ENDPOINT** | `http://127.0.0.1:1234/v1/chat/completions` | 本地模型端點（支援 LM Studio、vLLM、Ollama 等）。 |
| **MODEL_ID**              | `google/gemma-3-4b`                         | 本地模型 ID。                            |
| **OPENAI_API_KEY**        | `sk-xxxx`                                   | OpenAI 金鑰。                          |
| **GPT_API**               | `${OPENAI_API_KEY}`                         | 相容性用途，部分模組（如 `paths.py`）仍需此變數。      |
| **GPT_MODEL**             | `gpt-5`                                     | 主回答／最終判斷模型。                         |
| **OPENAI_CHAT_MODEL**     | `gpt-4o-mini`                               | 預設聊天模型（供 `dirt_removal.py` 使用）。     |
| **EXTRACT_MODEL**         | `gpt-4o-mini`                               | 三元組抽取模型（用於抽取事實與 KG 構建）。             |
| **EXPAND_GPT_MODEL**      | `gpt-4o-mini`                               | 查詢擴充詞模型，用於生成同義詞與相關詞以強化檢索。           |
| **DEDUP_GPT_MODEL**       | `gpt-4o-mini`                               | 去重（Deduplication）模型，用於過濾重複檢索結果。     |
| **DEDUP_USE_GPT**         | `1`                                         | 是否啟用 GPT 輔助去重：`1`=啟用、`0`=關閉。        |

> 💡 **建議**：大模型（如 GPT-5）用於最終生成與評估，小模型（如 GPT-4o-mini）用於輕量化抽取與過濾。

---

## 🔍 三、RAG 檢索與知識比對設定（Retrieval & Knowledge Alignment）

### 🔸 檢索門檻與數量（`search.py` / `subq_engine.py` 使用）

| 變數名稱           | 範例值   | 說明                         |
| -------------- | ----- | -------------------------- |
| **DUP_TH**     | `0.8` | 語義去重閾值。當兩向量相似度 ≥ 此值時視為重複。  |
| **SIM_TH**     | `0.6` | cosine 相似度初篩門檻（過低會檢索過多噪音）。 |
| **TOP_K**      | `100` | 最終返回的最大候選數。                |
| **LLM_ROUNDS** | `2`   | LLM 抽取輪數供純向量檢索使用，建議 1–3。如果啟用 USE_AGENT_RETRIEVAL，則此變數可設為 0，不影響 Agent 流程。|

### 🔸 檢索來源與限制

| 變數名稱                    | 範例值              | 說明                      |
| ----------------------- | ---------------- | ----------------------- |
| **RAG_ALLOWED_SOURCES** | `PTS,CNA,MOI,EY` | 來源白名單。空字串 = 不過濾。        |
| **RAG_CAND_FACTOR**     | `8`              | 候選池放大倍率（越高則召回率提升但速度下降）。 |

### 🔸 檢索嚴格度（evidence 命中規則）

| 變數名稱                           | 範例值 | 說明                                 |
| ------------------------------ | --- | ---------------------------------- |
| **EVIDENCE_MIN_HITS**          | `2` | 每句至少需命中幾個關鍵詞。                      |
| **REQUIRE_ANCHOR_IN_EVIDENCE** | `1` | 是否要求 anchor（head）必須命中：`1`=是、`0`=否。 |

> ⚙️ **說明**：
> 以上參數共同控制 RAG 檢索精度與嚴謹度。
> FactGraph 採用「語義擴充 + 逐句比對 + 向量召回」三層式策略，以確保一致性與可追溯性。

---

## 🦙 四、LlamaIndex SubQuestion 擴充設定（Multi-Hop Reasoning）

| 變數名稱                       | 範例值 | 說明                                           |
| -------------------------- | --- | -------------------------------------------- |
| **ENABLE_LLAMAINDEX_SUBQ** | `1` | 是否啟用 LlamaIndex 的 SubQuestion 模組（1=啟用，0=關閉）。 |
| **SUBQ_MAX**               | `6` | 最多允許生成的子問題數（建議 3–6）。                         |

> 🔍 **用途**：
> 當查詢過於複雜或包含多層事實時，系統會自動將主問題拆解為多個子問題進行檢索與比對，再由 LLM 統整最終判斷。
> 此模組可顯著提升長文本、多跳關聯類任務的查核品質。

---

## 🧾 五、除錯與日誌設定（Debug & Logging）

| 變數名稱                    | 範例值                                                       | 說明                                                     |
| ----------------------- | --------------------------------------------------------- | ------------------------------------------------------ |
| **VERIFIER_DEBUG**      | `0`                                                       | 除錯開關：`1`=啟用詳細紀錄、`0`=關閉。                                |
| **VERIFIER_DEBUG_PATH** | `data/processed/verifier/debug/verifier_search_debug.log` | 除錯日誌輸出位置。若部署於 Cloud Run，可改用 `/tmp/verifier_debug.log`。 |

> 🧩 **說明**：
> 啟用除錯模式後，系統會輸出：
>
> * 各階段檢索與匹配紀錄；
> * GPT 判斷過程；
> * 例外錯誤追蹤（stack trace）。

---

## 🧭 六、版本與維護狀態（Version & Maintenance Notes）

| 區塊            | 狀態     | 備註                  |
| ------------- | ------ | ------------------- |
| 資料庫設定         | ✅ 穩定   | 已支援 Neo4j + MongoDB |
| 模型設定          | ✅ 穩定   | 支援 GPT 與本地模型雙管架構    |
| RAG 檢索        | ⚙️ 可調  | 可依新聞主題微調閾值與候選參數     |
| SubQuestion   | 🧪 實驗性 | 適用於多跳推理與長文本驗證       |
| Debug Logging | ✅ 穩定   | 建議於測試環境啟用           |

---
## ✳️ 附錄： `.env` 的完整範例（敏感值已遮罩）

下面為專案根目錄實際使用的 `.env` 範本，我已將密碼 / API key 等敏感值遮罩（請以真實環境填入 `.env`）：
```
# =======================================
# 🔗 DATABASE & STORAGE CONFIGURATION
# =======================================

# === Neo4j Knowledge Graph ===
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=*****REDACTED*****
NEO4J_DATABASE=neo4j

# === MongoDB (事實查核紀錄 / Job Tracking) ===
MONGODB_URI=mongodb://*****REDACTED*****@localhost:27017/?authSource=admin


# =======================================
# 🧠 MODEL & LLM CONFIGURATION
# =======================================

# === Local / Custom Endpoint ===
MODEL_CONFIG_ENDPOINT=http://127.0.0.1:1234/v1/chat/completions
MODEL_ID=google/gemma-3-4b

# === OpenAI API Keys ===
OPENAI_API_KEY=*****REDACTED*****
GPT_API=${OPENAI_API_KEY}  # 與部分模組相容（paths.py 會讀）

# === Model Roles ===
GPT_MODEL=gpt-5                 # 主回答 / 判斷模型（paths.py 會讀）
OPENAI_CHAT_MODEL=gpt-4o-mini   # dirt_removal 預設使用的聊天模型
EXTRACT_MODEL=gpt-4o-mini       # 三元組抽取模型（extract.py）
EXPAND_GPT_MODEL=gpt-4o-mini    # 查詢擴充詞用（search.py）
DEDUP_GPT_MODEL=gpt-4o-mini     # 重複項過濾用（dedup.py）
DEDUP_USE_GPT=0                 # Dedup 的 GPT 輔助：1=啟用, 0=關閉


# =======================================
# 🔍 RAG & KNOWLEDGE RETRIEVAL SETTINGS
# =======================================

# === 檢索門檻與數量（供 search.py / subq_engine.py 使用） ===
DUP_TH=0.8                      # 語義去重臨界值（相似≥此值視為重覆）
SIM_TH=0.40                   # cosine 初篩門檻
TOP_K=400                       # 最終返回的最大候選數，視情況增加可讓去重後仍有料
LLM_ROUNDS=2                    # LLM 三元組抽取輪數，如果啟用 USE_AGENT_RETRIEVAL，此變數可設為 0

# === 檢索限制 ===
RAG_CAND_FACTOR=14                    # 候選池放大倍率（越大越多樣，但速度較慢）

# === 檢索嚴格度（evidence 命中規則） ===
EVIDENCE_MIN_HITS=2                  # 每句至少需命中幾個關鍵詞
REQUIRE_ANCHOR_IN_EVIDENCE=0         # 1=要求 anchor（head）命中, 0=可略過

# ============================
# 🧭 同名歧義過濾（用來要求「關聯實體交集」）
# ============================
DISAMBIG_REQUIRE_COENTITY=0
DISAMBIG_MIN_COMMON=0


# =======================================
# 🦙 LANGCHAIN SETTINGS
# =======================================

# 啟用由 LangGraph Agent 進行檢索（多輪智慧規劃）
USE_AGENT_RETRIEVAL=1       # 1=啟用, 0=關閉

# =======================================
# 只用離線 PG、本地知識（圖譜檢索設定）
# =======================================
ENABLE_LI_PG=1
ENABLE_LI_ONLINE=0
ENABLE_VECTOR_FALLBACK=1 # 向量備援（本地 PG / 線上 PG 都沒命中時啟用）
LI_PG_INDEX_JSON=data/processed/knowledge-graph/pg_index.json
# =======================================

#ENABLE_LI_PG=0, ENABLE_LI_ONLINE=0, ENABLE_VECTOR_FALLBACK=1     # 只用向量檢索
#ENABLE_LI_PG=0, ENABLE_LI_ONLINE=1, ENABLE_VECTOR_FALLBACK=0/1   # 只用線上檢索（有無向量備援）

LI_PG_HOPS=3                 # 使用本地知識圖譜檢索的跳數
LI_ONLINE_HOPS=3             # 使用線上知識圖譜檢索的跳數
LI_PG_TOPK=200               # ↑ 放寬本地 PG 檢索候選數（避免早停），可依機器調整

# === 離線 PG 原始 CSV 與欄位 ===
LI_PG_RAW_CSV=data/raw/knowledge-graph/neo4j-kg-raw-graph.csv
LI_PG_DATE_FIELD=date
LI_PG_EVIDENCE_FIELD=evidence
LI_PG_MIN_EVI=1                # 最少證據數過濾（避免噪音）

# ============================
# 🕒 時間窗過濾
# ============================
FILTER_TIME_WINDOW=0 # 開啟時間窗過濾（1=開, 0=關）
TIME_WINDOW_DAYS=60

# =======================================
# ⚙️ 自動調整跳數設定
# =======================================
AUTO_HOPS_ENABLE=1          # 開啟自動調整跳數（預設開）
AUTO_HOPS_START=2           # 初始跳數（建議 2）
AUTO_HOPS_MAX=3             # 最高跳數（建議 3）
AUTO_HOPS_MIN_HITS=10        # 命中數不足這個值就升一級（可依圖密度微調）
AUTO_HOPS_TOPK=50           # 每次查詢的 top_k（避免一次拉太多）

# =======================================
# 🧾 LOGGING & DEBUGGING
# =======================================

VERIFIER_DEBUG=1                      # 啟用驗證器除錯日誌（1=啟用, 0=關閉）
VERIFIER_DEBUG_PATH=data/processed/verifier/debug/verifier_search_debug.log

# =======================================
# ♻️ ReAct 代理（StateGraph）循環門檻 / 節流
# =======================================

# —— 安全防爆（輸入/證據節流）——
MAX_AGENT_CHARS=12000         # 進入 Agent 的新聞最大字元數（硬性截斷）
MAX_EVID_CHARS=150           # 每條證據描述的最長字元（避免 tool payload 肥大）

# —— 迴圈收斂條件（達標/停損）——
AGENT_MAX_STEPS=16            # 最多工具步數（達到就收斂）
AGENT_NO_NEW_PATIENCE=8       # 連續幾步沒有新增命中就放行收斂
AGENT_RECURSION_FACTOR=3      # 每輪約 3 個超步（agent→tools→accumulate），預設 3 即可。
AGENT_RECURSION_EXTRA=3      # 保險係數，避免邊界卡死，預設 3。
AGENT_MIN_PER_TRIPLE=0        # 每個 triple 至少要有幾條命中，如果設為0，則交由耐心/步數控制
AGENT_TOTAL_TARGET=0        # 全局至少命中幾條證據才算「充足」，同上
AGENT_TOP_K_MAX=200        # Agent 每次檢索的最大候選數

# ============================
# 🧠 Embeddings / ST 相關
# ============================
# ✅ 指定正式的 Sentence-Transformers 模型（避免臨時 mean pooling）
SENTENCE_TRANSFORMERS_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 避免 accelerate 把權重掛到 meta / 非預期裝置（與程式中一致）
ACCELERATE_DISABLE_DEVICE_MAP=1
TRANSFORMERS_NO_ACCELERATE=1

# ============================
# 🔤 查詢端同義規則（外部檔；可熱更新）
# ============================
# 不提供檔案時 → 不展開同義（保持原行為）
ALIAS_RULES_JSON=data/processed/knowledge-graph/alias_rules.json
# >0 代表啟用熱更新（秒）；0 表示每次呼叫以當下載入為準（不輪詢）
ALIAS_RELOAD_SEC=5

# ========= LLM-augmented Alias Rules =========
# 是否在 `alias` 指令時自動用 LLM 生成/擴充規則（1=開, 0=關）
ALIAS_LLM_ENABLE=1
# 使用的模型（預設沿用 OPENAI_CHAT_MODEL；也可填 gpt-4o-mini / gpt-4.1-mini 等）
ALIAS_LLM_MODEL=${OPENAI_CHAT_MODEL}
# 回應 token 上限與溫度
ALIAS_LLM_MAX_TOKENS=8000
ALIAS_LLM_TEMPERATURE=0.2
# 送進 LLM 的實體抽樣上限（避免爆料）
ALIAS_LLM_HEAD_TOPN=80
ALIAS_LLM_TAIL_TOPN=80
ALIAS_LLM_REL_TOPN=120
ALIAS_LLM_REL_CHUNK=10
ALIAS_LLM_HEAD_CHUNK=10
ALIAS_LLM_TAIL_CHUNK=10
ALIAS_LLM_GAP_SEC=0.4
ALIAS_PROGRESS_VERBOSE=1
ALIAS_WRITE_EVERY=1      # 每批即寫；改 5 = 每 5 批再寫一次
OPENAI_TIMEOUT_SEC=120

```

> 注意事項：
>
> - 請務必不要把包含真實密鑰或密碼的 `.env` 提交到公開的 Git 倉庫；把本檔（`env_example.md`）作為示例即可。
> - 若要在本地測試，請在專案根目錄建立 `.env` 並貼入真實值。參考：`.env.local`, `.env.production` 命名慣例。
> - 若使用 Docker 部署，可在 `docker-compose.yml` 或 Dockerfile 中讀取環境變數，或把敏感值放到雲端密鑰管理服務（Hashicorp Vault / GCP Secret Manager / AWS Secrets Manager）。
---
