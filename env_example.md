# 🧩 FactGraph `.env` 環境變數說明文件

> 本文件說明 **FactGraph / 芒狗偵探** 系統所使用的所有環境變數。
> 涵蓋：資料庫連線、模型與 API 設定、RAG 檢索參數、LangChain Agent、圖譜檢索、Reranking 融合、查詢生成與除錯設定。
>
> **建議使用方式：**
>
> 1.  將 `env_example.md` 搭配 `env.example` 一併存放於專案根目錄。
> 2.  複製 `env.example` 為 `.env`，再依實際環境（local / staging / production）填入。
> 3.  所有值皆以 UTF-8 編碼保存。

-----

## 🧱 一、資料庫設定（Database Configuration）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **NEO4J\_URI** | `bolt://localhost:7687` | Neo4j 伺服器連線 URI。遠端部署時改為對應主機位置。 |
| **NEO4J\_USER** | `neo4j` | Neo4j 登入帳號。 |
| **NEO4J\_PASSWORD** | `*****REDACTED*****` | Neo4j 密碼（請勿提交至公開 repo）。 |
| **NEO4J\_DATABASE** | `neo4j` | 目標資料庫，可切換為 `testing`、`experiment` 等測試用 DB。 |
| **MONGODB\_URI** | `mongodb://*****REDACTED*****@localhost:27017/?authSource=admin` | MongoDB 連線字串，用於儲存查核紀錄、任務狀態與系統日誌。 |

-----

## 🧠 二、模型與 API 設定（Model & LLM Configuration）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **MODEL\_CONFIG\_ENDPOINT** | `http://127.0.0.1:1234/v1/chat/completions` | 本地模型端點（支援 LM Studio、vLLM、Ollama 等）。 |
| **MODEL\_ID** | `google/gemma-3-4b` | 本地模型 ID。 |
| **OPENAI\_API\_KEY** | `sk-*****REDACTED*****` | OpenAI 金鑰。 |
| **GPT\_API** | `${OPENAI_API_KEY}` | 相容性用途，部分模組（如 `paths.py`）仍需此變數。 |
| **GPT\_MODEL** | `gpt-5` | 主回答／最終判斷模型 [cite: 2]。 |
| **OPENAI\_RESP\_MODEL** | `gpt-5` | `dirt_removal` 預設使用的聊天模型。 |
| **OPENAI\_CHAT\_MODEL** | `gpt-4.1` | `dirt_removal` 備用的聊天模型。 |
| **DIRT\_FILTER\_PROMPT\_PATH** | `src/qa/verifier/prompts/dirt_filter.md` | `dirt_removal` 使用的外部提示詞路徑。 |
| **EXTRACT\_MODEL** | `gpt-4o-mini` | 三元組抽取模型（用於抽取事實與 KG 構建）。 |
| **EXPAND\_GPT\_MODEL** | `gpt-4o-mini` | 查詢擴充詞模型，用於生成同義詞與相關詞以強化檢索。 |
| **DEDUP\_GPT\_MODEL** | `gpt-4o-mini` | 去重（Deduplication）模型，用於過濾重複檢索結果 [cite: 3]。 |
| **DEDUP\_USE\_GPT** | `0` | 是否啟用 GPT 輔助去重：`1`=啟用、`0`=關閉 [cite: 3]。 |

-----

## 🔍 三、RAG 檢索與知識比對設定（Retrieval & Knowledge Alignment）

### 🔸 檢索門檻與數量（`search.py` / `subq_engine.py` 使用）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **DUP\_TH** | `0.8` | 語義去重閾值。當兩向量相似度 ≥ 此值時視為重複 [cite: 4]。 |
| **SIM\_TH** | `0.45` | cosine 相似度初篩門檻（過低會檢索過多噪音）。 |
| **TOP\_K** | `150` | 最終返回的最大候選數。 |
| **LLM\_ROUNDS** | `0` | 初始 LLM 三元組抽取輪數。若啟用 `USE_AGENT_RETRIEVAL`，此變數可設為 0 [cite: 5]。 |

### 🔸 檢索來源與限制

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **RAG\_CAND\_FACTOR** | `6` | 候選池放大倍率（越高則召回率提升但速度下降）。 |

### 🔸 檢索嚴格度（evidence 命中規則）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **EVIDENCE\_MIN\_HITS** | `1` | 每句至少需命中幾個關鍵詞 [cite: 6]。 |
| **REQUIRE\_ANCHOR\_IN\_EVIDENCE** | `1` | 是否要求 anchor（head）必須命中：`1`=是、`0`=否 [cite: 6]。 |

### 🔸 同名歧義過濾（Disambiguation）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **DISAMBIG\_REQUIRE\_COENTITY** | `1` | 要求證據句要有共同實體，降低人名/政黨詞的假陽性。 |
| **DISAMBIG\_MIN\_COMMON** | `1` | 最少共同實體數。 |

-----

## 🦙 四、LangChain Agent 設定（LangChain Settings）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **USE\_AGENT\_RETRIEVAL** | `1` | 是否啟用由 LangGraph Agent 進行檢索（多輪智慧規劃）：`1`=啟用、`0`=關閉 [cite: 7]。 |

-----

## ⚙️ 五、圖譜檢索設定（Knowledge Graph Retrieval）

### 🔸 檢索來源（Retrieval Sources）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **ENABLE\_LI\_PG** | `1` | 啟用本地 LlamaIndex 屬性圖譜（Property Graph）檢索。 |
| **ENABLE\_LI\_ONLINE** | `0` | 啟用線上圖譜檢索。 |
| **ENABLE\_VECTOR\_FALLBACK** | `1` | 啟用向量備援（當圖譜檢索無命中時）。 |

### 🔸 LlamaIndex PG 設定（LlamaIndex PG Settings）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **LI\_PG\_INDEX\_JSON** | `data/processed/knowledge-graph/pg_index.json` | LlamaIndex 屬性圖譜索引檔路徑。 |
| **LI\_PG\_HOPS** | `3` | 本地知識圖譜檢索的跳數 [cite: 8]。 |
| **LI\_ONLINE\_HOPS** | `3` | 線上知識圖譜檢索的跳數。 |
| **LI\_PG\_TOPK** | `80` | 本地 PG 檢索候選數（避免早停）。 |

### 🔸 離線 PG 資料來源（Offline PG Data Source）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **LI\_PG\_RAW\_CSV** | `data/raw/knowledge-graph/neo4j-kg-raw-graph.csv` | 離線 PG 原始 CSV 資料。 |
| **LI\_PG\_DATE\_FIELD** | `date` | CSV 中儲存日期的欄位名稱。 |
| **LI\_PG\_EVIDENCE\_FIELD** | `evidence` | CSV 中儲存證據的欄位名稱。 |
| **LI\_PG\_MIN\_EVI** | `1` | 最少證據數過濾（避免噪音） [cite: 9]。 |

### 🔸 時間窗過濾（Time Window）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **FILTER\_TIME\_WINDOW** | `0` | 開啟時間窗過濾：`1`=開啟、`0`=關閉。 |
| **TIME\_WINDOW\_DAYS** | `0` | 時間窗天數（`0`=不設限）。 |

### 🔸 自動調整跳數（Auto Hops）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **AUTO\_HOPS\_ENABLE** | `1` | 開啟自動調整跳數 [cite: 10]。 |
| **AUTO\_HOPS\_START** | `2` | 初始跳數。 |
| **AUTO\_HOPS\_MAX** | `3` | 最高跳數。 |
| **AUTO\_HOPS\_MIN\_HITS** | `8` | 命中數不足此值則增加跳數 [cite: 11]。 |
| **AUTO\_HOPS\_TOPK** | `50` | 每次查詢的 top\_k（避免一次拉太多）。 |

-----

## ♻️ 六、ReAct 代理 (StateGraph) 節流設定

### 🔸 安全防爆（Throttling） [cite: 12]

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **MAX\_AGENT\_CHARS** | `12000` | 進入 Agent 的新聞最大字元數（硬性截斷）。 |
| **MAX\_EVID\_CHARS** | `300` | 每條證據描述的最長字元（避免 tool payload 肥大）。 |

### 🔸 迴圈收斂（Convergence）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **AGENT\_MAX\_STEPS** | `8` | 最多工具步數（達到就收斂）。 |
| **AGENT\_NO\_NEW\_PATIENCE** | `4` | 連續幾步沒有新增命中就放行收斂 [cite: 13]。 |
| **AGENT\_RECURSION\_FACTOR**| `3` | 每輪約 3 個超步（agent→tools→accumulate）。 |
| **AGENT\_RECURSION\_EXTRA** | `3` | 保險係數，避免邊界卡死。 |
| **AGENT\_MIN\_PER\_TRIPLE** | `0` | 每個 triple 至少需命中幾條 [cite: 14]。`0`=由耐心/步數控制。 |
| **AGENT\_TOTAL\_TARGET** | `0` | 全局至少命中幾條證據才算「充足」。`0`=由耐心/步數控制。 |
| **AGENT\_TOP\_K\_MAX** | `140` | Agent 每次檢索的最大候選數。 |

-----

## 🔤 七、查詢同義詞規則 (Alias Rules)

### 🔸 靜態與即時擴充 [cite: 15]

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **ALIAS\_RULES\_JSON** | `data/processed/knowledge-graph/alias_rules.json` | 外部同義詞規則檔路徑。 |
| **ALIAS\_LLM\_RUNTIME\_ENABLE**| `1` | 是否啟用查詢時即時 LLM 擴充同義詞：`1`=開、`0`=關。 |
| **ALIAS\_RELOAD\_SEC** | `8` | `alias_rules.json` 熱更新間隔（秒）。 |

### 🔸 LLM 輔助生成 (CLI `alias` 指令) [cite: 15]

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **ALIAS\_LLM\_ENABLE** | `1` | 是否在 `alias` 指令時自動用 LLM 生成/擴充規則。 |
| **ALIAS\_LLM\_MODEL** | `${OPENAI_CHAT_MODEL}` | 用於生成同義詞的模型。 |
| **ALIAS\_LLM\_MAX\_TOKENS** | `8000` | LLM 回應 token 上限。 |
| **ALIAS\_LLM\_TEMPERATURE**| `0.2` | LLM 生成溫度。 |
| **ALIAS\_LLM\_HEAD\_TOPN** | `80` | 送進 LLM 的 Head 實體抽樣上限。 |
| **ALIAS\_LLM\_TAIL\_TOPN** | `80` | 送進 LLM 的 Tail 實體抽樣上限。 |
| **ALIAS\_LLM\_REL\_TOPN** | `120` | 送進 LLM 的 Relation 實體抽樣上限。 |
| **ALIAS\_LLM\_REL\_CHUNK** | `10` | Relation 批次大小。 |
| **ALIAS\_LLM\_HEAD\_CHUNK** | `10` | Head 批次大小。 |
| **ALIAS\_LLM\_TAIL\_CHUNK** | `10` | Tail 批次大小。 |
| **ALIAS\_LLM\_GAP\_SEC** | `0.4` | 每批次 LLM 呼叫間隔秒數。 |
| **ALIAS\_PROGRESS\_VERBOSE**| `1` | 顯示詳細進度。 |
| **ALIAS\_WRITE\_EVERY** | `1` | 每幾批次回寫一次檔案。 |
| **OPENAI\_TIMEOUT\_SEC** | `120` | OpenAI API 超時秒數。 |

-----

## ⚖️ 八、檢索裁判 (Assessor) 與匹配邏輯

### 🔸 匹配門檻與策略（Matching Thresholds & Strategy）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **LI\_PG\_MIN\_SHOULD\_STRICT** | `1` | 資訊豐富查詢（≥2 組件）的最低命中數（高精準）。 |
| **LI\_PG\_MIN\_SHOULD\_LOOSE** | `1` | 資訊稀疏查詢（1 組件）的最低命中數（高召回）。 |
| **LI\_PG\_REQUIRE\_ENTITY** | `1` | 是否強制需命中 head 或 tail 任一實體：`1`=是、`0`=否。 |
| **LI\_PG\_EVENT\_PIVOT** | `1` | tail 判定為「事件」時，允許 (tail ∧ relation) 快速通過 [cite: 16]。 |
| **LI\_PG\_ALLOW\_INVERT** | `1` | 允許倒置三元組（H↔T）進行匹配。 |
| **LI\_PG\_INVERT\_STRICT** | `0` | 倒置匹配不比對 props，僅比對 h/r/t 名稱：`1`=開啟、`0`=關閉。 |

### 🔸 屬性 (Properties) 使用

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **LI\_PG\_HEAD\_USE\_PROPS** | `1` | head 允許用 props 輔助命中（如別名/職稱） [cite: 17]。 |
| **LI\_PG\_TAIL\_USE\_PROPS** | `0` | tail 允許用 props 輔助命中：`1`=開啟、`0`=關閉 [cite: 17]。 |
| **LI\_PG\_FORWARD\_USE\_PROPS** | `1` | 允許正向查詢使用 props。 |

### 🔸 顯名保底（Organization Seeds）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **ENABLE\_ORG\_SEEDS** | `1` | 開啟顯名保底（機構名種子） [cite: 19]。 |
| **MAX\_ORG\_SEEDS** | `6` | 最多使用幾個機構名種子 [cite: 19]。 |
| **ORG\_SEED\_ALLOW\_INVERT** | `0` | Org-seed 查詢是否允許倒置。 |
| **ORG\_SEED\_ONEPASS** | `1` | Org-seed 單輪模式（`1`=啟用）。 |
| **ORG\_SEED\_MODE** | `UNDIRECTED` | Org-seed 模式：`UNDIRECTED` (無向 OR) 或 `DIRECTED` (有向 AND)。 |
| **ORG\_SEED\_MIN\_SHOULD** | `1` | Org-seed 最低命中要求。 |
| **ORG\_SEED\_FIELD\_WEIGHTS** | `head:1.0,tail:1.0,relation:0.6` | 欄位加權。 |
| **ORG\_SEED\_USE\_REL** | `0` | Org-seed 是否使用 relation 欄位：`1`=開啟、`0`=關閉 [cite: 20]。 |

### 🔸 除錯（Debug）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **LI\_PG\_CLI\_META** | `1` | CLI 是否列印 debug meta [cite: 18]。 |

-----

## 🧬 九、檢索融合與重排 (Reranking & Fusion)

### 🔸 RRF (Reciprocal Rank Fusion)

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **RRF\_K** | `60` | RRF 融合演算法的 K 值。 |
| **RRF\_W\_PG** | `1.0` | 屬性圖譜（PG）檢索的權重。 |
| **RRF\_W\_VEC** | `1.0` | 向量（Vector）檢索的權重。 |
| **RRF\_W\_NEO4J** | `1.0` | Neo4j 檢索的權重。 |

### 🔸 Cross-Encoder 重排

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **CE\_MODEL** | `BAAI/bge-reranker-v2-m3` | Cross-Encoder 重排模型。 |
| **CE\_TOPN** | `120` | 送進重排的候選數。 |
| **CE\_BATCH\_SIZE** | `16` | 重排的批次大小。 |
| **CE\_MAX\_LENGTH** | `1024` | 重排模型最大序列長度。 |

### 🔸 分數融合（Score Fusion）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **RRF\_CE\_FUSE\_ALPHA** | `0.50` | 最終分數融合比例：`α * CE_prob + (1-α) * RRF_norm`。 |

### 🔸 最終報告門檻（Final Report Thresholds）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **CE\_MIN\_FINAL** | `0.40` | 最終輸出的主分數門檻。 |
| **CE\_MIN\_FINAL\_LO** | `0.45` | 關鍵詞很多時的低門檻。 |
| **CE\_MIN\_KWS** | `1` | 中段分數時至少命中幾個關鍵詞。 |
| **CE\_MID\_FINAL** | `0.50` | 中段分數門檻。 |
| **CE\_TOPK** | `120` | 最多保留幾條證據 [cite: 21]。 |
| **CE\_KWS\_MIN\_FREQ** | `1` | 主題詞 gate 關鍵詞最低頻率。 |
| **CE\_REQUIRE\_TOPIC** | `1` | 是否要求命中主題詞。 |
| **CE\_KEEP\_IF\_FINAL\_GE** | `0.60` | 若分數 ≥ 此值則保留（無視主題詞）。 |

-----

## ❓ 十、查詢生成 (Query Generation)

### 🔸 多問句 (Multi-Query)

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **QUERY\_LLM** | `gpt-4o-mini` | 用於生成多問句的 LLM。 |
| **CE\_NQ** | `10` | 要生成的問句數量。 |
| **QUERY\_PROMPT** | `(見 .env 範本)` | 生成問句的提示詞。可含 `{n}` 與 `{news}`。 |

### 🔸 查詢拆解 (Query Decomposition)

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **ENABLE\_QD** | `1` | 啟用查詢拆解（`1`=啟用）。 |
| **QD\_MODEL** | `gpt-4o-mini` | 查詢拆解使用的模型。 |
| **QD\_MIN\_SPLITS** | `3` | 最少拆解數。 |
| **QD\_MAX\_SPLITS** | `6` | 最多拆解數。 |
| **QD\_STYLE** | `factual-tight` | 拆解風格。 |
| **QD\_TEMPERATURE** | `0.2` | 拆解溫度。 |
| **QD\_TIMEOUT\_SEC** | `60` | 拆解 API 超時秒數。 |

-----

## 🧠 十一、Embeddings 設定

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **SENTENCE\_TRANSFORMERS\_MODEL** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 指定 Sentence-Transformers 模型。 |
| **ACCELERATE\_DISABLE\_DEVICE\_MAP** | `1` | 禁用 accelerate 的 device map。 |
| **TRANSFORMERS\_NO\_ACCELERATE** | `1` | 禁用 transformers 的 accelerate。 |

-----

## 🧾 十二、除錯與日誌設定（Debug & Logging）

| 變數名稱 | 範例值 | 說明 |
| --- | --- | --- |
| **VERIFIER\_DEBUG** | `1` | 除錯開關：`1`=啟用詳細紀錄、`0`=關閉。 |
| **VERIFIER\_DEBUG\_PATH** | `data/processed/verifier/debug/verifier_search_debug.log` | 除錯日誌輸出位置。 |

-----

## 🧭 十三、版本與維護狀態（Version & Maintenance Notes）

| 區塊 | 狀態 | 備註 |
| --- | --- | --- |
| 資料庫設定 | ✅ 穩定 | 支援 Neo4j + MongoDB|
| 模型設定 | ✅ 穩定 | 支援 GPT 模型  |
| RAG 檢索 | ⚙️ 可調 | 依需求微調閾值與候選參數|
| LangChain Agent | ✅ 穩定 | 核心檢索流程 |
| 圖譜檢索 | ⚙️ 可調 | LlamaIndex PG 檢索核心，包含 Auto-Hops|
| ReAct 代理 | ⚙️ 可調 | StateGraph 迴圈與節流控制 |
| 同義詞擴充 | 🧪 實驗性 | 支援靜態、LLM 生成與即時擴充 |
| Assessor 邏輯 | ⚙️ 可調 | 檢索匹配的核心規則 |
| Reranking 融合 | 🧪 實驗性 | 支援 RRF + Cross-Encoder 融合 |
| 查詢生成 | 🧪 實驗性 | 支援 Multi-Query 與 Query Decomposition |
| Debug Logging | ✅ 穩定 | 建議於測試環境啟用 |

-----

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
# 可切換資料庫：
# NEO4J_DATABASE=testing
# NEO4J_DATABASE=experiment

# === MongoDB (事實查核紀錄 / Job Tracking) ===
MONGODB_URI=mongodb://*****REDACTED*****@localhost:27017/?authSource=admin


# =======================================
# 🧠 MODEL & LLM CONFIGURATION
# =======================================

# === Local / Custom Endpoint ===
MODEL_CONFIG_ENDPOINT=http://127.0.0.1:1234/v1/chat/completions
MODEL_ID=google/gemma-3-4b

# === OpenAI API Keys ===
OPENAI_API_KEY=sk-*****REDACTED*****
GPT_API=${OPENAI_API_KEY}                       # 與部分模組相容（paths.py 會讀）

# === Model Roles ===
GPT_MODEL=gpt-5                             # 主回答 / 判斷模型（paths.py 會讀）
OPENAI_RESP_MODEL=gpt-5                    # dirt_removal 預設使用的聊天模型
OPENAI_CHAT_MODEL=gpt-4.1                   # dirt_removal 備用的聊天模型
DIRT_FILTER_PROMPT_PATH=src/qa/verifier/prompts/dirt_filter.md # dirt_removal 使用的外部提示詞路徑

EXTRACT_MODEL=gpt-4o-mini                       # 三元組抽取模型（extract.py）
EXPAND_GPT_MODEL=gpt-4o-mini                    # 查詢擴充詞用（search.py）
DEDUP_GPT_MODEL=gpt-4o-mini                     # 重複項過濾用（dedup.py）
DEDUP_USE_GPT=0                                 # Dedup 的 GPT 輔助：1=啟用, 0=關閉


# =======================================
# 🔍 RAG & KNOWLEDGE RETRIEVAL SETTINGS
# =======================================

# === 檢索門檻與數量（供 search.py / subq_engine.py 使用） ===
DUP_TH=0.8                                      # 語義去重臨界值（相似≥此值視為重覆）
SIM_TH=0.45                                     # cosine 初篩門檻
TOP_K=150                                       # 最終返回的最大候選數，視情況增加可讓去重後仍有料
LLM_ROUNDS=0                                    # 初始 LLM 三元組抽取輪數，如果啟用 USE_AGENT_RETRIEVAL，此變數可設為 0

# === 檢索限制 ===
RAG_CAND_FACTOR=6                               # 候選池放大倍率（越大越多樣，但速度較慢）

# === 檢索嚴格度（evidence 命中規則） ===
# 類比 minimum_should_match：至少 2 命中；一元查詢程式已 +1
EVIDENCE_MIN_HITS=1                             # 每句至少需命中幾個關鍵詞，可向下放寬，先求召回，髒污交給 Dirt Removal。
# 開啟 anchor（head 空→自動用 tail），抑制「只剩一元」濫抓
REQUIRE_ANCHOR_IN_EVIDENCE=1                    # 1=要求 anchor（head）命中, 0=可略過

# ============================
# 🧭 同名歧義過濾（用來要求「關聯實體交集」）
# ============================
DISAMBIG_REQUIRE_COENTITY=1                     # 要求證據句要有共同實體，降低人名/政黨詞的假陽性
DISAMBIG_MIN_COMMON=1                           # 最少共同實體數（預設 1）


# =======================================
# 🦙 LANGCHAIN SETTINGS
# =======================================

# 啟用由 LangGraph Agent 進行檢索（多輪智慧規劃）
USE_AGENT_RETRIEVAL=1                           # 1=啟用, 0=關閉

# =======================================
# 只用離線 PG、本地知識（圖譜檢索設定）
# =======================================
ENABLE_LI_PG=1
ENABLE_LI_ONLINE=0
ENABLE_VECTOR_FALLBACK=1                        # 向量備援（本地 PG / 線上 PG 都沒命中時啟用）
LI_PG_INDEX_JSON=data/processed/knowledge-graph/pg_index.json
# =======================================

#ENABLE_LI_PG=0, ENABLE_LI_ONLINE=0, ENABLE_VECTOR_FALLBACK=1      # 只用向量檢索
#ENABLE_LI_PG=0, ENABLE_LI_ONLINE=1, ENABLE_VECTOR_FALLBACK=0/1    # 只用線上檢索（有無向量備援）

LI_PG_HOPS=3                                    # 使用本地知識圖譜檢索的跳數
LI_ONLINE_HOPS=3                                # 使用線上知識圖譜檢索的跳數
LI_PG_TOPK=80                                   # 可放寬本地 PG 檢索候選數（避免早停），可依機器調整

# === 離線 PG 原始 CSV 與欄位 ===
LI_PG_RAW_CSV=data/raw/knowledge-graph/neo4j-kg-raw-graph.csv
LI_PG_DATE_FIELD=date
LI_PG_EVIDENCE_FIELD=evidence
LI_PG_MIN_EVI=1                                 # 最少證據數過濾（避免噪音）

# ============================
# 🕒 時間窗過濾
# ============================
FILTER_TIME_WINDOW=0                            # 開啟時間窗過濾（1=開, 0=關）
TIME_WINDOW_DAYS=0                              # 時間窗天數（0=不設限）

# =======================================
# ⚙️ 自動調整跳數設定
# =======================================
AUTO_HOPS_ENABLE=1                              # 開啟自動調整跳數（預設開）
AUTO_HOPS_START=2                               # 初始跳數（建議 2）
AUTO_HOPS_MAX=3                                 # 最高跳數（建議 3）
AUTO_HOPS_MIN_HITS=8                            # 命中數不足這個值就升一級（可依圖密度微調）
AUTO_HOPS_TOPK=50                               # 每次查詢的 top_k（避免一次拉太多）

# =======================================
# 🧾 LOGGING & DEBUGGING
# =======================================

VERIFIER_DEBUG=1                                # 啟用驗證器除錯日誌（1=啟用, 0=關閉）
VERIFIER_DEBUG_PATH=data/processed/verifier/debug/verifier_search_debug.log

# =======================================
# ♻️ ReAct 代理（StateGraph）循環門檻 / 節流
# =======================================

# —— 安全防爆（輸入/證據節流）——
MAX_AGENT_CHARS=12000                           # 進入 Agent 的新聞最大字元數（硬性截斷）
MAX_EVID_CHARS=300                              # 每條證據描述的最長字元（避免 tool payload 肥大）

# —— 迴圈收斂條件（達標/停損）——
AGENT_MAX_STEPS=8                               # 最多工具步數（達到就收斂）
AGENT_NO_NEW_PATIENCE=4                         # 連續幾步沒有新增命中就放行收斂
AGENT_RECURSION_FACTOR=3                        # 每輪約 3 個超步（agent→tools→accumulate），預設 3 即可。
AGENT_RECURSION_EXTRA=3                         # 保險係數，避免邊界卡死，預設 3。
AGENT_MIN_PER_TRIPLE=0                          # 每個 triple 至少要有幾條命中，如果設為0，則交由耐心/步數控制
AGENT_TOTAL_TARGET=0                            # 全局至少命中幾條證據才算「充足」，同上
AGENT_TOP_K_MAX=140                             # Agent 每次檢索的最大候選數

# ============================
# 🧠 Embeddings / ST 相關
# ============================
# ✅ 指定正式的 Sentence-Transformers 模型（避免臨時 mean pooling）
# （若已做本地 ST 轉檔，改填轉檔後的目錄；否則先用跨語多任務穩定款）
SENTENCE_TRANSFORMERS_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 避免 accelerate 把權重掛到 meta / 非預期裝置（與程式中一致）
ACCELERATE_DISABLE_DEVICE_MAP=1
TRANSFORMERS_NO_ACCELERATE=1

# ============================
# 🔤 查詢端同義規則（外部檔；可熱更新）
# ============================
# 不提供檔案時 → 不展開同義（保持原行為）
ALIAS_RULES_JSON=data/processed/knowledge-graph/alias_rules.json

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
ALIAS_WRITE_EVERY=1                             # 每批即寫；改 5 = 每 5 批再寫一次
OPENAI_TIMEOUT_SEC=120

# ============================
# ⚖️ 裁判（Assess）放寬參數
# ============================
# H/R/T 至少命中幾個才算過關（minimum_should_match）
# 檢索 minimum-should-match 與策略

LI_PG_MIN_SHOULD_STRICT=1   # 用於資訊豐富的查詢（>=2個組件），追求高精準度。
LI_PG_MIN_SHOULD_LOOSE=1    # 用於資訊稀疏的查詢（1個組件），確保高召回率。
LI_PG_REQUIRE_ENTITY=1      # 是否強制需命中 head 或 tail 任一實體（1=是, 0=否）

# --- 事件樞紐與倒置 ---
LI_PG_EVENT_PIVOT=1                             # tail 判定為「事件」時，允許 (tail ∧ relation) 快速通過
LI_PG_ALLOW_INVERT=1                            # 允許倒置三元組（H↔T）-> 方向顛倒（被動語態），允許倒置再判一次。
LI_PG_INVERT_STRICT=0                           # 啟用「倒置」匹配不吃 props，只比對 h/r/t 名稱，降低誤擊率；0=關、1=開啟不吃

# --- props 使用策略（ head/tail 細分；tail 預設關閉） ---
LI_PG_HEAD_USE_PROPS=1                          # head 允許用 props 幫忙命中（人名常有別名/職稱）
LI_PG_TAIL_USE_PROPS=0                          # tail 的吃 props 開關（關閉可避免事件描述 props 造成假陽性)，可視需求開啟；0=關閉、1=開啟
LI_PG_FORWARD_USE_PROPS=1                       # 允許正向吃 props（預設 1）
LI_PG_CLI_META=1                                # CLI 列印 debug meta（1=開）

# --- 即時擴充（查詢時寫回 relation.hints） ---
ALIAS_LLM_RUNTIME_ENABLE=1                      # 1=開；0=關
ALIAS_RELOAD_SEC=8                              # alias_rules.json 熱更新間隔（秒）

# --- 顯名保底種子 (organization seed) ---
ENABLE_ORG_SEEDS=1                              # 開關，預設開啟
MAX_ORG_SEEDS=6                                 # 最多幾個機構名

# === Org-seed 單輪＋無向 OR ===
ORG_SEED_ALLOW_INVERT=0    # 機構保底種子：預設不倒置（單輪無向 OR 已足夠）
ORG_SEED_ONEPASS=1         # Org-seed 單輪模式（避免 h→t 倒置再跑一次）；1=啟用, 0=關
ORG_SEED_MODE=UNDIRECTED   # Org-seed 模式：UNDIRECTED（無向 OR）、DIRECTED（有向 AND）
ORG_SEED_MIN_SHOULD=1      # Org-seed 最低命中要求（無向 OR，預設 1）
ORG_SEED_FIELD_WEIGHTS=head:1.0,tail:1.0,relation:0.6 # 欄位加權
ORG_SEED_USE_REL=0         # Org-seed 的 relation 欄位命中開關（關閉可避免 evidence/props 關鍵字造成誤收；0=關閉、1=開啟）

# ---------- RRF（多路檢索融合） ----------
RRF_K=60
RRF_W_PG=1.0
RRF_W_VEC=1.0
RRF_W_NEO4J=1.0

# ---------- Cross-Encoder（重排器） ----------
# 可選：BAAI/bge-reranker-v2-m3（多語、精度高）或 cross-encoder/ms-marco-MiniLM-L-6-v2（輕量）
CE_MODEL=BAAI/bge-reranker-v2-m3
CE_TOPN=120              # 送進重排的候選數（常見 80~200）
CE_BATCH_SIZE=16
CE_MAX_LENGTH=1024       # 官方微調長度 1024，推論建議 1024（雖支援更長）。

# ---------- 分數融合 ----------
# final_score = α * CE_prob + (1-α) * RRF_norm
RRF_CE_FUSE_ALPHA=0.50

# ---------- emit_final_report 輸出門檻 ----------
CE_MIN_FINAL=0.40        # 主分數門檻（放嚴一點）
CE_MIN_FINAL_LO=0.45     # 關鍵詞很多時的低門檻
CE_MIN_KWS=1             # 中段分數時至少命中幾個關鍵詞
CE_MID_FINAL=0.50       # 中段分數門檻
CE_TOPK=120              # 最多保留幾條證據

# —— emit_final_report 的主題詞 gate 與門檻 ——
CE_KWS_MIN_FREQ=1          # 搭配單段新聞摘要
CE_REQUIRE_TOPIC=1
CE_KEEP_IF_FINAL_GE=0.60

# —— LLM 產生多問句（Multi-Query） ——
QUERY_LLM=gpt-4o-mini
CE_NQ=10

# 自訂提示詞（可空；可含 {n} 與 {news}）
QUERY_PROMPT=請將以下文本拆成 {n} 句越具體越好的查核用問句，分別聚焦：主詞、行為、時間、資金流向、來源真實性與證據鏈、涉入單位或個人。每行一問句，不要編號：\n\n{news}\n

# 啟用 QD 與模型設定
ENABLE_QD=1
QD_MODEL=gpt-4o-mini
QD_MIN_SPLITS=3
QD_MAX_SPLITS=6
QD_STYLE=factual-tight
QD_TEMPERATURE=0.2
QD_TIMEOUT_SEC=60
```

> **注意事項：**
>
>   * 請務必不要把包含真實密鑰或密碼的 `.env` 提交到公開的 Git 倉庫；把本檔（`env_example.md`）作為示例即可。
>   * 若要在本地測試，請在專案根目錄建立 `.env` 並貼入真實值。參考：`.env.local`, `.env.production` 命名慣例。
>   * 若使用 Docker 部署，可在 `docker-compose.yml` 或 Dockerfile 中讀取環境變數，或把敏感值放到雲端密鑰管理服務（Hashicorp Vault / GCP Secret Manager / AWS Secrets Manager）。

-----