# FactGraph｜LangChain Agent「知識檢索」技術白皮書

> 這份文件完整說明本專案的 LangChain × LangGraph 檢索 Agent 如何把「原始新聞 → 可讀比對證據行」串成一條龍。包含：資料欄位、工具鏈、打分與正規化、多跳檢索、收斂策略、環境變數與除錯建議。

---

## 0. 整體鳥瞰（Architecture）

* **輸入**：一段新聞原文。
* **抽取**：先抽出一批 `triples = [{head, relation, tail}, ...]`。
* **Agent（LangGraph）循環**：對每個 triple 依序執行檢索工具（PG/Neo4j/Vector），每輪合併去重並累積可讀的 **[比對] 行**。
* **收斂**：以「步數上限／耐心值」收斂（無硬性下限），最後輸出：

  * `[原始文本]` 原文
  * `[比對知識]` 編號的證據行（每行**一律保證**出現「透過關係【…】」）
* **工具與框架關鍵**：

  * LangGraph 的 `StateGraph` 與 `ToolNode` 專職處理模型呼叫工具、把工具回傳寫回狀態（state）。([docs.langchain.com][1])
  * LangChain `ChatOpenAI.bind_tools(tool_choice="any")` 綁定多工具、交由模型規劃選用。([python.langchain.com][2])

---

## 1. 知識圖譜資料模型（CSV / Neo4j 對齊）

**本地圖譜欄位**：
```json
{"head", "relation", "tail", "head_props", "rel_props", "tail_props"}
```
其中三個 `*_props` 皆為 JSON
```json
{"type":"人物","name":"黃國昌"}`）
```

**查詢示例**：

```json
{"head":"黃國昌","relation":"涉嫌","tail":"狗仔團隊"}
```

**輸出比對行（保證樣式）**：

```
[比對] 黃國昌（政治身份:立委） 透過關係【涉嫌】與 狗仔團隊（type:組織） 建立連結，說明：事件日期 2025-09-26；來源：鏡報；別名：跟監團隊/偷拍集團；期間：>=7天。
```

為了避免「關係名空白」，程式會多路嘗試 (`tri.relation` → `rel_props.relation/name/label/type/...`) 並做**關係字正規化**（括號拆分與白名單優先）。

---

## 2. 檢索工具鏈（三路並進，可開關）

1. **PG（CSV → Property Graph 快取）**

   * 以本地 JSON 索引（PG Index）做圖式檢索，支援多跳（`hops`），時間新→舊排序。
   * 能自動處理**被動語態**：同時查正向與反向（交換 head/tail），回收更合理者。
   * 適用：離線可用、命中密度高、延遲低。

2. **Neo4j（線上圖庫，可多跳）**

   * 用 Cypher 的**可變長徑**模式 `[*1..N]` 做多跳展開（官方稱 variable-length patterns）。([Graph Database & Analytics][3])
   * 以 `rel_props.date` 由新至舊排序，`rel_props.evidence` 做句級過濾（最少字元）。
   * 底層依 LlamaIndex 的 **Neo4jPropertyGraphStore** 封裝驅動。([developers.llamaindex.ai][4])

3. **Vector（向量備援）**

   * 把查詢 triple 壓成一句嵌入，與 KG 行向量做 cosine 過濾（門檻 `SIM_TH`），再以證據句 `evidence` 的關鍵詞命中數做次級篩選。
   * 適用：字面不同但語義近似（例如 `涉嫌↔被指控`、`狗仔團隊↔跟監團隊`）補漏。

> Agent 會把上述工具輸出的 `{"lines":[...]} / [比對] ...` 統一收斂與去重。

---

## 3. 召回 → 正規化 → 打分（六欄精配）

### 3.1 文字召回（六欄同查，先撈候選）

* `head / relation / tail / head_props / rel_props / tail_props` 六欄做子字串與語彙比對（`props` 會**展平**後再比），**`relation` 權重最高**，其次 `head/tail`，再來各 props。
* 語彙來源：

  * 三元組原詞
  * 別名／同義（來源：外部關係詞庫、`props.aka/alias` 或 GPT 擴充，皆可注入）

### 3.2 候選稀少 → 別名與同義擴充（再跑一次 3.1）

* 關係簇例：`涉嫌 → 被指控、涉入、被質疑、被爆料`
* 實體別名例：`狗仔團隊 → 跟監團隊、偷拍集團、反蒐證團隊(對立說法)`

### 3.3 再不夠 → 向量召回

* 把「`head relation tail`」拼句嵌入，cosine 高者入列；再以 `evidence` 句內的**命中關鍵詞數**與**必要錨點**（如 head 名稱）過濾，避免語義漂移。

### 3.4 打分與方向修正

1. **六欄命中分**：`relation` > `head/tail` > 三個 props；部分命中也給分，**同義視為命中**。
2. **語義分**：`cosine ≥ SIM_TH` 視為相近，越高加權越多。
3. **規則加減分**：

   * `rel_props.date` 能貼近新聞日期／期間 → 加分
   * 若句式被動（如「狗仔團隊 指控 黃國昌」）→ 嘗試**反轉 (h,t)**；反轉後更合理即採用並小加分
   * 命中多出現在不相干 props → 小扣分

### 3.5 排序、去重、同義合併（Hop-1 收斂）

* 相同 `(h,r,t)` 多列 → **合併 props**（日期取較明確；來源可保留多筆）
* 關係同義簇（涉嫌/被指控/涉入…）視為一組，取最高分為代表，其餘列為**佐證**
* 以總分排序；**同分則日期較新優先**

---

## 4. 多跳檢索（Hop-2/3…）

當 Hop-1 證據不足或為 0 時才進入多跳：

* 以 `h` 或 `t` 當**支點**擴散，且**受 `relation` 同義簇約束**，避免亂跳。

  * 例 A（以 `h` 擴）：`("黃國昌", r同義之一, "*")` 找到尾端 X，再檢查 X 是否等於/包含/別名為「狗仔團隊」
  * 例 B（以 `t` 擴）：`("*", r同義之一, "狗仔團隊")` 找到頭端 Y，再檢查 Y 是否等於/指向「黃國昌」
* Neo4j 端以 **可變長徑** `[*1..N]` 展開路徑，這是 Cypher 的標準能力。([Graph Database & Analytics][3])

---

## 5. ReAct 反覆決策與收斂（LangGraph）

* Agent 的系統提示會要求**只處理當前 triple**、**僅呼叫可用工具**，並用 `merge_and_dedup` 合併去重。
* 工具掛載採 `ChatOpenAI.bind_tools(TOOLS, tool_choice="any")`，讓模型對齊 OpenAI 的 function-calling 介面、主動選用工具。([python.langchain.com][2])
* 圖形流程由 `StateGraph` 建置，「模型節點 → ToolNode → 累積節點 → 判斷是否繼續 / 結束」的循環。([docs.langchain.com][1])
* **收斂條件**：

  * 步數達上限 `AGENT_MAX_STEPS`
  * 連續無新增達 `AGENT_NO_NEW_PATIENCE`
  * 或所有 triple 已處理完
  * （可選）累積比對行超過 `AGENT_TOP_K_MAX` 時提早結束

> `ToolNode` 會把工具輸出寫回訊息狀態（`ToolMessage`），之後在累積節點被解析並轉為 `[比對]` 行。([langchain-ai.github.io][5])

---

## 6. 輸出格式保證（「透過關係【】」）

* 每筆輸出皆為**完整可讀句**，模板如下：

  ```
  [比對] {head（可含type/屬性）} 透過關係與 {tail（可含type/屬性）} 建立連結，說明：{evidence}；事件時間：{date}。
  ```
* 關係名缺失時會多路回退（`tri.relation → rel_props.relation/name/label/type...`），並套用**正規化**（移除括號噪音、白名單優先），避免出現空的 `【】`。

---

## 7. 主要環境變數（.env 摘要）

| 變數                                         | 說明                                                                                 |
| ------------------------------------------ | ---------------------------------------------------------------------------------- |
| `ENABLE_LI_PG`                             | 開/關 CSV PG 檢索（預設 1）                                                                |
| `LI_PG_INDEX_JSON`                         | PG JSON 索引路徑（預設 `data/processed/knowledge-graph/pg_index.json`）                    |
| `LI_PG_TOPK`, `LI_PG_HOPS`                 | 覆蓋 PG 工具的 top-k/hops（保留入參作為下限）                                                     |
| `ENABLE_LI_ONLINE`                         | 開/關 Neo4j 線上檢索                                                                     |
| `NEO4J_URI/USER/PASSWORD/DATABASE`         | Neo4j 連線設定（供 LlamaIndex Neo4jPropertyGraphStore 使用）([developers.llamaindex.ai][4]) |
| `ENABLE_VECTOR_FALLBACK`                   | 開/關向量備援（預設 1）                                                                      |
| `SIM_TH`, `TOP_K`, `RAG_CAND_FACTOR`       | 向量相似度門檻、上限與候選擴張因子（召回層）                                                             |
| `AGENT_MAX_STEPS`, `AGENT_NO_NEW_PATIENCE` | 收斂控制                                                                               |
| `AGENT_TOP_K_MAX`                          | 最終輸出上限（避免爆 token）                                                                  |
| `VERIFIER_DEBUG`, `VERIFIER_DEBUG_PATH`    | 除錯開關與路徑（建議開）                                                                       |

---

## 8. 典型執行流程（逐步）

1. **讀入原文** → 截斷到 `MAX_AGENT_CHARS`（預防極長輸入）
2. **抽取 triple**（一次性節點）：`[{head, relation, tail}, ...]`
3. **Agent 循環（每次處理一個 triple）**
   a. LLM 根據**工具可用性**選擇 PG／Neo4j／Vector
   b. 工具回傳 `{"lines":[...]} / "[比對] ..."`
   c. `merge_and_dedup` 合併去重 → 累積器更新統計
   d. 達到耐心/步數上限或 triples 全數處理完 → Finalize
4. **Finalize**：輸出 `[原始文本]` + `[比對知識]`（自動編號）

---

## 9. 例外保底（無命中時）

* 若 Agent 最終未留下任何 `[比對]` 行，系統會**保底**：以前 1–5 個 triples 做一次**向量檢索**，取若干條能過濾的候選，照樣輸出（避免「空檔」）。

---

## 10. CLI 範例

```bash
# 處理單一新聞（自動輸出中間檔）
python -m src.qa.verifier.agent_langchain s_1017_mirro.txt

# 偵錯 PG：直接丟查詢（會印出 {"lines":[...]}）
python -m src.qa.verifier.agent_langchain pg '{"head":"黃國昌","relation":"涉嫌","tail":"狗仔團隊"}' 80

# 產生/更新 alias 規則（委派 retriever）
python -m src.qa.verifier.agent_langchain alias
```

---

## 11. Troubleshooting（常見症狀 × 調參建議）

* **PG 檢索不可用**
  檢查：`ENABLE_LI_PG=1`、`LI_PG_INDEX_JSON` 路徑是否存在；或先透過 retriever 建索引。
  若改用 Neo4j 線上，請確認憑證與 DB 名稱，並確保可變長徑語法 `[*1..N]` 未被防火牆／權限阻擋。([Graph Database & Analytics][6])
* **同義／別名吃不夠**
  追加關係白名單或 alias 檔；必要時開啟 GPT 擴充查詢詞（有 LRU 快取避免過度調用）。
* **命中很多，但證據片斷雜訊多**
  調高 `SIM_TH` 與 `EVIDENCE_MIN_HITS`，並要求 evidence 句必含錨點（head 名稱）。
* **輸出沒有「透過關係【】」**
  代表原始資料或 `rel_props` 欄位過於疏漏；請確認 `relation` 來源與正規化白名單是否涵蓋。

---

## 12. 可擴充點（Roadmap）

* **加權學習化**：將六欄權重、日期加權與反轉加分以學習方式（如 PairRank）微調。
* **多來源一致性檢查**：在合併階段，對同一 `(h,r,t)` 的不同來源做一致性標記（支持度、矛盾度）。
* **圖上社群／連通子圖約束**：Hop 擴散時加入社群偵測，限制在「同議題子圖」內搜尋。

---

### 參考（框架與語法）

* LangGraph：`StateGraph`、工具節點 `ToolNode` 用於代理流程與工具呼叫。([docs.langchain.com][1])
* LangChain：`ChatOpenAI.bind_tools(tool_choice="any")` 綁定與強制工具策略。([python.langchain.com][2])
* LlamaIndex × Neo4j：`Neo4jPropertyGraphStore` 與 Property Graph 範例。([developers.llamaindex.ai][4])
* Cypher 可變長徑語法（`[*1..N]`）實作多跳檢索。([Graph Database & Analytics][6])

---

## 付錄 A｜簡易流程圖

```md
整理查詢
   ↓
抽取三元組（一次性）
   ↓
[Agent 迴圈 for 每個 triple]
  ├─ LLM 規劃 → 選工具（PG/Neo4j/Vector）
  │     └─ 工具執行 → 回傳 {"lines":[...]} / "[比對] ..."
  ├─ merge_and_dedup 合併去重
  └─ 累積計數（步數、連續無新增）
       └─ 收斂判定（步數/耐心/完成）
   ↓
Finalize：輸出 [原始文本] + [比對知識] <順號羅列>
```

---

## 付錄 B｜六欄精配與方向反轉的範例

**輸入 triple**：`h="黃國昌", r="涉嫌", t="狗仔團隊"`
**PG 命中**：

```bash
row1: head="黃國昌" relation="涉嫌" tail="狗仔團隊"
      rel_props={"date":"2025-09-26","source":"鏡報"}
      → 三欄直擊，高分候選

row2: head="黃國昌" relation="被指控" tail="跟監團隊"
      rel_props={"aka":"狗仔團隊","period":">=7天"}
      → 關係近義、tail 近義，納入候選

row3: head="狗仔團隊" relation="指控" tail="黃國昌"
      → 被動語句（顛倒），嘗試反轉 (h,t) 後再評估
```

---

[1]: https://docs.langchain.com/oss/javascript/langgraph/quickstart?utm_source=chatgpt.com "Quickstart - Docs by LangChain"
[2]: https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html?utm_source=chatgpt.com "ChatOpenAI — 🦜🔗 LangChain documentation"
[3]: https://neo4j.com/docs/cypher-manual/current/patterns/variable-length-patterns/?utm_source=chatgpt.com "Variable-length patterns - Cypher Manual"
[4]: https://developers.llamaindex.ai/python/framework-api-reference/storage/graph_stores/neo4j/?utm_source=chatgpt.com "Neo4j"
[5]: https://langchain-ai.github.io/langgraphjs/how-tos/tool-calling/?utm_source=chatgpt.com "How to call tools using ToolNode"
[6]: https://neo4j.com/docs/cypher-manual/current/patterns/?utm_source=chatgpt.com "Patterns - Cypher Manual"
