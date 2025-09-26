# 數據實驗解釋文件

## 實驗目的
本實驗旨在比較 **三種不同資訊回應來源** 的差異，包含：
1. **本系統（LLM + 知識圖譜檢索）**：使用語意檢索技術，結合知識圖譜後，以 LLM 進行事實驗證。
2. **純 LLM 回答（ChatGPT）**：僅依照 ChatGPT 的生成能力進行回答，未額外接入事實檢索。
3. **事實查核中心官方闢謠**：具有權威性的人工查核單位，作為真實可靠的對照組。

透過比較三者在 **正確性、完整性、細節度** 的表現，驗證本系統在新聞查核應用中的價值。

---

## 實驗資料來源
選用 **臺灣事實查核中心（TFC）** 已發布的 10 篇闢謠報告，涵蓋國防、外交、政策、社會議題面向。每則均為近期在網路社群廣泛流傳的 **錯誤或不實消息**。

### 測試文章與原始來源
1. **臺灣興達發電廠爆炸原因曝光，台空軍F16夜間對抗實彈射擊演習誤射所致！**  
   🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/false-video-hsingda-power-plant-explosion-f16-misfire-claim/)

2. **國內外9條海底電纜同時斷線，台灣險淪網路孤島。**  
   🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/taiwan-submarine-cables-operational-three-southeast-asia-routes-under-repair/)

3. **總統賴清德參拜日本靖國神社。**  
   🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/lai-ching-te-video-yasukuni-fake-taoyuan-temple/)

4. **軍人抗命將啟動軍審，被訓話時手握拳頭、偷吃早餐可判7年以上。**  
   🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/politician-misinterprets-law-soldiers-breakfast-not-mutiny/)

5. **美國衆議院今天(2025/8/15)首次向全世界正式宣布：「台灣是主權獨立的國家」，美國總統川普在美中關稅貿易談判中，向中國提出要求，必須承認台灣的主權與獨立地位。**  
   🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/us-house-did-not-declare-taiwan-sovereign-state/)

6. **總統令公告：只要憲法法庭通過，立刻全民普發1萬元。**  
   🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/constitutional-court-does-not-trigger-cash-payout/)

7. **美國前總統川普在接受媒體簡短採訪時，突然透露即將宣布一項「與貿易無關的重大消息」，同時美國國務院也下架長期維持台美非官方關係的網頁，引發外界揣測台美關係可能出現歷史性突破，所謂的「重大消息」即為「台美正式建交」，並稱「中共暴怒，連夜召回駐美大使」等。美國川普透露美台即將建交，中共連夜召回大使**  
   🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/fake-news-us-taiwan-establish-diplomatic-ties/)

8. **移民署要求14萬陸配需限期補繳放棄原籍證明書，否則將被驅逐。**  
   🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/china-media-social-misrepresent-spouses-certificate-issue/)

9. **川普已經啟動推動台灣加入聯合國的計劃**  
   🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/trump-taiwan-un-fake-news/)

10. **公投方式不同於上次的反罷免投票，公投通過條件為投票人數大於全國公民人數的25％，且同意大於不同意。**  
    🔗 [連結](https://tfc-taiwan.org.tw/fact-check-reports/referendum-recall-threshold-25-percent-rule-taiwan/)

---

## 實驗方法
0. **主題定義（來自 TFC）**：以臺灣事實查核中心之 10 篇闢謠為「權威主題」，作為查詢、比對與評測單位標準。  

### 實驗流程與程式碼路徑
1. **提取謠言三元組**  
   程式：`experiment/news_retrieval/10_sentence_extraction.py`  
   功能：將 10 個謠言轉換為結構化的三元組關係，作為檢索與建圖依據。

2. **檢索原始新聞庫**  
   程式：`experiment/news_retrieval/search_10_related.py`  
   功能：利用 (1) 的三元組，從 MongoDB 中取出相關新聞原文，建立知識來源基底。

3. **知識抽取與建圖**  
   程式：`experiment/kg_neo4j/pipeline.py`  
   功能：將 (2) 的新聞原文進行知識抽取，生成三元組與屬性，並寫入 Neo4j 知識圖譜。

4. **圖譜向量化**  
   程式：`experiment/preliminary_work/pipeline.py`  
   功能：將知識圖譜中節點與關係取出，進行向量化，以利後續檢索與比對。

5. **生成查核報告**  
   程式：`experiment/answerer/pipeline.py`  
   執行方式：  
   ```bash
   python -m experiment.answerer.pipeline <1~10.txt>
   ```
   > LLM + KG : 以 GPT-5 結合向量檢索與 Neo4j 子圖生成查核報告。

6. **結果比對**
   三組輸出——**TFC 官方闢謠**、**LLM+KG**、**純 LLM**——逐主題進行對齊與評測。
   > 註：以 GPT-5 直接生成查核報告，不接入**知識圖譜**並**關閉網路檢索功能**，作為對照組。

7. **重現性控制**
   固定隨機種子、記錄模型版本與提示詞，保存所有中間產物（檢索結果、三元組、子圖快照、最終報告）。

---

## 比較設計與評分標準

* **正確性（Accuracy）**：能否明確判定主張之真偽並與 TFC 結論一致。
* **完整性（Completeness）**：是否涵蓋關鍵脈絡（時間、地點、涉事對象、事件機制）與反證點。
* **細節度（Granularity）**：是否提供具體事證（原文摘錄、時間點、機構名稱）與可核對的引用。
* **可溯性（Traceability）**：引用是否對應到可公開核驗的來源。

> **計分規則（0–10 分）**：待討論。
> **總分**：`Accuracy X% + Completeness X% + Granularity X% + Traceability X%`（待調整）。

---

## 預期結果

* **LLM + 知識圖譜**：在正確性與細節度上顯著優於純 LLM，與 TFC 結論的一致率最高；可提供清楚的來源鏈接與圖譜依據。
* **純 LLM**：能給出方向性判定，但易出現時序誤差與來源不確定，對複雜事件的機制說明較弱，甚至出現幻覺。
* **TFC 官方闢謠**：作為黃金標準，提供最權威之結論與脈絡校準。

---

## 後續應用

* **檢索優化**：調整 Top-k、重排序策略與門檻，降低噪音並提升命中關鍵證據的比率。
* **圖譜覆蓋**：針對錯判或弱判主題補齊節點/關係類型與同名異形規則。