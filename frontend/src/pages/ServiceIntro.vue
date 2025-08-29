<template>
  <!-- 裝飾圖案：維持原樣 -->
  <img src="/dog hend.png" class="dog-hand dog-hand-1" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-2" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-3" alt="dog hand" />

  <!-- Header：與首頁相同 -->
  <header class="header">
    <div class="header-left">
      <span class="header-icon" role="img" aria-label="camera">
        <img src="/camara icon.png" alt="camera icon" style="width:60px;height:60px;display:block;" />
      </span>
      <div class="header-text">
        <div class="header-title">芒狗偵探</div>
        <div class="header-subtitle">新聞查核，透明可信，快速回覆。</div>
      </div>
    </div>

    <div class="header-right">
      <button class="menu-btn" @click="toggleMenu" :aria-expanded="showMenu" aria-label="toggle menu">
        <span class="menu-bar"></span><span class="menu-bar"></span><span class="menu-bar"></span>
      </button>
      <nav class="menu-dropdown" :class="{ show: showMenu }">
        <router-link to="/">首頁</router-link>
        <router-link to="/service" class="active">服務介紹</router-link>
        <router-link to="/about">關於我們</router-link>
      </nav>
    </div>
  </header>

  <!-- 背景圖 -->
  <img src="/dog icon.png" class="bg-dog" alt="dog icon" />

  <!-- 內容 -->
  <div class="container">
    <!-- 橘字大標題 -->
    <h1 class="page-title"
        @mouseenter="titleHover = true"
        @mouseleave="titleHover = false"
        :style="{ transform: titleHover ? 'scale(1.01)' : 'scale(1)', transition: 'transform .15s ease' }">
      服務介紹
    </h1>

    <!-- 我們在做什麼 -->
    <section class="section">
      <h2>為什麼選擇芒狗偵探？</h2>
      <p>
        　　台灣新聞環境真假混雜、獵奇標題與斷章取義屢見不鮮。在使用「文案查詢」與「詢問模式」下，以
        <strong>Prompt engineering 運用 LLM ＋ 知識圖譜（Neo4j）＋ 向量檢索（CKIP-BERT）</strong>
        的三層設計，協助你判讀新聞是否可信，並提供詳細且精確且可回溯的引用與來源。
      </p>
      <p>  
        　　若選擇「實時 RAG」模式，則結合 <strong>RAG 向量庫</strong>，並搭配長、短句分類及關鍵字提取的 <strong>Agent 設計</strong>
        ，能以最快的速度的進行回覆，以達到事實的查核目標。 
      </p>
    </section>

    <!-- 三種模式 -->
    <section class="section">
      <h2>使用模式介紹</h2>
      <div class="cards">
        <article class="intro-card">
          <h3 style="text-align: center;">實時 RAG（最新）</h3>
          <p>直接檢索最新的資料向量庫進行比對，<strong>無需預先建圖譜</strong>即可迅速回覆。</p>
            <li>適用：突發議題的快速核對</li>
            <li>輸入：新聞段落/全文 或 短問句</li>
            <li>特點：迅速解答、資料庫覆蓋新</li>
        </article>

        <article class="intro-card">
          <h3 style="text-align: center;">文案查詢（長文）</h3>
          <p>上傳或貼上完整新聞／長文，做全面比對與說明。</p>
            <li>適用：需要整體脈絡與逐條比對</li>
            <li>輸入：完整文案、新聞日期</li>
            <li>等待：約 5–10 分鐘</li>
        </article>

        <article class="intro-card">
          <h3 style="text-align: center;">詢問模式（短問）</h3>
          <p>以問句釐清單點事實，快速得到精簡有力的結論。</p>
            <li>適用：欲確認單一事實或關鍵點</li>
            <li>輸入：短問句、發問日期</li>
            <li>等待：約 1–5 分鐘</li>
        </article>
      </div>
    </section>

    <!-- 操作流程（強化為資料流 + 判斷規則摘要） -->
    <section class="section">
      <h2>如何運作</h2>
      <ol class="steps">
        <li>選擇模式：實時 RAG／文案查詢／詢問模式。</li>
        <li>填寫必要欄位（內容與日期），點擊「開始查核」。</li>
        <li>系統建立查核任務並提供 <strong>臨時網址</strong>。</li>
        <li>可於原頁面進行等候，或使用回訪連結查看最終查核結果。</li>
      </ol>
      <div class="diagram-panel">
        <button
          class="diagram-toggle"
          @click="toggleFlow"
          :aria-expanded="showFlow.toString()"
          aria-controls="flowchart-panel"
          type="button"
        >
          <span class="label">{{ showFlow ? '收起流程圖' : '展開流程圖' }}</span>
          <span class="chevron" :class="{ open: showFlow }">⌄</span>
        </button>

        <div id="flowchart-panel" class="diagram" v-show="showFlow">
          <img :src="flowChart" alt="系統操作流程圖" />
        </div>
      </div>
      <div class="chips">
        <span class="chip ok">✅ 一致</span>
        <span class="chip warn">⚠️ 缺漏</span>
        <span class="chip bad">❌ 矛盾</span>
        <span class="chip score">評分：1–10 分</span>
      </div>
    </section>

    <!-- 評分與三分類規則 -->
    <section class="section">
      <h2>三分類與評分怎麼看？</h2>
      <ul class="bullets">
        <li><strong>一致：</strong>新聞敘述與檢索知識相符，引用可互相佐證。</li>
        <li><strong>缺漏：</strong>文本提及但目前知識庫查無充分佐證；不等於「假」。</li>
        <li><strong>矛盾：</strong>與可信來源明確相衝突；需特別留意來源時間與語境。</li>
        <li><strong>1–10 分綜合評級：</strong>綜合一致/缺漏/矛盾密度、來源數量與覆蓋度、語意匹配品質進行給分。</li>
      </ul>
    </section>

    <!-- 資料來源與更新頻率 -->
    <section class="section">
      <h2>佐證資料來源與更新頻率</h2>
      <ul class="bullets">
        <li><strong>來源：</strong>公視、中央社、內政部、行政院之爬蟲資料；向量化並去重。</li>
        <li><strong>更新：</strong>定時進行批次補抓；實時 RAG 可直接對應最新資料。</li>
        <li><strong>引用呈現：</strong>結果區列出對照知識與其來源，便於回查。</li>
      </ul>
    </section>

    <!-- 隱私與倫理聲明 -->
    <section class="section">
      <h2>隱私與倫理聲明</h2>
      <ul class="bullets">
        <li>輸入內容僅用於本次查核；不做個人輪廓建模或廣告追蹤。</li>
        <li>若為機密或敏感個資，建議去識別化後再提交。</li>
        <li>引用皆來自可追溯來源；避免將模型臆測當作事實。</li>
      </ul>
    </section>

    <!-- 限制與風險 -->
    <section class="section">
      <h2>限制與風險</h2>
      <ul class="bullets">
        <li>時間敏感：最新訊息更新前可能出現「缺漏」。</li>
        <li>語境差異：不同版本的報導，可能造成看似「矛盾」；須以來源時間與上下文為準。</li>
        <li>來源偏誤：因多來源交叉引用，可能會有偏誤影響（已使用中立媒體盡量避免）。</li>
      </ul>
    </section>

    <!-- 未來規劃（文件中的展望 → 服務承諾版） -->
    <section class="section">
      <h2>未來規劃</h2>
      <ul class="bullets">
        <li>持續升級語言模型與檢索策略，改善速度與準確度。</li>
        <li>擴大知識庫涵蓋（國際、體育、金融等），強化跨域查核力。</li>
        <li>探索多模態輸入（影像/影片/社群貼文），逐步支援更豐富的查核素材。</li>
        <li>服務場景擴充：提供公部門與媒體審核流程之 API 介接。</li>
      </ul>
    </section>

    <!-- 適用對象與場景 -->
    <section class="section">
      <h2>適用對象與場景</h2>
      <ul class="bullets">
        <li>一般讀者：遇到可疑標題與轉傳訊息，想快速核對真偽。</li>
        <li>媒體工作者：撰稿前後的交叉查核與背景補充。</li>
        <li>研究者／公部門：需要可追溯、可引用的查核依據。</li>
      </ul>
    </section>

    <!-- 常見問題 -->
    <section class="section">
      <h2>常見問題</h2>

      <details><summary>為何顯示「缺漏」？</summary>
        <p>當資料庫檢索不到足夠證據，為避免過度推論會標示缺漏；建議嘗試更換關鍵詞或補充時間線。</p>
      </details>

      <details><summary>為什麼沒有標示「矛盾」？</summary>
        <p>矛盾需要系統檢索到明確衝突的事實才會顯示。如果資料數量有限或相關資訊不完整，結果可能只會顯示「缺漏」。</p>
      </details>

      <details><summary>可以引用結果嗎？</summary>
        <p>可以，但請務必連同系統提供的來源引用一併標示，避免斷章取義。</p>
      </details>

      <details><summary>輸入問題應該怎麼寫才最適合？</summary>
        <p>建議包含關鍵實體（人物、地點、組織）、事件動詞與時間點，能有效提升匹配品質。</p>
      </details>

      <details><summary>支援哪些語言？</summary>
        <p>目前主要支援繁體中文新聞語料，其他語言的查核效果有限。</p>
      </details>

      <details><summary>資料更新頻率是多久？</summary>
        <p>新聞資料庫會每日自動更新，保證收錄最新的新聞事件。</p>
      </details>

      <details><summary>為什麼和其他媒體報導有差異？</summary>
        <p>系統僅比對收錄的媒體來源，若資料庫沒有收錄該媒體的特定報導，可能會出現落差。</p>
      </details>

      <details><summary>可信度分數怎麼解讀？</summary>
        <p>分數 1～10 分，越高代表新聞與資料庫比對結果越一致。缺漏多會扣分，但真正矛盾才會大幅降低分數。</p>
      </details>

      <details><summary>為什麼同一篇新聞，有時分數不同？</summary>
        <p>分數會受到檢索結果、新聞撰寫方式、關鍵詞差異影響。建議嘗試更換提問方式以獲得更穩定的結果。</p>
      </details>

      <details><summary>可以查到幾年前的新聞？</summary>
        <p>目前收錄的資料來源多為近期新聞，時間範圍取決於各媒體開放的存檔。較舊的新聞可能無法檢索到。</p>
      </details>

      <details><summary>系統會不會有偏差或錯誤？</summary>
        <p>系統依賴新聞資料庫與檢索模型，不會自行創造事實，但仍可能因資料缺漏或表述差異產生偏差。</p>
      </details>

      <details><summary>使用這個系統需要付費嗎？</summary>
        <p>目前僅供研究與展示用途，不會進行收費。</p>
      </details>

      <details><summary>查詢時我的問題會被記錄嗎？</summary>
        <p>系統僅會暫時儲存查詢結果以便檢索，不會蒐集個人隱私資訊。</p>
      </details>

      <details><summary>如果我要檢查的新聞不在資料庫怎麼辦？</summary>
        <p>系統只能比對已收錄的來源。若資料庫缺漏，結果會標註「缺漏」，建議等待資料更新或改用其他查核方式。</p>
      </details>

      <details><summary>這套系統和人工查核有什麼不同？</summary>
        <p>系統能快速提供自動化比對，但不具備人工查核的背景知識與深度判斷，因此建議搭配人工查核使用。</p>
      </details>

      <details><summary>為什麼不同新聞會出現大量「一致」，卻很少「矛盾」？</summary>
        <p>因為「一致」與「缺漏」較容易被檢索出來，而「矛盾」需要資料中存在直接對立的描述，相對少見。</p>
      </details>

      <details><summary>可以輸入圖片或 PDF 來查核嗎？</summary>
        <p>目前僅支援文字輸入，圖片或 PDF 需要先轉換成文字才能進行檢索。</p>
      </details>

      <details><summary>是否能一次查多篇新聞？</summary>
        <p>目前建議一次輸入一篇新聞，以避免檢索混淆；未來版本可能會支援批次查核。</p>
      </details>

      <details><summary>系統會判斷新聞真假嗎？</summary>
        <p>不會直接判定「真或假」，而是提供「一致、缺漏、矛盾」的比對結果，讓使用者自行判斷可信度。</p>
      </details>
    </section>


    <!-- CTA -->
    <div class="cta-row">
      <router-link to="/" class="primary-btn">回到首頁，開始查核</router-link>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import flowChart from '../assets/flow-chart.png' 

const showMenu = ref(false)
const titleHover = ref(false)
const showFlow = ref(false)

const toggleFlow = () => { showFlow.value = !showFlow.value }

function toggleMenu () {
  showMenu.value = !showMenu.value
}

// 點擊頁面其他處時關閉選單
function handleClickOutside (e) {
  const menu = document.querySelector('.menu-dropdown')
  const btn = document.querySelector('.menu-btn')
  if (!menu || !btn) return
  if (!menu.contains(e.target) && !btn.contains(e.target)) {
    showMenu.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
/* 保留你的全站樣式，由外部 service.css 控制主要版型 */
@import '../assets/service.css';

/* 僅補上少量區塊距離與輔助樣式，避免覆蓋你原有主題色與字體設定 */
.container { padding: 24px; }
.page-title {
  font-size: clamp(24px, 4.2vw, 56px); /* 手機→桌機自動放大 */
  line-height: 1.1;
  margin-bottom: 12px;
  color: #f39b2e;
  letter-spacing: .5px;
}
.section { margin: 20px 0; }
.bullets { margin-left: 1.25rem; }

.cards { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.card { background: #fff; border-radius: 16px; padding: 14px; box-shadow: 0 6px 18px rgba(0,0,0,.06); }

/* 卡片本身別被內容撐出邊界 */
.intro-card {
  box-sizing: border-box;
  width: 100%;
  padding: 16px;
  border-radius: 16px;
  background: #fff;
}

/* 列表縮排用 padding，不要靠 marker 掛在卡片外 */
.intro-card ul {
  margin: 0;
  padding-left: 1.2rem;
  list-style: disc;
  list-style-position: outside; /* 或 inside 視覺看你喜歡 */
}

/* 核心：讓中文也能在任何點換行，URL/RAG/英文長詞不爆框 */
.intro-card li {
  line-height: 1.6;
  overflow-wrap: anywhere;     /* 最高招：任何地方都可斷行 */
  word-break: break-word;      /* 相容舊版瀏覽器 */
  line-break: loose;           /* 針對 CJK 改善分行（支援就會生效） */
}

/* 手機再收一點字級與內距 */
@media (max-width: 480px) {
  .intro-card { padding: 14px; }
  .intro-card li { font-size: 0.95rem; }
}

/* 整頁保險，避免誤用絕對定位裝飾圖造成水平捲軸 */
html, body { overflow-x: hidden; }


.steps { margin-left: 1.25rem; }

.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.chip { padding: 4px 10px; border-radius: 999px; font-size: .85rem; background: #f7f7f7; }
.chip.ok { background: #eaf8ef; }
.chip.warn { background: #fff7e8; }
.chip.bad { background: #ffefef; }
.chip.score { background: #eef5ff; }
/* 讓整個按鈕容器水平置中 */
.cta-row {
  margin-top: 24px;
  display: flex;
  justify-content: center;   /* 水平置中 */
}

/* 讓文字在按鈕裡也置中，並維持漂亮的內距 */
.primary-btn {
  display: inline-flex;       /* 比 inline-block 更好置中 */
  align-items: center;
  justify-content: center;
  padding: .6rem 1rem;
  border-radius: 12px;
  background: #f39b2e;
  color: #fff;
  text-decoration: none;
  font-weight: 700;
}

.diagram-panel { margin-top: 12px; }

/* 重置 + 自訂按鈕，避免被全站 button 樣式吃掉 */
.diagram-toggle {
  all: unset;                  /* 重要：清掉瀏覽器/全站預設 */
  box-sizing: border-box;
  width: 100%;
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 12px;
  background: #fff7ef;
  color: #a5732a;
  border: 1px solid #f0b56a;
  border-radius: 10px;
  font-weight: 700;
  cursor: pointer;
}

.diagram-toggle .label {
  flex: 1;                     /* 撐滿可用寬度 */
  text-align: center;          /* 文字在整行置中 */
}

.diagram-toggle .chevron {
  margin-left: auto;           /* 推到最右側 */
  transition: transform .2s ease;
}

.diagram-toggle .chevron.open { transform: rotate(180deg); }
.diagram-toggle:hover { filter: brightness(1.02); }
.diagram-toggle:focus { outline: 2px solid rgba(243,155,46,.35); outline-offset: 2px; }

.chevron { transition: transform .2s ease; display: inline-block; }
.chevron.open { transform: rotate(180deg); }

/* 圖片乖乖塞在框內（自適應高度） */
.diagram {
  margin-top: 8px;
  padding: 8px;
  border: 1px dashed #f0b56a;
  border-radius: 12px;
  background: #fff;
  box-shadow: 0 6px 18px rgba(0,0,0,.06);
  overflow: hidden;
}
.diagram img {
  display: block;
  width: 100%;
  height: auto;
  object-fit: contain;
  border-radius: 8px;
}


</style>
