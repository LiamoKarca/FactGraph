<template>
  <!-- 裝飾圖案 -->
  <img src="/dog hend.png" class="dog-hand dog-hand-1" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-2" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-3" alt="dog hand" />

  <!-- Header：與首頁相同 -->
  <header class="header">
    <RouterLink to="/" class="header-left" aria-label="回到主查詢頁面" style="text-decoration:none; color:inherit; display:flex; align-items:center;">
      <span class="header-icon" role="img" aria-label="camera">
        <img src="/camara icon.png" alt="camera icon" style="width:60px;height:60px;display:block;" />
      </span>
      <div class="header-title">
        <div class="title">芒狗偵探</div>
        <div class="header-subtitle">新聞查核，透明可信，快速回覆。</div>
      </div>
    </RouterLink>

    <div class="header-right">
      <button class="menu-btn" @click="toggleMenu" :aria-expanded="showMenu" aria-label="toggle menu">
        <span class="menu-bar"></span><span class="menu-bar"></span><span class="menu-bar"></span>
      </button>
      <nav class="menu-dropdown" :class="{ show: showMenu }">
        <router-link to="/">首頁</router-link>
        <router-link to="/service">服務介紹</router-link>
        <router-link to="/about">關於我們</router-link>
      </nav>
    </div>
  </header>

  <!-- 同款背景圖 -->
  <img src="/dog icon.png" class="bg-dog" alt="dog icon" />

  <!-- 內容 -->
  <div class="container">
    <h1
      class="page-title"
      @mouseenter="titleHover = true"
      @mouseleave="titleHover = false"
      :style="{ transform: titleHover ? 'scale(1.01)' : 'scale(1)', transition: 'transform .15s ease' }"
    >
      關於我們
    </h1>

    <section class="section">
      <h2>使命 Mission</h2>
      <p>
        在資訊爆炸與立場分歧的環境裡，我們以「可追溯、可驗證、可重現」為核心，
        幫助大眾在最短時間內看見「事實與說法的差距」；堅持中立、引用可查、過程透明，
        讓每一次判斷都有跡可循。
      </p>
    </section>

    <section class="section">
      <h2>方法論 Methodology</h2>
      <ul class="bullets">
        <li>知識圖譜：以 Neo4j 建立事件關聯，搭配三元組提取與向量比對，降低模型幻覺，提供可驗證依據。</li>
        <li>AI Agent 設計：使用提示工程，進行知識提取、關鍵字抽取、分類任務、結果判斷與生成，完整流水線。</li>
        <li>一致／矛盾／缺漏／評分：逐條比對來源與內容，避免過度推論，保留不確定性並清楚標示，最終給予綜合評分。</li>
        <li>引用片段對應：每段輸出對應來源與段落區間（段落索引），讓讀者可自行回溯。</li>
        <li>任務可回訪：產生專屬連結，查核完成後可隨時回到同一結果，強化可追蹤性與分享性。</li>
      </ul>
    </section>

    <section class="section">
      <h2>技術棧 Tech Stack</h2>
        <p><strong>－　開發環境：</strong>WSL2</p>
        <p><strong>－　IDE：</strong>Visual Studio Code、Cursor</p>
        <p><strong>－　DevOps：</strong>GitHub、Git</p>

        <p><strong>－　前端</strong></p>
        <ul class="bullets">
          <li><strong>UI 設計：</strong>Figma</li>
          <li><strong>網頁框架與畫面建構：</strong>Vue、Vite、HTML、CSS、JavaScript、NPM/Yarn、Node.js</li>
          <li><strong>專案網站部屬：</strong>Firebase</li>
        </ul>

        <p><strong>－　後端</strong></p>
        <ul class="bullets">
          <li><strong>資料蒐集：</strong>Python、web crawler</li>
          <li><strong>向量運算：</strong>BERT</li>
          <li><strong>API 串接：</strong>FastAPI、OpenAI API</li>
          <li><strong>專案程式包裝與部屬：</strong>Docker、Google Cloud Run</li>
        </ul>

        <p><strong>－　資料庫</strong></p>
        <ul class="bullets">
          <li><strong>資料儲存：</strong>Neo4j、MongoDB、Firestore</li>
        </ul>
        
        <!-- 可收合的架構圖 -->
        <div class="diagram-panel">
          <button
            class="diagram-toggle"
            @click="toggleArch"
            :aria-expanded="showArch.toString()"
            aria-controls="arch-panel"
            type="button"
          >
            <span class="label">{{ showArch ? '收起系統架構圖' : '展開系統架構圖' }}</span>
            <span class="chevron" :class="{ open: showArch }">⌄</span>
          </button>

          <div id="arch-panel" class="diagram" v-show="showArch">
            <img :src="archDiagram" alt="系統架構圖（Architecture Diagram）" />
          </div>
        </div>

    </section>

    <section class="section">
      <h2>研究與倫理 Research & Ethics</h2>
      <ul class="bullets">
        <li>中立與可重現：同樣輸入應得到同樣結論與引用，模型推論結果須可交叉驗證。</li>
        <li>資料治理：尊重著作權與隱私，清楚標示引用來源並維持合規使用。</li>
        <li>限制說明：使用嚴格的提示工程進行設計，且證據不足時，主動標示「缺漏」，不以想像補齊事實。</li>
      </ul>
    </section>

    <section class="section">
      <h2>團隊與夥伴 Team & Partners</h2>
      <p><strong>指導教授｜陳灯能 老師</strong></p>
      <p>B11156007 張宏瑋｜前端切版、美術設計</p>
      <p>B11156057 賴彥霖｜前端&後端設計、整合與部署</p>
      <p>B11156039 黃品翰｜文書處理</p>
    </section>

    <section class="section">
      <h2>里程碑 & 路線圖 Milestones & Roadmap</h2>
      <ul class="bullets">
        <li>2025 Q2：專題計劃與資料蒐集；確立新聞查核流程與準則。</li>
        <li>2025 Q3：功能實作、Google Cloud Run 部署與公開測試。</li>
        <li>2025 Q4：專題發表，正式部署。</li>
      </ul>
    </section>

    <section class="section">
      <h2>聯絡我們 Contact Us</h2>
      <p>信箱：<a href="mailto:karca5103@gmail.com">karca5103@gmail.com</a></p>
      <p>GitHub：<a href="https://github.com/LiamoKarca/FactGraph" target="_blank" rel="noopener">https://github.com/LiamoKarca/FactGraph</a></p>
    </section>

    <section class="section">
      <h2>文件下載 Document Download</h2>
        <p>
          計劃書：
          <a :href="planPdf" target="_blank" rel="noopener">線上預覽（PDF）</a>
          ｜<a :href="planPdf" download="AI-Fact-Graph-Plan.pdf">下載</a>
        </p>

        <!-- 可收合的 PDF 預覽 -->
        <div class="diagram-panel">
          <button
            class="diagram-toggle"
            @click="togglePlanPdf"
            :aria-expanded="showPlanPdf.toString()"
            aria-controls="plan-pdf"
            type="button"
          >
            <span class="label">{{ showPlanPdf ? '收起計劃書預覽' : '展開計劃書預覽' }}</span>
            <span class="chevron" :class="{ open: showPlanPdf }">⌄</span>
          </button>

          <div id="plan-pdf" class="diagram" v-show="showPlanPdf">
            <!-- 用 iframe 內嵌，不新增 CSS：高度用 inline style 控制 -->
            <iframe
              :src="planPdf"
              title="AI Fact Graph 計劃書（PDF）"
              style="width:100%;height:min(70vh,900px);border:0;"
            ></iframe>

            <!-- 行動裝置或瀏覽器不支援內嵌 PDF 的備援連結 -->
            <p style="margin-top:8px;">
              若無法預覽，請改為
              <a :href="planPdf" target="_blank" rel="noopener">在新分頁開啟</a>
              或 <a :href="planPdf" download="AI-Fact-Graph-Plan.pdf">直接下載</a>。
            </p>
  </div>
</div>

    </section>

  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
const showMenu = ref(false)
const titleHover = ref(false)
const toggleMenu = () => (showMenu.value = !showMenu.value)

/* 點擊其他地方時關閉選單 */
import archDiagram from '../assets/architecture-diagram.png'
const showArch = ref(false)
const toggleArch = () => (showArch.value = !showArch.value)

/* PDF */
import planPdf from '../assets/AI-Fact-Graph-Plan.pdf'  
const showPlanPdf = ref(false)                           
const togglePlanPdf = () => (showPlanPdf.value = !showPlanPdf.value)

</script>

<style scoped>
@import '../assets/service.css';
.container{ padding:24px; }
.page-title {
  font-size: clamp(28px, 4.2vw, 56px); /* 手機→桌機自動放大 */
  line-height: 1.1;
  margin-bottom: 12px;
  color: #f39b2e;
  letter-spacing: .5px;
}.section{ margin:20px 0; }
.bullets{ margin-left:1.25rem; }

/* 收合區容器與按鈕（沿用 ServiceIntro 的風格） */
.diagram-panel { margin-top: 10px; }
.diagram-toggle {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 12px;
  background: #FFA500;
  border: 1px solid #ffe1bd;
  box-shadow: 0 6px 18px rgba(0,0,0,.06);
  cursor: pointer;
  user-select: none;
}
.diagram-toggle .label { font-weight: 600; letter-spacing: .2px; }
.diagram-toggle .chevron {
  margin-left: auto;
  transition: transform .2s ease;
  display: inline-block;
}
.diagram-toggle .chevron.open { transform: rotate(180deg); }
.diagram-toggle:hover { filter: brightness(1.02); }
.diagram-toggle:focus { outline: 2px solid rgba(243,155,46,.35); outline-offset: 2px; }

/* 圖片外框與自適應 */
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
.diagram-toggle .label { 
  flex: 1;              /* 讓標籤撐滿可用寬度 */
  text-align: center;   /* 文字置中 */
}
.diagram-toggle .chevron { margin-left: auto; }  /* 保持箭頭靠右 */

</style>
