<template>
  <div class="error-message" :class="{ show: errorMessage }" v-if="errorMessage">
    {{ errorMessage }}
  </div>

  <!-- 裝飾圖案（與首頁相同） -->
  <img src="/dog hend.png" class="dog-hand dog-hand-1" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-2" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-3" alt="dog hand" />

  <!-- Header（與首頁相同結構與 class） -->
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
        <span class="menu-bar"></span>
        <span class="menu-bar"></span>
        <span class="menu-bar"></span>
      </button>
      <nav class="menu-dropdown" :class="{ show: showMenu }" ref="menu">
        <a href="/">首頁</a>
        <a href="#">服務介紹</a>
        <a href="#">關於我們</a>
      </nav>
    </div>
  </header>

  <img src="/dog icon.png" class="bg-dog" alt="dog icon" />

  <!-- 主體容器 -->
  <div class="container">
    <h1
      @mouseenter="titleHover = true"
      @mouseleave="titleHover = false"
      :style="{ transform: titleHover ? 'scale(1.01)' : 'scale(1)', transition: 'transform .15s ease' }"
    >
      查核結果
    </h1>

    <!-- 任務資訊列 -->
    <p style="margin:8px 0 12px;color:#666;font-size:.95rem;">
      Task ID：<code>{{ taskId }}</code>
      <span v-if="status"> ｜ 狀態：{{ status }} ｜ {{ modeLabel }}</span>
    </p>

    <!-- 分數條（僅在 verifier / rag_mode；自動從結果文本抓「x/10」或「x分」或「可信度評分 x」） -->
    <div
      v-if="status === 'DONE' && (normalizedMode === 'verifier' || normalizedMode === 'rag_mode') && Number.isFinite(score)"
      class="score-wrap"
      role="region"
      aria-label="分數"
    >
      <div class="score-head">
        <span class="score-label">{{ normalizedMode === 'rag_mode' ? '可信度評分' : '可信度評分' }}</span>
        <span class="score-value">{{ scoreDisplay }}</span>
      </div>
      <div class="score-track" role="progressbar" :aria-valuemin="0" :aria-valuemax="10" :aria-valuenow="score">
        <div class="score-fill" :style="{ width: scorePercent + '%' }"></div>
      </div>
    </div>

    <!-- 結果卡片 -->
    <div class="answer-card" :class="{ default: isResultEmpty }">
      <template v-if="loading || status !== 'DONE'">
        <div class="loading-spinner"></div>
        <p class="loading-text">芒狗調查中… 請稍候</p>
      </template>
      <template v-else>
        <div v-if="isResultEmpty" v-html="emptyMsg"></div>
        <div v-else v-html="result"></div>
      </template>
    </div>

    <!-- 對照知識：僅在非 RAG 模式顯示 -->
    <div class="knowledge-wrapper" v-if="mode !== 'rag'">
      <button class="collapse-btn" @click="knowledgeCollapsed = !knowledgeCollapsed">
        {{ knowledgeCollapsed ? '展開對照知識' : '收起對照知識' }}
        <span class="triangle" :class="{ rotated: !knowledgeCollapsed }">
          <svg width="18" height="18" viewBox="0 0 18 18"><polygon points="4,7 9,12 14,7" fill="#fff" /></svg>
        </span>
      </button>

      <transition name="fade">
        <div v-show="!knowledgeCollapsed" class="knowledge-card" :class="{ default: isKnowledgeEmpty }">
          <template v-if="loading || status !== 'DONE'">
            <div class="loading-spinner"></div>
            <p class="loading-text">芒狗調查中… 請稍候</p>
          </template>
          <template v-else>
            <div v-if="isKnowledgeEmpty" v-html="emptyMsg"></div>
            <div v-else v-html="knowledge"></div>
          </template>
        </div>
      </transition>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, computed } from 'vue'
import { useRoute } from 'vue-router'
import { db } from '../firebase'
import { doc, onSnapshot } from 'firebase/firestore'

/** 空狀態訊息 */
const emptyMsg = `<p>查無結果。</p>`

// Header 狀態
const showMenu = ref(false)
function toggleMenu(){ showMenu.value = !showMenu.value }
const titleHover = ref(false)

// 來自路由的 task id
const route = useRoute()
const taskId = ref(String(route.params.id || ''))

// 任務狀態與資料
const status = ref('PENDING')   // Firestore status
const loading = ref(true)
const mode = ref('')            // 'rag' | 'writing' | 'question'

const result = ref('')          // 主要結果（HTML 字串）
const knowledge = ref('')       // 對照知識
const knowledgeCollapsed = ref(true)

const errorMessage = ref('')
let errorTimeout = null
let unsubscribe = null

function showError(msg) {
  errorMessage.value = msg
  clearTimeout(errorTimeout)
  errorTimeout = setTimeout(() => (errorMessage.value = ''), 2000)
}

/** 將後端 mode 轉成人類可讀（原樣保留） */
const modeLabel = computed(() => {
  switch (mode.value) {
    case 'rag': return '實時 RAG'
    case 'writing': return '文案查詢'
    case 'question': return '詢問模式'
    default: return mode.value || '即將出爐...'
  }
})

/** 統一前端顯示用的模式：writing→verifier、rag→rag_mode、question→answerer */
const normalizedMode = computed(() => {
  switch (mode.value) {
    case 'rag': return 'rag_mode'
    case 'writing': return 'verifier'
    case 'question': return 'answerer'
    default: return mode.value || ''
  }
})

/** 從結果 HTML 中解析分數：優先 x/10；其次「x分」；或「可信度評分 x」；回傳 0~10 的數字 */
const score = computed(() => {
  try {
    const html = String(result.value || '')
    const plain = html.replace(/<[^>]+>/g, ' ')
    const normalized = plain.replace(/[０-９]/g, ch => String.fromCharCode(ch.charCodeAt(0) - 0xFF10 + 0x30))
    let m = normalized.match(/([0-9]+(?:\.[0-9]+)?)\s*\/\s*10/)
    if (!m) m = normalized.match(/\b([0-9]+(?:\.[0-9]+)?)\s*分\b/)
    if (!m) m = normalized.match(/可信度\s*評分[^0-9]*([0-9]+(?:\.[0-9]+)?)/)
    if (!m) return undefined
    const n = parseFloat(m[1])
    if (!Number.isFinite(n)) return undefined
    return Math.max(0, Math.min(10, n))
  } catch {
    return undefined
  }
})
const scorePercent = computed(() => (score.value != null ? Math.round((score.value / 10) * 1000) / 10 : 0))
const scoreDisplay = computed(() => (score.value != null ? `${Number(score.value.toFixed(2))}/10` : ''))

/** 判斷值是否「為空」：支援字串/HTML/陣列/物件 */
function isEmptyValue(v) {
  if (v == null) return true
  if (typeof v === 'string') {
    const noTags = v.replace(/<[^>]*>/g, '').replace(/&nbsp;/g, ' ')
    return noTags.trim() === ''
  }
  if (Array.isArray(v)) return v.length === 0
  if (typeof v === 'object') return Object.values(v).every(isEmptyValue)
  return false
}
const isResultEmpty = computed(() => isEmptyValue(result.value))
const isKnowledgeEmpty = computed(() => isEmptyValue(knowledge.value))

/** 將 Firestore 資料套用到畫面（保留原模式邏輯，補齊「空結果」處理） */
function applySnapshot(data) {
  if (data?.mode) mode.value = data.mode
  status.value = data?.status || 'PENDING'

  if (status.value !== 'DONE') {
    // 尚未完成：維持載入中
    loading.value = true
    return
  }

  // 完成：依模式填資料
  if (mode.value === 'rag') {
    result.value = data?.ragAnswer || ''
    knowledge.value = '' // RAG 不顯示對照知識
  } else if (mode.value === 'question') {
    result.value = data?.questionAnswer || ''
    knowledge.value = data?.questionKnowledge || ''
  } else {
    // 默認以 writing 視之
    result.value = data?.writingAnswer || ''
    knowledge.value = data?.writingKnowledge || ''
  }

  loading.value = false
}

/** 訂閱任務狀態：若文件不存在或內容為空，也視為 DONE + 查無結果 */
function subscribe(id) {
  if (!id) {
    showError('缺少任務 ID')
    return
  }
  if (unsubscribe) { try { unsubscribe() } catch {} unsubscribe = null }

  loading.value = true
  status.value = 'PENDING'
  result.value = ''
  knowledge.value = ''

  const docRef = doc(db, 'url-results', id)
  unsubscribe = onSnapshot(
    docRef,
    snap => {
      if (!snap.exists()) {
        // 明確：Firestore 查無此任務 → 顯示「查無結果」
        loading.value = false
        status.value = 'DONE'
        mode.value = mode.value || '' // 保持現狀
        result.value = ''
        knowledge.value = ''
        return
      }

      const data = snap.data()

      // 後端若標註錯誤，直接顯示於結果區
      if (data?.error?.code && data?.error?.message) {
        loading.value = false
        status.value = 'DONE'
        mode.value = data.mode || mode.value
        result.value = `<p>⚠️ ${data.error.message}</p>`
        knowledge.value = mode.value === 'rag' ? '' : ''
        return
      }

      applySnapshot(data)
    },
    err => {
      loading.value = false
      status.value = 'DONE'
      result.value = `<p>⚠️ 讀取任務狀態失敗：${err.message}</p>`
      knowledge.value = ''
      showError('讀取失敗')
    }
  )
}

onMounted(() => subscribe(taskId.value))
onBeforeUnmount(() => { if (unsubscribe) { try { unsubscribe() } catch {} } })
watch(() => route.params.id, v => {
  taskId.value = String(v || '')
  subscribe(taskId.value)
})
</script>

<style scoped>
@import '../assets/service.css';

.container { padding: 1rem; }

/* ====== 分數條 ====== */
.score-wrap{
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(0,0,0,0.05);
  border-radius: 14px;
  padding: 10px 12px;
  margin: 12px 0 14px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.06);
  backdrop-filter: saturate(120%) blur(2px);
}
.score-head{ display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.score-label{ font-weight:600; color:#333; letter-spacing: .02em; }
.score-value{ color:#555; font-variant-numeric: tabular-nums; font-weight:600; }
.score-track{
  position: relative;
  height: 14px;
  background: #ececec;
  border-radius: 999px;
  overflow: hidden;
}
.score-fill{
  height: 100%;
  width: 0%;
  border-radius: 999px;
  background: linear-gradient(90deg, #f59e0b, #fb923c, #f59e0b);
  box-shadow: inset 0 1px 1px rgba(255,255,255,0.6);
  transition: width .6s ease;
}
@media (prefers-color-scheme: dark) {
  .score-wrap{ background: rgba(20,20,20,0.55); border-color: rgba(255,255,255,0.06); box-shadow: 0 6px 18px rgba(0,0,0,0.35); }
  .score-label{ color:#f3f3f3; }
  .score-value{ color:#ddd; }
  .score-track{ background: rgba(255,255,255,0.12); }
}
</style>
