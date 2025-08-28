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
    <div class="header-left">
      <span class="header-icon" role="img" aria-label="camera">
        <img src="/camara icon.png" alt="camera icon" style="width:60px;height:60px;display:block;" />
      </span>
      <div class="header-title">
        <div class="title">芒狗偵探</div>
        <div class="header-subtitle">新聞查核，透明可信，快速回覆。</div>
      </div>
    </div>

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

    <!-- 結果卡片 -->
    <div class="answer-card" :class="{ default: !result }">
      <template v-if="loading || status !== 'DONE'">
        <div class="loading-spinner"></div>
        <p class="loading-text">芒狗調查中… 請稍候</p>
      </template>
      <template v-else>
        <div v-html="result || defaultMsg"></div>
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
        <div v-show="!knowledgeCollapsed" class="knowledge-card" :class="{ default: !knowledge }">
          <template v-if="loading || status !== 'DONE'">
            <div class="loading-spinner"></div>
            <p class="loading-text">芒狗調查中… 請稍候</p>
          </template>
          <template v-else>
            <div v-html="knowledge || defaultKnowledgeMsg"></div>
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

/** 與首頁相同的預設文字 */
const defaultMsg = `<p>這裡會顯示結果。</p>`
const defaultKnowledgeMsg = `<p>這裡會顯示對照知識。</p>`

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

const result = ref('')          // 主要結果
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

/** 將後端 mode 轉成人類可讀 */
const modeLabel = computed(() => {
  switch (mode.value) {
    case 'rag': return '實時 RAG'
    case 'writing': return '文案查詢'
    case 'question': return '詢問模式'
    default: return mode.value || '即將出爐...'
  }
})

/** 將 Firestore 資料套用到畫面 */
function applySnapshot(data) {
  if (data.mode) mode.value = data.mode

  status.value = data.status || 'PENDING'
  if (status.value !== 'DONE') {
    loading.value = true
    return
  }

  // DONE：依模式填資料
  if (mode.value === 'rag') {
    result.value = data.ragAnswer || ''
    knowledge.value = '' // RAG 不顯示對照知識
  } else if (mode.value === 'question') {
    result.value = data.questionAnswer || ''
    knowledge.value = data.questionKnowledge || ''
  } else {
    // 默認以 writing 視之
    result.value = data.writingAnswer || ''
    knowledge.value = data.writingKnowledge || ''
  }

  loading.value = false
}

/** 訂閱任務狀態 */
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
      if (!snap.exists()) return
      const data = snap.data()

      // 後端若標註錯誤，直接顯示在結果區
      if (data.error?.code && data.error?.message) {
        loading.value = false
        status.value = 'DONE'
        mode.value = data.mode || mode.value
        result.value = `<p>⚠️ ${data.error.message}</p>`
        knowledge.value = mode.value === 'rag' ? '' : `<p>（無對照知識）</p>`
        return
      }

      applySnapshot(data)
    },
    err => {
      loading.value = false
      status.value = 'DONE'
      result.value = `<p>⚠️ 讀取任務狀態失敗：${err.message}</p>`
      knowledge.value = ''
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

</style>
