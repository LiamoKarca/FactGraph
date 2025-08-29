<template>
  <div class="error-message" :class="{ show: errorMessage }" v-if="errorMessage">
    {{ errorMessage }}
  </div>

  <!-- 裝飾圖案（沿用） -->
  <img src="/dog hend.png" class="dog-hand dog-hand-1" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-2" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-3" alt="dog hand" />

  <!-- Header -->
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
        <span class="menu-bar"></span>
        <span class="menu-bar"></span>
        <span class="menu-bar"></span>
      </button>
      <nav class="menu-dropdown" :class="{ show: showMenu }" ref="menu">
        <a href="/">首頁</a>
        <a href="/service">服務介紹</a>
        <a href="/about">關於我們</a>
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
      來找芒狗偵探，幫你了解真相！
    </h1>

    <!-- 分頁 -->
    <div class="tab-switch">
      <button class="tab-btn" :class="{ active: tabType === 'rag' }" @click="switchTab('rag')">實時 RAG</button>
      <button class="tab-btn" :class="{ active: tabType === 'writing' }" @click="switchTab('writing')">文案查詢</button>
      <button class="tab-btn" :class="{ active: tabType === 'question' }" @click="switchTab('question')">詢問模式</button>
    </div>

    <!-- 輸入區域 -->
    <div class="input-area">
      <div class="input-group">
        <input type="text" v-model="input" :placeholder="placeholderText" />

        <!-- 日期選擇（僅 writing / question 顯示） -->
        <input
          v-if="tabType !== 'rag'"
          ref="dateInput"
          class="date-input flatpickr-input"
          type="text"
          :readonly="isIOS ? false : true"
          :inputmode="isIOS ? 'none' : undefined"
          v-model="date"
          :placeholder="datePlaceholder"
          @focus="fp && fp.open()"
          @click="fp && fp.open()"
          @keydown.prevent
          style="cursor:pointer;"
        />

        <button
          id="query-btn"
          :class="{ bounce: isBouncing }"
          @click="tabType === 'rag' ? startRagCheck() : validateAndQuery()"
        >
          {{ loading ? '調查中…' : '開始查核' }}
        </button>
      </div>
    </div>

    <!-- 查核結果 -->
    <div class="answer-card" :class="{ default: !result }">
      <template v-if="loading">
        <div class="loading-spinner"></div>
        <p class="loading-text">芒狗調查中… 請稍候</p>
      </template>
      <template v-else>
        <div v-html="result || defaultMsg"></div>
      </template>
    </div>

    <!-- 對照知識（實時 RAG 隱藏） -->
    <div class="knowledge-wrapper" v-if="tabType !== 'rag'">
      <button class="collapse-btn" @click="knowledgeCollapsed = !knowledgeCollapsed">
        {{ knowledgeCollapsed ? '展開對照知識' : '收起對照知識' }}
        <span class="triangle" :class="{ rotated: !knowledgeCollapsed }">
          <svg width="18" height="18" viewBox="0 0 18 18"><polygon points="4,7 9,12 14,7" fill="#fff" /></svg>
        </span>
      </button>

      <transition name="fade">
        <div v-show="!knowledgeCollapsed" class="knowledge-card" :class="{ default: !knowledgeResult }">
          <template v-if="loading">
            <div class="loading-spinner"></div>
            <p class="loading-text">芒狗調查中… 請稍候</p>
          </template>
          <template v-else>
            <div v-html="knowledgeResult || defaultKnowledgeMsg"></div>
          </template>
        </div>
      </transition>
    </div>
  </div>

  <!-- 臨時網址彈窗 -->
  <transition name="fade">
    <div class="modal-mask" v-if="showUrlModal" @click.self="showUrlModal = false">
      <div class="modal-wrapper">
        <div class="modal-container">
          <h3>已建立臨時查詢網址</h3>
          <p class="url-text" @click="copyUrl" title="點擊複製">{{ tempUrl }}</p>
          <small>稍後可由此網址查看調查結果</small>
          <button class="modal-ok" @click="showUrlModal = false">確定</button>
        </div>
      </div>
    </div>
  </transition>
</template>

<script setup>
import axios from "axios";
import { ref, onMounted, computed, nextTick, watch, onBeforeUnmount } from "vue";
import flatpickr from "flatpickr";
import "flatpickr/dist/flatpickr.min.css";
import { getAuth, signInAnonymously } from "firebase/auth";
import { watchRagJob } from "../rag/watchRagJob";

// ─────────────────────────────────────────────
// 常量與環境偵測
// ─────────────────────────────────────────────
const BASE_URL = ""; // 同網域反向代理可留空
const isIOS =
  /iP(hone|ad|od)/.test(navigator.userAgent) ||
  (/Macintosh/.test(navigator.userAgent) && "ontouchend" in document);

// ─────────────────────────────────────────────
// 預設訊息
// ─────────────────────────────────────────────
const defaultMsg = `<p>這裡會顯示結果。</p>`;
const defaultKnowledgeMsg = `<p>這裡會顯示對照知識。</p>`;

// ─────────────────────────────────────────────
// 狀態（UI 與資料）
// ─────────────────────────────────────────────
const loading = ref(false);
const tabType = ref("writing"); // 'rag' | 'writing' | 'question'

// 輸入與日期（RAG 不用日期；此值僅給另外兩模式）
const input = ref("");
const date = ref("");

// 呈現結果
const result = ref("");
const knowledgeResult = ref("");
const knowledgeCollapsed = ref(true);

// RAG 專用監聽狀態
const jobId = ref(null);
const ragDoc = ref(null);
let stopWatch = null;

// UI 控制
const errorMessage = ref("");
const isBouncing = ref(false);
const titleHover = ref(false);
const showMenu = ref(false);
let errorTimeout = null;

// ─────────────────────────────────────────────
// 計算屬性
// ─────────────────────────────────────────────
const placeholderText = computed(() => {
  switch (tabType.value) {
    case "rag":
      return "臺灣的任何社會、政治問題，或新聞文案...";
    case "writing":
      return "請輸入完整新聞文案...";
    case "question":
      return "請輸入你的問題...";
    default:
      return "請輸入內容…";
  }
});

const datePlaceholder = computed(() => {
  if (date.value) return date.value;
  if (tabType.value === "question") return "今日或事件約莫日期";
  return "選擇新聞發布日期";
});

// ─────────────────────────────────────────────
// Header 與 UI 輔助
// ─────────────────────────────────────────────
function toggleMenu() {
  showMenu.value = !showMenu.value;
}
function pulseBtn() {
  isBouncing.value = true;
  setTimeout(() => (isBouncing.value = false), 300);
}
function showError(msg) {
  errorMessage.value = msg;
  clearTimeout(errorTimeout);
  errorTimeout = setTimeout(() => (errorMessage.value = ""), 2000);
}

// ─────────────────────────────────────────────
// flatpickr（僅 writing / question 使用）
// ─────────────────────────────────────────────
const dateInput = ref(null);
let fp = null;

function initDatePicker() {
  if (!dateInput.value) return;
  if (fp) {
    fp.destroy();
    fp = null;
  }
  fp = flatpickr(dateInput.value, {
    dateFormat: "Y/m/d",
    defaultDate: date.value || null,
    allowInput: isIOS ? true : false, // iOS 讓 placeholder 正常
    clickOpens: true,
    disableMobile: true,
    onChange: (_selected, dateStr) => {
      date.value = dateStr;
    },
  });
}
function destroyDatePicker() {
  if (fp) {
    fp.destroy();
    fp = null;
  }
}

// ─────────────────────────────────────────────
/** 切換分頁：清空並重建對應 UI */
// ─────────────────────────────────────────────
function switchTab(t) {
  tabType.value = t;
  input.value = "";
  date.value = "";
  result.value = "";
  knowledgeResult.value = "";
  jobId.value = null;
  ragDoc.value = null;
  if (stopWatch) {
    stopWatch();
    stopWatch = null;
  }
  nextTick(() => {
    if (tabType.value !== "rag") initDatePicker();
    else destroyDatePicker();
  });
}

// 同步 picker 狀態
watch(date, (d) => {
  if (fp && d) fp.setDate(d, true);
});
watch(tabType, async (t) => {
  if (t !== "rag") {
    await nextTick();
    initDatePicker();
  } else {
    destroyDatePicker();
  }
});

// ─────────────────────────────────────────────
// 生命週期
// ─────────────────────────────────────────────
onMounted(async () => {
  // Firestore 若需身份，匿名登入
  try {
    await signInAnonymously(getAuth());
  } catch {}
  if (tabType.value !== "rag") {
    await nextTick();
    initDatePicker();
  }
});
onBeforeUnmount(() => {
  destroyDatePicker();
  if (stopWatch) stopWatch();
});

// ─────────────────────────────────────────────
// RAG：建立任務 + Firestore 監聽（含降級輪詢）
// ─────────────────────────────────────────────
async function startRagCheck() {
  pulseBtn();
  if (!input.value) return showError("請輸入內容！");

  loading.value = true;
  result.value = "";
  knowledgeResult.value = defaultKnowledgeMsg;

  if (stopWatch) {
    stopWatch();
    stopWatch = null;
  }

  // 1) 建立後端任務（mode=rag；date 後端忽略）
  let taskId = null;
  try {
    const { data: task } = await axios.post(`${BASE_URL}/api/tasks`, {
      url: input.value,
      mode: "rag",
      date: "2000/01/01",
    });
    taskId = task.id;
    jobId.value = taskId;
    tempUrl.value = `${window.location.origin}/tasks/${taskId}`;
    showUrlModal.value = true;
  } catch (e) {
    console.error("建立 RAG 任務失敗", e);
    showError("無法建立 RAG 任務，請稍後再試");
    loading.value = false;
    return;
  }

  // 2) 監聽（失敗自動降級輪詢）
  stopWatch = watchRagJob(taskId, (doc) => {
    ragDoc.value = doc;

    // 若你的後端會在文件上塞 error.code=502 作為轉址信號，保留相容性
    if (doc?.error?.code === 502) {
      window.location.href = tempUrl.value;
      return;
    }

    if (doc.status === "PENDING" || doc.status === "RUNNING") return;

    if (doc.status === "FAILED") {
      result.value = `<p style="color:#c00;">RAG 失敗：${doc.last_error || "未知錯誤"}</p>`;
      loading.value = false;
      if (stopWatch) {
        stopWatch();
        stopWatch = null;
      }
      return;
    }

    if (doc.status === "DONE") {
      const ans = (doc.ragAnswer || "").trim();
      result.value = ans || "<p>已完成，但沒有可顯示的內容。</p>";
      loading.value = false;
      if (stopWatch) {
        stopWatch();
        stopWatch = null;
      }
    }
  });
}

// ─────────────────────────────────────────────
// 其餘兩種模式：保持你的原本寫法（只稍作整理）
// ─────────────────────────────────────────────
async function validateAndQuery() {
  pulseBtn();

  if (!input.value)
    return showError(tabType.value === "writing" ? "請輸入內容！" : "請輸入你的問題！");
  if (!date.value) return showError("請選擇日期！");

  loading.value = true;
  result.value = "";
  knowledgeResult.value = "";

  // 清掉上一輪監聽（若有）
  if (stopWatch) {
    stopWatch();
    stopWatch = null;
  }

  // 建立任務
  let taskId = null;
  try {
    const { data: task } = await axios.post(`${BASE_URL}/api/tasks`, {
      url: input.value,
      mode: tabType.value,
      date: date.value,
    });
    taskId = task.id;
    tempUrl.value = `${window.location.origin}/tasks/${taskId}`;
    showUrlModal.value = true;
  } catch (e) {
    console.error("建立臨時任務失敗", e);
    showError("無法建立查詢任務，請稍後再試");
    loading.value = false;
    return;
  }

  // 既有 onSnapshot（僅非 RAG 使用）
  const { doc, onSnapshot } = await import("firebase/firestore");
  const { db } = await import("../firebase");
  const docRef = doc(db, "url-results", taskId);

  const unsub = onSnapshot(
    docRef,
    (snap) => {
      if (!snap.exists()) return;
      const data = snap.data();

      if (data.error?.code === 502) {
        window.location.href = tempUrl.value;
        return;
      }
      if (data.status !== "DONE") return;

      const noAnswer =
        tabType.value === "question"
          ? !data.questionAnswer && !data.questionKnowledge
          : !data.writingAnswer && !data.writingKnowledge;

      if (noAnswer) {
        result.value = "<p>查無結果</p>";
        knowledgeResult.value = "<p>查無結果</p>";
      } else {
        if (tabType.value === "question") {
          result.value = data.questionAnswer || "<p>查無結果</p>";
          knowledgeResult.value = data.questionKnowledge || "<p>查無結果</p>";
        } else {
          result.value = data.writingAnswer || "<p>查無結果</p>";
          knowledgeResult.value = data.writingKnowledge || "<p>查無結果</p>";
        }
      }
      loading.value = false;
      unsub(); // 完成即取消
    },
    (error) => {
      console.error("[onSnapshot] error", error);
      showError("讀取結果失敗，請稍後再試");
      loading.value = false;
    }
  );
}

// ─────────────────────────────────────────────
// 臨時網址彈窗
// ─────────────────────────────────────────────
const showUrlModal = ref(false);
const tempUrl = ref("");
function copyUrl() {
  navigator.clipboard
    .writeText(tempUrl.value)
    .then(() => alert("已複製到剪貼簿"))
    .catch(() => alert("複製失敗，請手動複製"));
}
</script>

<style lang="scss" src="../assets/service.css"></style>

<style scoped>
.date-input {
  -webkit-appearance: none;
  appearance: none;
  border-radius: 12px !important;
}
.date-input::placeholder {
  color: #666;
  opacity: 1;
}

/* 正確的遮罩（全螢幕 + 置中） */
.modal-mask {
  position: fixed;
  z-index: 10000;
  inset: 0;
  background: rgba(0,0,0,0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
}
.modal-wrapper { width: 100%; max-width: 560px; }
.modal-container {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  padding: 20px 24px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.url-text {
  word-break: break-all;
  background: #f7f7f7;
  padding: 0.5rem;
  border-radius: 0.5rem;
  margin: 0.5rem 0 0.75rem;
  cursor: pointer;
  user-select: text;
}
.modal-ok {
  margin-top: 0.25rem;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 0.5rem;
  background: #409eff;
  color: #fff;
  cursor: pointer;
}

/* 淡入淡出動畫 */
.fade-enter-active, .fade-leave-active { transition: opacity .2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

/* 手機板按鈕置中修正（不改 template） */
@media (max-width: 480px) {
  #query-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: clamp(220px, 92vw, 360px);
    max-width: 100%;
    margin: 12px auto 0;
    box-sizing: border-box;
    align-self: center !important;
    float: none !important;
    position: static !important;
    text-align: center;
  }
}

/* 手機小螢幕：縮小「第一個輸入框」（非日期）的字與 placeholder，避免吃字 */
@media (max-width: 420px) {
  .input-group > input[type="text"]:not(.date-input) {
    font-size: clamp(13px, 3.6vw, 16px);
    width: 100%;
    box-sizing: border-box;
  }
  .input-group > input[type="text"]:not(.date-input)::placeholder {
    font-size: clamp(11px, 3.2vw, 14px);
    letter-spacing: .2px;
  }
  .input-group > input[type="text"]:not(.date-input)::-webkit-input-placeholder { font-size: clamp(11px, 3.2vw, 14px); }
  .input-group > input[type="text"]:not(.date-input):-ms-input-placeholder     { font-size: clamp(11px, 3.2vw, 14px); }
  .input-group > input[type="text"]:not(.date-input)::-moz-placeholder        { font-size: clamp(11px, 3.2vw, 14px); opacity:1; }
}
</style>
