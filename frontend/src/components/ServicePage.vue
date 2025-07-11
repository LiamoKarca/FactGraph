<script setup>
import { ref, onMounted, watch } from 'vue'
import flatpickr from 'flatpickr'
import 'flatpickr/dist/flatpickr.min.css'

const tabType = ref('link')
const input = ref('')
const date = ref('')
const tabResults = ref({
  link: '',
  writing: '',
  question: ''
})
const defaultMsg = '請輸入問題並點擊查詢，這裡會顯示答案。'
const result = ref(defaultMsg)
const errorMessage = ref('')
const isBouncing = ref(false)
let errorTimeout = null

const datePicker = ref(null)

onMounted(() => {
  if (datePicker.value) {
    flatpickr(datePicker.value, {
      dateFormat: 'Y/m/d',
      defaultDate: date.value,
      onChange: (selectedDates, dateStr) => {
        date.value = dateStr
      }
    })
  }
})

function switchTab(type) {
  tabType.value = type
  input.value = ''
  date.value = ''
  result.value = tabResults.value[type] || defaultMsg
}

function validateAndQuery() {
  // 按鈕 bounce 效果
  isBouncing.value = true
  setTimeout(() => isBouncing.value = false, 300)

  if (!input.value) {
    showError('請輸入問題！')
    return
  }
  if (!date.value) {
    showError('請選擇日期！')
    return
  }
  if (tabType.value === 'link' && !/^https?:\/\/.+\..+/.test(input.value)) {
    showError('請輸入網址！')
    return
  }

  const output = `選擇的日期：${date.value}<br>你輸入的內容：${input.value}`
  result.value = output
  tabResults.value[tabType.value] = output
}

function showError(msg) {
  errorMessage.value = msg
  clearTimeout(errorTimeout)
  errorTimeout = setTimeout(() => {
    errorMessage.value = ''
  }, 2000)
}
</script>

<template>
  <div class="error-message" :class="{ show: errorMessage }" v-if="errorMessage">{{ errorMessage }}</div>

  <img src="/dog hend.png" class="dog-hand dog-hand-1" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-2" alt="dog hand" />
  <img src="/dog hend.png" class="dog-hand dog-hand-3" alt="dog hand" />

  <header class="header">
    <div class="header-left">
      <span class="header-icon" role="img" aria-label="camera">
        <img src="/camara icon.png" alt="camera icon" style="width:60px;height:60px;display:block;" />
      </span>
      <span class="header-title">圖破謠言，新聞真偽辨識AI</span>
    </div>
    <div class="header-right">
      <button class="login-btn">Login</button>
      <button class="menu-btn" @click="$refs.menu.classList.toggle('show')">
        <span class="menu-bar"></span>
        <span class="menu-bar"></span>
        <span class="menu-bar"></span>
      </button>
      <nav class="menu-dropdown" ref="menu">
        <a href="#">首頁</a>
        <a href="#">軟體服務</a>
        <a href="#">產品介紹</a>
        <a href="#">關於我們</a>
        <a href="#">聯絡我們</a>
      </nav>
    </div>
  </header>

  <img src="/dog icon.png" class="bg-dog" alt="dog icon" />

  <div class="container">
    <h1>來找芒狗偵探，幫你了解真相！</h1>

    <div class="tab-switch">
      <button
        class="tab-btn"
        :class="{ active: tabType === 'link' }"
        @click="switchTab('link')"
      >連結查詢</button>
      <button
        class="tab-btn"
        :class="{ active: tabType === 'writing' }"
        @click="switchTab('writing')"
      >文案查詢</button>
      <button
        class="tab-btn"
        :class="{ active: tabType === 'question' }"
        @click="switchTab('question')"
      >詢問模式</button>
    </div>

    <div class="input-area">
      <div class="input-group">
        <input type="text" v-model="input" :placeholder="tabType === 'link' ? '請輸入新聞連結...' : tabType === 'writing' ? '請輸入文案內容...' : '請輸入你的問題...'" />
        <input ref="datePicker" class="date-input" placeholder="年/月/日" readonly style="background:#fff8f0; cursor:pointer;" />
        <button
          id="query-btn"
          :class="{ bounce: isBouncing }"
          @click="validateAndQuery"
          >開始查詢</button>
      </div>
    </div> 

    <div class="answer-card" v-html="result"></div>
  </div>
</template>

<style lang="scss" src="../assets/service.css"></style>
