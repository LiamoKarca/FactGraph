import { createApp } from 'vue'
import './assets/service.css'
import App from './App.vue'

// 引入 firebase.js（初始化）＆ router
import { router } from "./router";
import "./firebase";        // 確保 firebase.js 執行一次.

// 本地開發時，載入測試環境設定
if (import.meta.env.DEV) {
  import("./testEnv.js");
}

// ── 初始化 Vue 應用 ──
createApp(App)
  .use(router)             // 掛載 router
  .mount("#app");

// 本地開發時，Vite 會自動處理路由和靜態資源
// % cd ~/dev/FactGraph/frontend
// % yarn install
// % yarn run dev

// 前端更新時，需要重新部署到 Firebase Hosting
// % cd ~/dev/FactGraph/frontend
// % yarn install
// % yarn run build
// % firebase deploy --only hosting