// frontend/src/firebase.js
// ① Import 需要的 SDK
import { initializeApp } from "firebase/app";
import {
  initializeFirestore,
  persistentLocalCache,
  persistentMultipleTabManager,
} from "firebase/firestore";

// ② 貼上從 Console 複製的 Web App config
// 或將 frontend/.env.local 進行設置
const firebaseConfig = {
  apiKey:            import.meta.env.VITE_FB_API_KEY,
  authDomain:        import.meta.env.VITE_FB_AUTH_DOMAIN,
  projectId:         import.meta.env.VITE_FB_PROJECT_ID,
  storageBucket:     import.meta.env.VITE_FB_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FB_MESSAGING_SENDER_ID,
  appId:             import.meta.env.VITE_FB_APP_ID,
  measurementId:     import.meta.env.VITE_FB_MEASUREMENT_ID,
};

// ③ 初始化 Firebase App
const firebaseApp = initializeApp(firebaseConfig);

// ④ 初始化 Firestore（相容模式 + 本地快取；避免 listen 被外掛/代理擋掉）
export const db = initializeFirestore(firebaseApp, {
  experimentalAutoDetectLongPolling: true,
  useFetchStreams: false,
  localCache: persistentLocalCache({
    tabManager: persistentMultipleTabManager(),
  }),
});
