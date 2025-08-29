// frontend/src/firebase.js
// ① Import 需要的 SDK
import { initializeApp } from "firebase/app";
import {
  initializeFirestore,
  persistentLocalCache,
  persistentMultipleTabManager,
} from "firebase/firestore";

// ② 貼上從 Console 複製的 Web App config
const firebaseConfig = {
  apiKey:            "AIzaSyBwNIB5dukrM1rWeuu2bjEitJKSut9chg0",
  authDomain:        "factgraph-38be7.firebaseapp.com",
  projectId:         "factgraph-38be7",
  storageBucket:     "factgraph-38be7.firebasestorage.app",
  messagingSenderId: "671154684730",
  appId:             "1:671154684730:web:94c81f0af3589fc603",
  measurementId:     "G-YGF3FZ1SG3"
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
