// 僅在 DEV 會被載入
console.log("VITE_FB_API_KEY 存在？", !!import.meta.env.VITE_FB_API_KEY);
console.log("VITE_FB_AUTH_DOMAIN 存在？", !!import.meta.env.VITE_FB_AUTH_DOMAIN);
console.log("VITE_FB_PROJECT_ID 存在？", !!import.meta.env.VITE_FB_PROJECT_ID);
console.log("VITE_FB_STORAGE_BUCKET 存在？", !!import.meta.env.VITE_FB_STORAGE_BUCKET);
console.log("VITE_FB_MESSAGING_SENDER_ID 存在？", !!import.meta.env.VITE_FB_MESSAGING_SENDER_ID);
console.log("VITE_FB_APP_ID 存在？", !!import.meta.env.VITE_FB_APP_ID);
console.log("VITE_FB_MEASUREMENT_ID 存在？", !!import.meta.env.VITE_FB_MEASUREMENT_ID);