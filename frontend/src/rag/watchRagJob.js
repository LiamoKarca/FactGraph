// frontend/src/rag/watchRagJob.js
// 只處理 RAG 文件的 Firestore 監聽。監聽失敗或 5 秒內沒等到第一筆，就自動降級成輪詢。
// 不影響其他模式的既有程式。

import { doc, onSnapshot, getDoc } from "firebase/firestore";
import { db } from "../firebase";

/** @typedef {"PENDING"|"RUNNING"|"DONE"|"FAILED"} JobStatus */
/**
 * @typedef {Object} RagDoc
 * @property {JobStatus} status
 * @property {"rag"} mode
 * @property {string|null} ragAnswer
 * @property {string|null} last_error
 */

export function watchRagJob(jobId, onUpdate) {
    const ref = doc(db, "url-results", jobId);
    let firstSnapArrived = false;
    let closed = false;

    // 2 秒輪詢，最多 90 秒
    const poll = async () => {
        const start = Date.now();
        const limit = 90_000;
        while (!closed && Date.now() - start < limit) {
            try {
                const s = await getDoc(ref);
                if (s.exists()) {
                    onUpdate(/** @type {RagDoc} */(s.data()));
                    const st = s.data().status;
                    if (st === "DONE" || st === "FAILED") return;
                }
            } catch { }
            await new Promise(r => setTimeout(r, 2000));
        }
    };

    const unsub = onSnapshot(
        ref,
        (snap) => {
            firstSnapArrived = true;
            if (!snap.exists()) return;
            const data = /** @type {RagDoc} */ (snap.data());
            onUpdate(data);
            if (data.status === "DONE" || data.status === "FAILED") {
                closed = true;
                unsub();
            }
        },
        async () => {
            // 監聽失敗（常見：ERR_BLOCKED_BY_CLIENT）→ 直接降級輪詢
            try { unsub(); } catch { }
            await poll();
        }
    );

    // 安全網：5 秒內完全沒有 snapshot，就啟用輪詢
    setTimeout(() => {
        if (!firstSnapArrived && !closed) {
            try { unsub(); } catch { }
            poll();
        }
    }, 5000);

    return () => {
        closed = true;
        try { unsub(); } catch { }
    };
}
