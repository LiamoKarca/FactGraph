# -*- coding: utf-8-sig -*-
"""
MiniLM-L12-v2 模型下載與快速測試腳本

此腳本會：
1) 使用 Sentence-Transformers 載入多語句向量模型
   'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
2) 將模型快取至自訂資料夾 'models/MiniLM'
3) 對數句中文測試語句進行向量化，並輸出向量摘要

執行方式：
    (venv) python models/download_MiniLM-L12-v2.py

備註：
- 該模型的句向量維度為 384（官方模型頁面說明）。
- SentenceTransformer 支援以 `cache_folder` 指定本機快取路徑。
- `encode()` 參數 `convert_to_numpy=True` 可直接回傳 NumPy 陣列。

Google Style Docstring 規範：
- 本檔案 docstring 與函式 docstring 均採用 Google Style。
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
    """主程式入口。

    流程：
        1) 載入 MiniLM-L12-v2 多語句向量模型（384 維）
        2) 進行簡單的向量化測試
        3) 輸出維度與部分向量摘要

    Raises:
        Exception: 任一階段發生未預期錯誤時拋出，並由下方 try/except 捕捉印出。
    """
    # 指定模型與自訂快取資料夾
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_cache_path: str = "models/MiniLM"  # 自訂的模型儲存資料夾

    print(f"🚀 [1/3] 開始載入模型：{model_name}")
    t0 = time.time()
    # 說明：SentenceTransformer 支援以 cache_folder 指定本機快取。
    # 若本機已有快取，將直接載入；否則會自動下載到該路徑。
    model = SentenceTransformer(
        model_name,
        cache_folder=model_cache_path,
        trust_remote_code=True,
    )
    load_secs = time.time() - t0
    print(f"✅ 模型成功載入（耗時 {load_secs:.2f} 秒）")

    # 額外輸出嵌入維度（此模型為 384 維）
    try:
        dim = model.get_sentence_embedding_dimension()
        print(f"ℹ️  句向量維度：{dim}")
    except Exception:
        # 少數版本無 get_sentence_embedding_dimension() 時，透過一次 encode 推斷維度
        dim = None

    try:
        # 測試語句
        sentences: List[str] = [
            "柯文哲於2022年再度當選台北市市長，引發民眾熱烈討論。",
            "台北市市長的任期為四年。",
            "今天天氣不錯，適合外出走走。",
        ]

        print("🚀 [2/3] 開始進行向量嵌入（embedding）...")
        t1 = time.time()

        # convert_to_numpy=True 直接回傳 NumPy 陣列，便於後續保存或計算
        embeddings: np.ndarray = model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        enc_secs = time.time() - t1
        print(f"✅ 嵌入完成（耗時 {enc_secs:.2f} 秒）")

        # 輸出摘要
        print("🔍 [3/3] 嵌入結果摘要：")
        print(f"    ➤ 向量數量：{embeddings.shape[0]}")
        print(f"    ➤ 向量維度：{embeddings.shape[1]}")
        # 顯示第一筆向量前 5 維
        preview = np.array2string(embeddings[0][:5], precision=4, separator=", ")
        print(f"    ➤ 第一筆向量前 5 維：{preview}")

    except Exception as e:
        print("❌ 發生錯誤：")
        print(e)


if __name__ == "__main__":
    main()
