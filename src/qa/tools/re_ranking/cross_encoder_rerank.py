"""
Cross-Encoder Reranker（預設支援 BAAI/bge-reranker-v2-m3）
用途：
    - 對 (query, passage) 打分並輸出 0~1 機率（logits 經 sigmoid），方便與 RRF 融合。
依賴：transformers >= 4.35, torch
環境變數：
    - CE_MODEL（預設 "BAAI/bge-reranker-v2-m3"）
    - CE_BATCH_SIZE（預設 16）
    - CE_MAX_LENGTH（建議 1024；官方建議值）
    - BGE_RERANKER_DIR（本地模型根目錄，可指到 snapshots/<hash>/）
說明：Hugging Face 本地載入必須指向含 config.json 的模型根目錄。
"""
from __future__ import annotations
import os
import glob
import time
from typing import List, Sequence
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


class CrossEncoderReranker:
    """薄封裝：提供 .score_batch(query, passages) -> List[float]（0~1）"""

    def __init__(self, model_name: str | None = None,
                 batch_size: int = 16, max_length: int = 1024) -> None:
        # 來源優先序：BGE_RERANKER_DIR > 參數 > CE_MODEL（預設 hub 名稱）
        raw = os.getenv("BGE_RERANKER_DIR") or model_name or os.getenv("CE_MODEL", "BAAI/bge-reranker-v2-m3")
        self.model_name = self._resolve_model_dir(raw)
        self.batch_size = int(os.getenv("CE_BATCH_SIZE", batch_size))
        self.max_length = int(os.getenv("CE_MAX_LENGTH", max_length))

        t0 = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()
        print(f"[CrossEncoderReranker] loaded '{self.model_name}' on {self.device} "
              f"(bs={self.batch_size}, max_len={self.max_length}) in {time.time()-t0:.2f}s")

    @staticmethod
    def _resolve_model_dir(path_or_name: str) -> str:
        """若給的是本地最外層資料夾，嘗試自動下潛到 snapshots/<hash>/（含 config.json）"""
        p = str(path_or_name or "").strip()
        if not p or "://" in p or "/" not in p:
            return p  # hub 名稱或看起來已是有效路徑
        cfg = os.path.join(p, "config.json")
        if os.path.exists(cfg):
            return p
        # 掃 snapshots/*/config.json
        candidates = sorted(glob.glob(os.path.join(p, "snapshots", "*", "config.json")))
        if candidates:
            return os.path.dirname(candidates[-1])  # 取最新快照
        return p  # 讓 Transformers 去報更明確的錯（沒有 config.json）
    
    @torch.inference_mode()
    def score_batch(self, query: str, passages: Sequence[str]) -> List[float]:
        """對 (query, passage) 批次打分，回傳 sigmoid(logits) ∈ [0,1]。"""
        if not passages:
            return []
        out: List[float] = []
        q = str(query or "")
        total = (len(passages) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(passages), self.batch_size),
                      total=total, desc="Cross-Encoder", unit="batch"):
            batch = [str(p or "") for p in passages[i:i + self.batch_size]]
            enc = self.tokenizer([q] * len(batch), batch,
                                 padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits.squeeze(-1)  # [B]
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            out.extend(float(x) for x in probs)
        return out