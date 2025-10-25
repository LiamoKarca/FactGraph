"""
Cross-Encoder Re-Ranker with Multi-Query support.

目的:
    - 針對前一階段的候選清單（通常是經 RRF/Hybrid 排序的 Top-N），
      以 Cross-Encoder 對 (query, passage) 配對打分，並輸出融合分數。
    - 查詢 (query) 來源支援:
        1) 透過 LLM 由「原始新聞文本」拆出 n 句具體問句（推薦）
        2) 由 JSON/TXT 檔載入 queries
        3) 由 CLI --query 或檔名推測（fallback）

主要特性:
    - Multi-Query: 對每個候選 passage，以所有 queries 的 CE 分數取最大值作為該候選 CE 分數。
    - 分數融合: final = alpha * CE + (1 - alpha) * RRF
    - 可設定輸出路徑:
        RRF 產物:   data/processed/verifier/re-ranking/RRF/<basename>.rrf.jsonl
        CE 產物:    data/processed/verifier/re-ranking/Cross-Encoder/<basename>.ce.jsonl

依賴:
    - transformers >= 4.35（若使用本地 CE 模型）
    - torch（若使用本地 CE 模型）
    - openai >= 1.0.0（若開啟 LLM 產生查詢）
    - 已存在的候選來源（可由 .txt 解析或由 .rrf.jsonl 載入）

環境變數（可選）:
    # Multi-Query / LLM
    QUERY_LLM=gpt-4o-mini
    CE_NQ=5
    QUERY_PROMPT=...（可含 {n} 與 {news}）

    # RRF x CE 融合權重（同名相容）
    CE_ALPHA=0.5
    RRF_CE_FUSE_ALPHA=0.5

    # RRF 權重（相容舊名）
    RRF_K=60
    W_PG=1.0 / RRF_W_PG
    W_VEC=1.0 / RRF_W_VEC
    W_NEO=1.0 / RRF_W_NEO4J or RRF_W_NEO

    # CE 設定
    CE_MODEL=BAAI/bge-reranker-v2-m3
    CE_TOPN=120
    CE_MAX_LENGTH=1024
    CE_BATCH_SIZE=16

使用方式（範例）:
    python -m src.qa.tools.re_ranking.rerank_from_recall \
      --txt data/processed/verifier/news_kg_t_1022_agent.txt

    # 指定 queries JSON
    python -m src.qa.tools.re_ranking.rerank_from_recall \
      --txt data/processed/verifier/news_kg_xxx.txt \
      --queries-json data/processed/verifier/re-ranking/queries_xxx.json

注意:
    - 若要讓 RRF 也吃 Multi-Query，需要上游各檢索器對每個 query 各自召回並做跨查詢 RRF 融合；
      本模組先聚焦在 CE Multi-Query 與融合輸出。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .query_decomposer import decompose_queries, QDConfig  # 新增: 本地模組


# 嘗試使用專案內封裝的 CE；若不可用則退回本檔內簡單包裝
try:
    from .cross_encoder_rerank import CrossEncoderReranker  # type: ignore
except Exception:
    CrossEncoderReranker = None  # 由 _LocalCrossEncoderReranker 代管

# -------------------------------
# 基礎工具
# -------------------------------


def _dlog(msg: str) -> None:
    """Lightweight debug logger."""
    print(msg, file=sys.stderr, flush=True)


def _env_str(name: str, default_v: str = "") -> str:
    v = os.getenv(name, "").strip()
    return v if v else default_v


def _env_int(name: str, default_v: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default_v
    try:
        return int(v)
    except Exception:
        return default_v


def _env_float(name: str, default_v: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return default_v
    try:
        return float(v)
    except Exception:
        return default_v


def _env_float_multi(names: Sequence[str], default_v: float) -> float:
    """Return the first present env var in names as float, else default."""
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            try:
                return float(v)
            except Exception:
                continue
    return default_v


def _env_weight(name_primary: str, fallback_names: Sequence[str], default_v: float) -> float:
    """Support legacy/new env names for weights."""
    v = os.getenv(name_primary, "").strip()
    if v:
        try:
            return float(v)
        except Exception:
            pass
    for n in fallback_names:
        v2 = os.getenv(n, "").strip()
        if v2:
            try:
                return float(v2)
            except Exception:
                continue
    return default_v


def _read_text(p: Union[str, Path]) -> str:
    with open(p, "r", encoding="utf-8-sig") as f:
        return f.read()


def _write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _split_lines(s: str) -> List[str]:
    return [ln.strip() for ln in s.splitlines() if ln.strip()]


# -------------------------------
# 候選載入與簡易 RRF（保留原生整合）
# -------------------------------


@dataclass
class Candidate:
    """單一候選段落，含多路檢索得分與文本。"""

    text: str
    rank_pg: Optional[int] = None
    rank_vec: Optional[int] = None
    rank_neo: Optional[int] = None
    score_rrf: float = 0.0
    score_ce: float = 0.0
    score_final: float = 0.0
    meta: Dict[str, Any] = None  # 可放來源、路徑等


def _parse_candidates_from_txt(txt: str) -> List[Candidate]:
    """
    從 .txt 解析候選。
    支援格式：
      A) 你的專案輸出格式：
         [原始文本]
         ...
         [比對知識]
         [1] ...（可能跨行）
         [2] ...
         ...
      B) 通用備援：以「編號/score」起始的清單、或以空行分段。
    """
    lines = _split_lines(txt)
    # --- A) 專案格式：[比對知識] 區塊 + [\d+] 條目 ---
    try:
        start = None
        for i, ln in enumerate(lines):
            if ln.strip() == "[比對知識]":
                start = i + 1
                break
        if start is not None and start < len(lines):
            pat_item = re.compile(r"^\[\s*\d+\s*\]")  # 例如 "[1] ..."
            items: List[str] = []
            buf: List[str] = []
            for ln in lines[start:]:
                if pat_item.match(ln):
                    # 遇到新條目，先收前一條
                    if buf:
                        items.append("\n".join(buf).strip())
                        buf = []
                    buf.append(ln)  # 含開頭 [n]
                else:
                    if buf:
                        buf.append(ln)
            if buf:
                items.append("\n".join(buf).strip())
            # 清理每條：去掉開頭編號
            if items:
                cands = []
                for it in items:
                    text_clean = re.sub(r"^\[\s*\d+\s*\]\s*", "", it).strip()
                    if text_clean:
                        cands.append(Candidate(text=text_clean, meta={}))
                if cands:
                    return cands
    except Exception:
        # 不中斷，落回備援解析
        pass

    # --- B) 備援：以「數字/score 起始」或空行分段 ---
    blocks: List[str] = []
    buf: List[str] = []
    pat_head = re.compile(r"^\s*\d+[\.\)]\s+|^\s*score\s*=\s*[-+]?\d+(\.\d+)?", re.I)
    for ln in lines:
        if pat_head.search(ln) and buf:
            blocks.append("\n".join(buf).strip())
            buf = [ln]
        else:
            # 以明顯分隔線或空行切段
            if ln.strip("-=._ ") == "" and buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            else:
                buf.append(ln)
    if buf:
        blocks.append("\n".join(buf).strip())

    cands: List[Candidate] = []
    for b in blocks:
        b2 = re.sub(r"^\s*(\d+[\.\)]\s+)?(score\s*=\s*[-+]?\d+(\.\d+)?\s*\|\s*)?", "", b, flags=re.I)
        txt_clean = b2.strip()
        if txt_clean:
            cands.append(Candidate(text=txt_clean, meta={}))
    if not cands:
        cands.append(Candidate(text="\n".join(lines).strip(), meta={}))
    return cands


def _rrf_score(rank: Optional[int], k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


def _fuse_rrf(cands: List[Candidate], k: int,
              w_pg: float, w_vec: float, w_neo: float) -> List[Candidate]:
    """
    使用 rank-based RRF 融合（穩健基線）。若無 rank，視為不貢獻。
    這裡假設 rank_* 已由上游標註；若上游未提供，簡單用原始順序近似 rank。
    """
    # 若沒有 rank，回退為以現有順序標 rank
    for i, c in enumerate(cands):
        if c.rank_pg is None and c.rank_vec is None and c.rank_neo is None:
            c.rank_pg = i + 1

    for c in cands:
        s = 0.0
        if c.rank_pg is not None:
            s += w_pg * _rrf_score(c.rank_pg, k)
        if c.rank_vec is not None:
            s += w_vec * _rrf_score(c.rank_vec, k)
        if c.rank_neo is not None:
            s += w_neo * _rrf_score(c.rank_neo, k)
        c.score_rrf = s

    # 依 RRF 由高到低排序
    cands.sort(key=lambda x: x.score_rrf, reverse=True)
    return cands


# -------------------------------
# CE 封裝
# -------------------------------


class _LocalCrossEncoderReranker:
    """
    本地簡易 CE 包裝：以 huggingface 的 cross-encoder（BGE v2-m3）做 0..1 打分。

    若專案已提供 cross_encoder_rerank.CrossEncoderReranker（上方 import 成功），
    會優先使用該版本。
    """

    def __init__(self,
                 model_name: str = "BAAI/bge-reranker-v2-m3",
                 max_length: int = 1024,
                 batch_size: int = 16) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        # 延遲載入，避免在不需要 CE 時帶來依賴
        self._pipe = None

    def _ensure(self) -> None:
        if self._pipe is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._mdl.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mdl.to(self._device)

    def score_batch(self, query: str, passages: Sequence[str]) -> List[float]:
        """
        對 (query, passage) 配對批次打分，輸出 0..1 之間的相似度（對 logits 做 sigmoid）。
        """
        self._ensure()
        import torch
        from torch.nn.functional import sigmoid

        scores: List[float] = []
        B = self.batch_size
        for i in range(0, len(passages), B):
            batch_psg = passages[i:i + B]
            pairs = [(query, p) for p in batch_psg]
            enc = self._tok.batch_encode_plus(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            for k in enc:
                enc[k] = enc[k].to(self._device)
            with torch.no_grad():
                out = self._mdl(**enc)
                logits = out.logits.squeeze(-1)  # shape: [B]
                probs = sigmoid(logits).detach().cpu().tolist()
                scores.extend([float(x) for x in probs])
        return scores


def _get_ce() -> Any:
    """
    取得 CE 物件：
      - 有專案版 CrossEncoderReranker 就用之
      - 否則使用本地 _LocalCrossEncoderReranker
    """
    ce_model = _env_str("CE_MODEL", "BAAI/bge-reranker-v2-m3")
    ce_maxlen = _env_int("CE_MAX_LENGTH", 1024)
    ce_bs = _env_int("CE_BATCH_SIZE", 16)

    if CrossEncoderReranker is not None:
        try:
            return CrossEncoderReranker(model_name=ce_model, max_length=ce_maxlen, batch_size=ce_bs)  # type: ignore
        except Exception as e:
            _dlog(f"[CE] 專案版初始化失敗，改用內建 CE：{e}")
            return _LocalCrossEncoderReranker(ce_model, ce_maxlen, ce_bs)
    return _LocalCrossEncoderReranker(ce_model, ce_maxlen, ce_bs)


# -------------------------------
# LLM 產生多問句（Multi-Query）
# -------------------------------


def _llm_generate_queries(text: str, n: int, model: str, prompt: Optional[str] = None) -> List[str]:
    """
    呼叫 LLM 將輸入文本拆成 n 句具體問句。
    失敗時回傳空清單，不中斷主流程。

    Args:
        text: 輸入新聞或摘要。
        n: 期望問句數。
        model: LLM 名稱（如 gpt-4o-mini）。
        prompt: 自訂提示（可包含 {n} 與 {news}）。

    Returns:
        去重後的問句清單（最多 n 句）。
    """
    try:
        from openai import OpenAI  # 官方 SDK
    except Exception as e:  # pragma: no cover
        _dlog(f"[LLM] OpenAI SDK 不可用：{e}")
        return []

    sys_t = (
        "任務：將輸入文本拆成多個具體、可核查的中文問句。"
        "每個問句都要明確涉及主體、行為、時間或來源脈絡，避免空泛。"
        "輸出使用純文字，一行一問句，不要編號。"
    )
    usr_t = prompt or (
        "請將以下文本拆成 {n} 句越具體越好的問句，分別聚焦：主詞、行為、時間、資金流向、來源與證據鏈、涉入單位或個人。"
        "每行一問句，不要編號：\n\n{news}\n"
    )
    content = usr_t.format(n=n, news=textwrap.dedent(text).strip())
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_t}, {"role": "user", "content": content}],
            temperature=0.2,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # pragma: no cover
        _dlog(f"[LLM] 產生 queries 失敗：{e}")
        return []

    raw = _split_lines(out)
    seen, qs = set(), []
    for q in raw:
        if q not in seen:
            qs.append(q)
            seen.add(q)
        if len(qs) >= n:
            break
    return qs


# -------------------------------
# 主流程
# -------------------------------


def infer_query_from_news(txt_path: Union[str, Path]) -> str:
    """
    從檔名或新聞首句粗略推測單一查詢（Fallback）。
    建議僅作保底；主流程優先使用 Multi-Query。
    """
    name = Path(txt_path).stem
    news = _read_text(txt_path)
    first_line = _split_lines(news)[:1]
    if first_line:
        return first_line[0]
    return name


def _load_candidates(txt_path: Union[str, Path], rrf_jsonl: Optional[str] = None) -> List[Candidate]:
    """
    若提供 rrf_jsonl，優先讀取該 JSONL（每行需含 text、可選 rank/score）。
    否則從 <news>.txt 嘗試解析候選。可視需要調整解析器。
    """
    if rrf_jsonl and Path(rrf_jsonl).exists():
        rows = [json.loads(ln) for ln in _read_text(rrf_jsonl).splitlines() if ln.strip()]
        cands: List[Candidate] = []
        for r in rows:
            c = Candidate(text=r.get("text", ""), meta=r.get("meta", {}) or {})
            # 若上游已有 rank，可帶上以利 RRF 或診斷
            c.rank_pg = r.get("rank_pg")
            c.rank_vec = r.get("rank_vec")
            c.rank_neo = r.get("rank_neo")
            c.score_rrf = float(r.get("score_rrf", 0.0))
            cands.append(c)
        _dlog(f"Loaded candidates from RRF jsonl: {len(cands)}")
        return cands
    # fallback: 從 txt 解析
    txt = _read_text(txt_path)
    cands = _parse_candidates_from_txt(txt)
    _dlog(f"Loaded candidates from txt: {len(cands)}")
    return cands


def _ensure_out_paths(news_txt_path: Union[str, Path]) -> Tuple[Path, Path]:
    """
    組出預設輸出路徑：
        RRF: data/processed/verifier/re-ranking/RRF/<basename>.rrf.jsonl
        CE:  data/processed/verifier/re-ranking/Cross-Encoder/<basename>.ce.jsonl
    """
    base = Path(news_txt_path).name  # e.g., news_kg_t_1022_agent.txt
    stem = base.replace(".txt", "")
    rrf_out = Path("data/processed/verifier/re-ranking/RRF") / f"{stem}.rrf.jsonl"
    ce_out = Path("data/processed/verifier/re-ranking/Cross-Encoder") / f"{stem}.ce.jsonl"
    return rrf_out, ce_out

def _ensure_query_id_dir(news_txt_path: Union[str, Path]) -> Path:
    """Ensure an ID-scoped directory for this run.

    目標目錄：
        data/processed/verifier/re-ranking/<ID>/
    其中 <ID> 為輸入 .txt 的檔名（去掉副檔名）。
    """
    stem = Path(news_txt_path).stem
    out_dir = Path("data/processed/verifier/re-ranking") / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _persist_queries_files(news_txt_path: Union[str, Path], queries: Sequence[str]) -> Tuple[Path, Path]:
    """Write queries to both TXT and JSON under the ID-scoped directory.

    會輸出兩個檔案：
      - queries.txt  ：每行一問句
      - queries.json ：{"queries": [...]}

    Args:
        news_txt_path: 輸入新聞 .txt 路徑
        queries: 問句清單

    Returns:
        (txt_path, json_path)
    """
    out_dir = _ensure_query_id_dir(news_txt_path)
    txt_p = out_dir / "queries.txt"
    json_p = out_dir / "queries.json"

    # 寫 TXT
    with txt_p.open("w", encoding="utf-8-sig", newline="\n") as f:
        for q in queries:
            q = str(q).strip()
            if q:
                f.write(q + "\n")

    # 寫 JSON
    payload = {"queries": [str(q).strip() for q in queries if str(q).strip()]}
    with json_p.open("w", encoding="utf-8-sig") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return txt_p, json_p


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Cross-Encoder reranker with Multi-Query")
    parser.add_argument("--txt", type=str, required=True, help="原始新聞/候選輸入的 .txt 檔")
    parser.add_argument("--rrf-in", type=str, default="", help="直接從 RRF JSONL 載入候選（優先）")    
    parser.add_argument("--query", type=str, default="", help="單一查詢（若提供則覆蓋推測/LLM）")
    parser.add_argument("--queries-json", type=str, default="", help='讀取 {"queries":[...]} JSON 檔')
    parser.add_argument("--queries-txt", type=str, default="", help="讀取每行一問句的 TXT 檔")
    
    # --- Query Decomposition (QD) 相關 ---
    parser.add_argument("--qd", action="store_true",
                        help="啟用 Query Decomposition（使用 gpt-4o-mini 拆分子問題）")
    parser.add_argument("--qd-use-splits", action="store_true",
                        help="CE 打分時納入拆分後的子問題（原查詢 + 子問題 一起送 CE）")
    parser.add_argument("--qd-model", type=str, default=_env_str("QD_MODEL", "gpt-4o-mini"))
    parser.add_argument("--qd-max-splits", type=int, default=_env_int("QD_MAX_SPLITS", 6))
    parser.add_argument("--qd-min-splits", type=int, default=_env_int("QD_MIN_SPLITS", 3))
    parser.add_argument("--qd-style", type=str, default=_env_str("QD_STYLE", "factual-tight"))

    # Multi-Query 相關
    parser.add_argument("--nq", type=int, default=_env_int("CE_NQ", 5), help="LLM 產生的問句數")
    parser.add_argument("--llm", type=str, default=_env_str("QUERY_LLM", "gpt-4o-mini"),
                        help="用於產生查詢句的 LLM 名稱")
    parser.add_argument("--query-prompt", type=str, default=_env_str("QUERY_PROMPT", ""),
                        help="自訂提示詞（可含 {n} 與 {news}）")

    # RRF 權重
    parser.add_argument("--rrf-k", type=int, default=_env_int("RRF_K", 60), help="RRF k（預設 60）")
    parser.add_argument("--w-pg", type=float, default=_env_weight("W_PG", ["RRF_W_PG"], 1.0),
                        help="RRF 權重：pg")
    parser.add_argument("--w-vec", type=float, default=_env_weight("W_VEC", ["RRF_W_VEC"], 1.0),
                        help="RRF 權重：vec")
    parser.add_argument("--w-neo", type=float, default=_env_weight("W_NEO", ["RRF_W_NEO4J", "RRF_W_NEO"], 1.0),
                        help="RRF 權重：neo")

    # CE 設定與融合
    parser.add_argument("--topn", type=int, default=_env_int("CE_TOPN", 120), help="進入 CE 的候選數上限")
    parser.add_argument("--alpha", type=float,
                        default=_env_float_multi(["CE_ALPHA", "RRF_CE_FUSE_ALPHA"], 0.5),
                        help="融合權重 α（CE 權重）")
    parser.add_argument("--ce-model", type=str, default=_env_str("CE_MODEL", "BAAI/bge-reranker-v2-m3"),
                        help="CE 模型名稱（相容環境變數）")
    parser.add_argument("--ce-maxlen", type=int, default=_env_int("CE_MAX_LENGTH", 1024),
                        help="CE 最長序列長度（建議 1024）")
    parser.add_argument("--ce-batch", type=int, default=_env_int("CE_BATCH_SIZE", 16),
                        help="CE 批次大小")

    return parser


def main() -> int:
    args = build_argparser().parse_args()

    # 1) 載入候選（此處從 .txt 解析；如有標準化 JSONL 可自行切換）
    cands = _load_candidates(args.txt, rrf_jsonl=args.rrf_in)

    # 2) RRF 融合（若無 rank_* 則以原始順序近似）
    cands = _fuse_rrf(cands, k=args.rrf_k, w_pg=args.w_pg, w_vec=args.w_vec, w_neo=args.w_neo)
    _dlog(f"RRF: fused={len(cands)} (k={args.rrf_k}, w_pg={args.w_pg}, w_vec={args.w_vec}, w_neo={args.w_neo})")

    # 2.1 輸出 RRF JSONL（便於診斷/留檔）
    rrf_out, ce_out = _ensure_out_paths(args.txt)
    rrf_rows = []
    for idx, c in enumerate(cands, 1):
        rrf_rows.append({
            "rank": idx,
            "text": c.text,
            "score_rrf": c.score_rrf,
            "meta": c.meta or {},
        })
    _write_jsonl(rrf_out, rrf_rows)
    _dlog(f"[RRF] wrote: {rrf_out}")

    # 3) Multi-Query 取得
    queries: List[str] = []
    if args.queries_json and Path(args.queries_json).exists():
        try:
            payload = json.loads(_read_text(args.queries_json))
            queries = [str(q).strip() for q in payload.get("queries", []) if str(q).strip()]
        except Exception as e:
            _dlog(f"[Queries] 讀取 JSON 失敗：{e}")
    elif args.queries_txt and Path(args.queries_txt).exists():
        queries = [ln.strip() for ln in _split_lines(_read_text(args.queries_txt))]

    if not queries:
        if args.query:
            queries = [args.query.strip()]
        else:
            # 使用 LLM 生成問句；失敗則回退單一 query
            if args.nq > 0 and args.llm:
                news_txt = _read_text(args.txt)
                queries = _llm_generate_queries(news_txt, n=args.nq, model=args.llm,
                                                prompt=(args.query_prompt or None))
            if not queries:
                queries = [infer_query_from_news(args.txt)]

    _dlog(f"Multi-Query enabled: {len(queries)} queries")
    for i, q in enumerate(queries, 1):
        _dlog(f"Q{i}: {q}")

    
    # 3.1) 將產生/取得的 queries 落盤，方便後續 emit_final_report 使用
    try:
        q_txt_p, q_json_p = _persist_queries_files(args.txt, queries)
        _dlog(f"[Queries] saved: {q_txt_p} ; {q_json_p}")
        _dlog("  提示：可直接使用 CE_QUERIES_TXT={} 執行 emit_final_report".format(q_txt_p))
    except Exception as e:  # pragma: no cover
        _dlog(f"[Queries] 無法寫入 queries 檔案：{e}")
        
    # 3.2) （可選）Query Decomposition：把每條 query 再拆成子問題，輸出到同一目錄
    enable_qd = args.qd or (os.getenv("ENABLE_QD", "0") == "1")
    qd_splits: List[str] = []
    if enable_qd:
        try:
            qd_cfg = QDConfig(
                model=args.qd_model,
                max_splits=args.qd_max_splits,
                min_splits=args.qd_min_splits,
                style=args.qd_style,
            )
            re_rank_dir = _ensure_query_id_dir(args.txt)
            recs, qd_splits = decompose_queries(queries, out_dir=re_rank_dir, cfg=qd_cfg)
            _dlog(f"[QD] queries={len(queries)} -> splits_total={len(qd_splits)} | model={qd_cfg.model}")
            print(f"[QD] wrote: {re_rank_dir/'queries.split.json'} ; {re_rank_dir/'queries.split.txt'}")
        except Exception as e:
            _dlog(f"[QD] 發生例外，跳過 QD：{e}")
    # 4) 取 TopN 進 CE
    topn = min(args.topn, len(cands))
    passages = [cands[i].text for i in range(topn)]

    # 5) 交叉編碼器打分（對每 query 都打一輪，最後取 max）
    #    取得 CE 物件時覆蓋參數（對專案版 CE 以 cfg 設置）
    ce = _get_ce()

    # CE 查詢清單：預設使用原 queries；若指定，併入子問題
    ce_query_list = list(queries)
    if args.qd_use_splits and qd_splits:
        seen_q = set()
        merged: List[str] = []
        for q in [*queries, *qd_splits]:
            q = str(q).strip()
            if q and q not in seen_q:
                seen_q.add(q)
                merged.append(q)
        ce_query_list = merged

    ce_scores_all: List[List[float]] = []
    for q in ce_query_list:
        ce_scores_all.append(ce.score_batch(q, passages))
    # zip 聚合後取 max
    ce_scores: List[float] = [max(scores) for scores in zip(*ce_scores_all)]

    # 6) 融合分數與排序
    alpha = args.alpha
    for i in range(topn):
        cands[i].score_ce = ce_scores[i]
        cands[i].score_final = alpha * cands[i].score_ce + (1.0 - alpha) * cands[i].score_rrf
    cands[:topn] = sorted(cands[:topn], key=lambda x: x.score_final, reverse=True)

    _dlog(
        "CE: model={} queries={} topn={} alpha={} scored={} | env_topn={} env_alpha={}".format(
            args.ce_model,
            len(ce_query_list),
            topn,
            alpha,
            len(ce_scores),
            os.getenv("CE_TOPN", ""),
            (os.getenv("CE_ALPHA", "") or os.getenv("RRF_CE_FUSE_ALPHA", "")),
        )
    )

    # 7) 輸出 CE JSONL
    rows = []
    for idx, c in enumerate(cands[:topn], 1):
        rows.append({
            "rank": idx,
            "text": c.text,
            "rrf_score": round(c.score_rrf, 6),
            "ce_score": round(c.score_ce, 6),
            "final_score": round(c.score_final, 6),
            "meta": c.meta or {},
        })
    _write_jsonl(ce_out, rows)
    _dlog(f"[CE] wrote: {ce_out} (rows={len(rows)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
