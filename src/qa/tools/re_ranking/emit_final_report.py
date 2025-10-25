# -*- coding: utf-8-sig -*-
"""
Emit Final Report from CE-ranked results with topic-gate.

目的:
    - 讀取 CE 產物（.ce.jsonl），基於融合分數與主題詞 gate 篩選最終條目，
      產出易讀的 re-rank 報告（.txt）。
    - 主題詞來源優先使用「Multi-Query 問句 + 人工 curated」，必要時以新聞全文補齊。

特性:
    - 彈性門檻:
        keep if:
            final >= min_final
         or final >= mid_final and >= min_kws 命中
         or final >= min_final_lo and >= (min_kws + 1) 命中
      Topic gate:
        - require_topic=True 時，零命中者會被丟棄，除非 final >= keep_if_final_ge
    - 關鍵詞抽取支援:
        - 自動偵測 queries（JSON/TXT）並取詞（優先）
        - 由 --kws 人工清單加入（逗點分隔）
        - 必要時從 news 全文抽詞補齊（min_freq 可調）

依賴:
    - 僅標準庫

環境變數（可選）:
    CE_TOPK=100
    CE_MIN_FINAL=0.40
    CE_MID_FINAL=0.50
    CE_MIN_FINAL_LO=0.45
    CE_MIN_KWS=1
    CE_REQUIRE_TOPIC=1
    CE_KEEP_IF_FINAL_GE=0.62
    CE_KWS_MIN_FREQ=1

使用方式:
    python -m src.qa.tools.re_ranking.emit_final_report \
      --news data/processed/verifier/news_kg_xxx.txt \
      --ce   data/processed/verifier/re-ranking/Cross-Encoder/news_kg_xxx.ce.jsonl \
      --out  data/processed/verifier/news_kg_xxx_re-rank.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


# ----------------------------- 基礎工具 -----------------------------


def _dlog(msg: str) -> None:
    """Debug logger（立即 flush）"""
    print(msg, flush=True)


def _read_text(p: Path) -> str:
    """以 utf-8-sig 讀取文字檔。"""
    with p.open("r", encoding="utf-8-sig") as f:
        return f.read()


def _split_lines(s: str) -> List[str]:
    """分行並去除空白行。"""
    return [ln.strip() for ln in s.splitlines() if ln.strip()]


def _tokenize(text: str) -> List[str]:
    """極簡 tokenizer：保留中英文與數字，移除其餘符號。

    Args:
        text: 輸入文字。

    Returns:
        以空白切分的 token 清單。
    """
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text, flags=re.U)
    toks = [t.strip() for t in text.split() if t.strip()]
    return toks


# ----------------------------- 關鍵詞處理 -----------------------------


def _load_curated(kws: str) -> List[str]:
    """從 --kws 參數載入人工關鍵詞（逗號分隔），去重保序。"""
    if not kws:
        return []
    arr = [w.strip() for w in kws.split(",") if w.strip()]
    out, seen = [], set()
    for w in arr:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out


def extract_keywords(
    text: str,
    curated: Sequence[str] | None = None,
    min_freq: int = 1,
    max_k: int = 100,
) -> List[str]:
    """以詞頻抽取關鍵詞，並與人工詞彙合併。

    Args:
        text: 來源文字。
        curated: 人工關鍵詞。
        min_freq: 最小詞頻。
        max_k: 最多保留關鍵詞數。

    Returns:
        保序去重並截斷至 max_k 的關鍵詞清單。
    """
    freq = Counter()
    for tok in _tokenize(text):
        freq[tok] += 1
    common = [w for w, c in freq.items() if c >= min_freq]
    kws = list(dict.fromkeys([*(curated or []), *common]))[:max_k]
    return kws


def keyword_hits(text: str, keywords: Sequence[str]) -> Tuple[int, List[str]]:
    """計算命中關鍵詞。

    Args:
        text: 欲檢測的文字。
        keywords: 關鍵詞清單。

    Returns:
        (命中數, 命中詞片段清單)
    """
    hits: List[str] = []
    low = text.lower()
    for w in keywords:
        if w and w.lower() in low:
            hits.append(w)
    # 去重保序（避免同詞重覆加入）
    hits = list(dict.fromkeys(hits))
    return len(hits), hits

# ============================================================
# Auto-detect queries for this run (no env var required)
# ============================================================
def _id_scoped_dir(news_txt_path: Path) -> Path:
    """Return data/processed/verifier/re-ranking/<ID>/ where <ID> is news stem."""
    # 例如：news_kg_t_1022_agent.txt -> news_kg_t_1022_agent
    stem = news_txt_path.stem  # Path.stem：檔名去副檔名（官方定義）
    return Path("data/processed/verifier/re-ranking") / stem


def _try_read_queries_txt(p: Path) -> List[str]:
    """Read queries from a .txt file (one per line), utf-8-sig tolerant."""
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8-sig") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _try_read_queries_json(p: Path) -> List[str]:
    """Read queries from a .json file with schema: {"queries": [...]}."""
    if not p.is_file():
        return []
    try:
        with p.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("queries"), list):
            return [str(x).strip() for x in data["queries"] if str(x).strip()]
    except Exception:
        return []
    return []


def _auto_load_queries_for_news(news_txt_path: Path) -> List[str]:
    """Auto-detect queries under <re-ranking>/<ID>/{queries.txt,queries.json}.

    搜尋順序：
      1) <ID>/queries.txt
      2) <ID>/queries.json
    任一讀到非空即返回；都讀不到則回空清單。
    """
    base = _id_scoped_dir(news_txt_path)
    txt_path = base / "queries.txt"
    jsn_path = base / "queries.json"
    q = _try_read_queries_txt(txt_path)
    if q:
        _dlog(f"[Queries] auto-loaded TXT: {txt_path} ({len(q)})")
        return q
    q = _try_read_queries_json(jsn_path)
    if q:
        _dlog(f"[Queries] auto-loaded JSON: {jsn_path} ({len(q)})")
        return q
    _dlog(f"[Queries] no queries file found under: {base}")
    return []


# ----------------------------- 查詢檔自動偵測 -----------------------------


def _id_scoped_dir(news_txt_path: Path) -> Path:
    """取得 `data/processed/verifier/re-ranking/<ID>/` 目錄。

    <ID> 為 `--news` 檔名的 stem（去副檔名）。Path.stem 定義見官方文件。"""
    # 例如：news_kg_t_1022_agent.txt -> news_kg_t_1022_agent
    stem = news_txt_path.stem  # 官方定義：回傳不含副檔名的檔名
    return Path("data/processed/verifier/re-ranking") / stem


def _try_read_queries_txt(p: Path) -> List[str]:
    """讀取 queries.txt（每行一問句），容忍 BOM。"""
    if not p.is_file():
        return []
    return _split_lines(_read_text(p))


def _try_read_queries_json(p: Path) -> List[str]:
    """讀取 queries.json，schema: {"queries": [...]}。"""
    if not p.is_file():
        return []
    try:
        payload = json.loads(_read_text(p))
        if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
            return [str(q).strip() for q in payload["queries"] if str(q).strip()]
    except Exception:
        return []
    return []


def _auto_load_queries_for_news(news_txt_path: Path) -> List[str]:
    """自動偵測 <ID>/queries.{txt,json}（先 txt 後 json）。"""
    base = _id_scoped_dir(news_txt_path)
    q_txt = base / "queries.txt"
    q_jsn = base / "queries.json"
    q = _try_read_queries_txt(q_txt)
    if q:
        _dlog(f"[Queries] auto-loaded TXT: {q_txt} ({len(q)})")
        return q
    q = _try_read_queries_json(q_jsn)
    if q:
        _dlog(f"[Queries] auto-loaded JSON: {q_jsn} ({len(q)})")
        return q
    _dlog(f"[Queries] not found under: {base}")
    return []


# ----------------------------- 主體流程 -----------------------------


def filter_items(
    items: Sequence[Dict[str, Any]],
    keywords: Sequence[str],
    min_final: float,
    min_kws: int,
    min_final_lo: float,
    topk: int,
    require_topic: bool,
    keep_if_final_ge: float,
    mid_final: float,
) -> List[Dict[str, Any]]:
    """依分數與主題詞 gate 篩選條目。

    保留條件:
        - final >= min_final；或
        - final >= mid_final 且 >= min_kws 命中；或
        - final >= min_final_lo 且 >= (min_kws + 1) 命中。
      Topic gate:
        - require_topic=True 時，命中數為 0 的條目將被過濾，
          除非 final >= keep_if_final_ge。

    Args:
        items: CE 產物列。
        keywords: 主題關鍵詞。
        min_final: 高段門檻。
        min_kws: 中段所需的最少命中。
        min_final_lo: 低段門檻。
        topk: 最多保留條目數。
        require_topic: 是否強制命中主題詞。
        keep_if_final_ge: 高分保底門檻。
        mid_final: 中段門檻。

    Returns:
        通過篩選的條目清單（保留原順序，至多 topk）。
    """
    kept: List[Dict[str, Any]] = []
    for it in items:
        txt = str(it.get("text", ""))
        final = float(it.get("final_score", 0.0))
        hit_n, _ = keyword_hits(txt, keywords)

        if require_topic and hit_n == 0 and final < keep_if_final_ge:
            continue
        if final >= min_final:
            kept.append(it)
        elif final >= mid_final and hit_n >= min_kws:
            kept.append(it)
        elif final >= min_final_lo and hit_n >= (min_kws + 1):
            kept.append(it)

        if len(kept) >= topk:
            break
    return kept


def build_argparser() -> argparse.ArgumentParser:
    """建立 CLI 參數解析器。"""
    ap = argparse.ArgumentParser("Emit final report from CE results")
    ap.add_argument("--news", type=str, required=True, help="原始新聞 .txt（作為關鍵詞抽取備援）")
    ap.add_argument("--ce", type=str, required=True, help=".ce.jsonl 檔案路徑")
    ap.add_argument("--out", type=str, required=True, help="輸出 .txt 路徑")
    ap.add_argument("--kws", type=str, default="", help="人工關鍵詞（逗點分隔，優先）")
    # 仍保留 CLI 指定 queries 檔（選用）；若未指定，將自動偵測 <ID>/queries.{txt,json}
    ap.add_argument("--queries-txt", type=str, default="")
    ap.add_argument("--queries-json", type=str, default="")

    ap.add_argument("--topk", type=int, default=int(os.getenv("CE_TOPK", "100")))
    ap.add_argument("--min-final", type=float, default=float(os.getenv("CE_MIN_FINAL", "0.40")))
    ap.add_argument(
        "--mid-final", type=float, default=float(os.getenv("CE_MID_FINAL", "0.50")),
        help="中段門檻，需搭配關鍵詞命中"
    )
    ap.add_argument("--min-kws", type=int, default=int(os.getenv("CE_MIN_KWS", "1")))
    ap.add_argument("--min-final-lo", type=float, default=float(os.getenv("CE_MIN_FINAL_LO", "0.45")))
    ap.add_argument("--require-topic", type=int, default=int(os.getenv("CE_REQUIRE_TOPIC", "1")))
    ap.add_argument("--keep-if-final-ge", type=float, default=float(os.getenv("CE_KEEP_IF_FINAL_GE", "0.62")))
    return ap


def main() -> int:
    """主程式入口。"""
    args = build_argparser().parse_args()

    news_p = Path(args.news)
    ce_p = Path(args.ce)
    out_p = Path(args.out)

    news_p = Path(args.news)
    ce_p = Path(args.ce)
    out_p = Path(args.out)

    # 1) 載入 CE 結果（jsonl，每行一物件；newline-delimited JSON）
    rows: List[Dict[str, Any]] = []
    with ce_p.open("r", encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    _dlog(f"[emit] ce_rows={len(rows)}")

    # 2) 建立主題詞來源（優先序：CLI 指定 queries > 自動偵測 > --kws > news 全文）
    curated = _load_curated(args.kws)
    qlist: List[str] = []
    if args.queries_txt:
        qlist = _try_read_queries_txt(Path(args.queries_txt))
    elif args.queries_json:
        qlist = _try_read_queries_json(Path(args.queries_json))
    else:
        qlist = _auto_load_queries_for_news(news_p)

    base_text = "\n".join(qlist) if qlist else _read_text(news_p)
    min_freq = int(os.getenv("CE_KWS_MIN_FREQ", "1"))
    keywords = extract_keywords(base_text, curated=curated, min_freq=min_freq)
    _dlog(f"[emit] keywords={len(keywords)} samples={keywords[:12]}")

    # 3) 依分數與關鍵詞 gate 篩選
    kept = filter_items(
        rows,
        keywords=keywords,
        min_final=args.min_final,
        min_kws=args.min_kws,
        min_final_lo=args.min_final_lo,
        topk=args.topk,
        require_topic=bool(args.require_topic),
        keep_if_final_ge=args.keep_if_final_ge,
        mid_final=args.mid_final,
    )

    # 4) 輸出報告（簡明格式；可依需求客製化）
    out_lines: List[str] = []
    for i, it in enumerate(kept, 1):
        txt = str(it.get("text", "")).strip()
        rrf = float(it.get("rrf_score", 0.0))
        ce = float(it.get("ce_score", 0.0))
        fin = float(it.get("final_score", 0.0))
        hit_n, hit_w = keyword_hits(txt, keywords)
        out_lines.append(
            f"{i:02d}. final={fin:.4f} | ce={ce:.4f} | rrf={rrf:.4f} | "
            f"kws_hit={hit_n} [{', '.join(hit_w[:6])}]"
        )
        out_lines.append(txt)
        out_lines.append("-" * 80)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8-sig") as f:
        if out_lines:
            f.write("\n".join(out_lines).rstrip() + "\n")
        else:
            f.write("(no items kept)\n")

    _dlog(f"[emit] wrote: {out_p} (kept={len(kept)}/{len(rows)}) | kws={len(keywords)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
