"""
注意：此檔為尚未重構前的原始檔案！

ReAct 檢索代理（LangGraph）

用法
----
python -m src.qa.verifier.agent.agent_langchain <news_id|檔名>

特性
----
1) 保證 PG 可用：先嘗試載入快取（PKL/JSON），失敗則記錄 traceback 並在工具回傳 error 與 hint。
2) 工具錯誤不吞：所有工具皆以 JSON 回覆錯誤訊息，並寫入 debug log。
3) ReAct＋LangGraph：每輪強制使用工具，合併去重，達到耐心值或步數上限後結束（不再以「最低條件」卡住）。
4) 三元組鍵名正規化：容忍多種鍵名（head/h/source... 等）。
5) StructuredTool 調用修正：保底階段直接呼叫內部實作，避免再次透過工具層。
6) 結果輸出：[原始文本] + [比對知識]，格式回復完整資訊（type/屬性/關係名/事件時間/說明）。
7) 工具入參支援 {"triples": [...]}；累積器可解析工具 JSON 的 {"lines":[...]}。

環境變數
--------
- VERIFIER_DEBUG: "1" 啟用除錯日誌（預設 0）
- VERIFIER_DEBUG_PATH: 除錯日誌路徑（預設 data/processed/verifier/debug/verifier_search_debug.log）
- AGENT_MAX_STEPS: 最大步數（預設 24）
- AGENT_NO_NEW_PATIENCE: 連續無新增步數耐心值（預設 5）
- MAX_AGENT_CHARS: 單篇輸入最大長度（預設 15000）
- MAX_EVID_CHARS: 單條 evidence 顯示上限（預設 300）
- AGENT_TOP_K_MAX: 最終輸出上限（預設 200；「只有上限，無下限」）
- ENABLE_LI_PG: 是否啟用 PG 檢索（預設 1）
- LI_PG_INDEX_JSON: PG 快取路徑
- ENABLE_LI_ONLINE: 是否啟用 Neo4j 線上檢索（0=停用；1=啟用）
- OPENAI_CHAT_MODEL: LangChain 用的對話模型 id（預設 gpt-4o-mini）
- LI_PG_TOPK, LI_PG_HOPS：以 .env 覆蓋 PG 工具的 top_k/hops
- ALIAS_RULES_JSON / ALIAS_RELOAD_SEC：由 retriever 支援（此檔提供 alias 子命令代理）
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
import traceback
import hashlib

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict, Set, Optional
from uuid import uuid4
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError

# 專案模組
from ..core.dedup import deduplicate
from ..core.embeddings import embed_triple
from ..core.paths import RES_DIR, USER_INPUT_DIR
from ..kg.search import cosine_search, kg_row_to_detail
from ..llm.extract import extract_entities_relations

load_dotenv(override=True)

# 禁用 accelerate 對 transformers 的自動 device map 以穩定執行
os.environ.setdefault("ACCELERATE_DISABLE_DEVICE_MAP", "1")
os.environ.setdefault("TRANSFORMERS_NO_ACCELERATE", "1")

# =========================
# 全域設定與除錯日誌
# =========================

MAX_AGENT_CHARS = int(os.getenv("MAX_AGENT_CHARS", "15000"))
VERIFIER_DEBUG = os.getenv("VERIFIER_DEBUG", "0") == "1"
VERIFIER_DEBUG_PATH = Path(
    os.getenv(
        "VERIFIER_DEBUG_PATH",
        "data/processed/verifier/debug/verifier_search_debug.log",
    )
)

AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "24"))
# 依 LangGraph「超步」概念計算遞迴上限：每輪約 3 個超步（agent→tools→accumulate）
AGENT_RECURSION_FACTOR: int = int(os.getenv("AGENT_RECURSION_FACTOR", "3"))
AGENT_RECURSION_EXTRA: int = int(os.getenv("AGENT_RECURSION_EXTRA", "3"))
RECURSION_LIMIT: int = max(AGENT_MAX_STEPS * AGENT_RECURSION_FACTOR + AGENT_RECURSION_EXTRA, 30)

# 移除最低門檻：改由耐心值/步數上限收斂；TOTAL/MIN_PER 設為 0 代表「不啟用」
AGENT_TOTAL_TARGET = int(os.getenv("AGENT_TOTAL_TARGET", "0"))
AGENT_MIN_PER_TRIPLE = int(os.getenv("AGENT_MIN_PER_TRIPLE", "0"))
AGENT_NO_NEW_PATIENCE = int(os.getenv("AGENT_NO_NEW_PATIENCE", "5"))
AGENT_TOP_K_MAX = int(os.getenv("AGENT_TOP_K_MAX", "200"))
MAX_EVID_CHARS = int(os.getenv("MAX_EVID_CHARS", "300"))

# ── 顯名保底種子（機構名）控制 ──
ENABLE_ORG_SEEDS: bool = os.getenv("ENABLE_ORG_SEEDS", "1").lower() in ("1", "true", "yes", "y")
MAX_ORG_SEEDS: int = int(os.getenv("MAX_ORG_SEEDS", "8") or "8")
# 單輪 org 種子（避免 h→t 倒置再跑一次）
ORG_SEED_ONEPASS: bool = os.getenv("ORG_SEED_ONEPASS", "1").lower() in ("1", "true", "yes", "y")

VERIFIER_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)

# 讓工具可以知道這次任務的 id（如 t_1022）；若外部沒傳，就用 timestamp 兜一個
CURRENT_RUN_ID = os.getenv("NEWS_RUN_ID", "").strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _dlog(msg: str) -> None:
    """寫入除錯日誌。"""
    if not VERIFIER_DEBUG:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(VERIFIER_DEBUG_PATH, "a", encoding="utf-8-sig") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

# ====== 寫檔與記錄工具 ======
def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception:
        _dlog(f"write_jsonl failed: {path}\n" + traceback.format_exc())

def _derive_run_id_from_input(input_path_or_id: str) -> str:
    base = Path(input_path_or_id).stem
    # news_kg_t_1022_agent -> t_1022
    m = re.search(r"(t_\d+)", base)
    return m.group(1) if m else base

# =========================
# 健壯 JSON 解析（容忍 LLM 雜訊）
# =========================

def _strip_code_fence(text: str) -> str:
    """移除 Markdown 風格的 code fence。"""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _find_first_json_blob(text: str) -> str | None:
    """擷取第一段平衡的大括號/中括號 JSON 片段。"""
    start_idx = None
    open_ch = None
    depth = 0
    for i, ch in enumerate(text):
        if start_idx is None and ch in "{[":
            start_idx = i
            open_ch = ch
            depth = 1
            continue
        if start_idx is not None:
            if ch == "{":
                if open_ch == "{":
                    depth += 1
            elif ch == "}":
                if open_ch == "{":
                    depth -= 1
            elif ch == "[":
                if open_ch == "[":
                    depth += 1
            elif ch == "]":
                if open_ch == "[":
                    depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _try_fix_minor_json_issues(s: str) -> str:
    """移除尾逗號等小問題。"""
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def parse_json_safely(raw: str) -> Any:
    """健壯 JSON 解析器。"""
    text = _strip_code_fence(raw or "")

    try:
        return json.loads(text)
    except Exception:
        pass

    blob = _find_first_json_blob(text)
    if blob:
        try:
            return json.loads(blob)
        except Exception:
            try:
                return json.loads(_try_fix_minor_json_issues(blob))
            except Exception:
                pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            per = _find_first_json_blob(line)
            if per:
                try:
                    return json.loads(_try_fix_minor_json_issues(per))
                except Exception:
                    continue

    raise ValueError("Unable to parse JSON from input.")

# =========================
# 標準化 RankItem
# =========================
class RankItem(TypedDict):
    id: str
    text: str
    source: str        # 'pg' | 'vector' | 'neo4j'
    rank: int          # 來源內排序（1 起算）
    score: float       # 來源內相似度（可為 0）
    payload: Dict[str, Any]

def _mk_id(text: str, src: str) -> str:
    return f"{src}:{hashlib.sha1((src+'||'+(text or '')).encode('utf-8-sig')).hexdigest()[:12]}"

def _items_from_hits(hits: List[Tuple[Dict, Dict]], src: str) -> List[RankItem]:
    """把 (tri, det) 命中轉成 RankItem 列表；text 以 tri+det 的 JSON 字串表示，確保可還原。"""
    items: List[RankItem] = []
    for i, (tri, det) in enumerate(hits, start=1):
        txt = json.dumps({"tri": tri, "det": det}, ensure_ascii=False)
        items.append({
            "id": _mk_id(txt, src),
            "text": txt,
            "source": src,
            "rank": i,
            "score": float(det.get("meta", {}).get("_debug", {}).get("cond_sum", 0.0)) if isinstance(det, dict) else 0.0,
            "payload": {"tri": tri, "det": det},
        })
    return items


# =========================
# PG 快取路徑（JSON-only）
# =========================

CACHE_DIR = "data/processed/knowledge-graph"
JSON_CACHE_PATH = os.path.join(CACHE_DIR, "pg_index.json")

# =========================
# 三元組鍵名正規化
# =========================

_TRIPLE_KEY_CANDIDATES: Dict[str, List[str]] = {
    "head": ["head", "h", "source", "src", "s", "from", "subject"],
    "relation": ["relation", "rel", "r", "edge", "predicate", "type"],
    "tail": ["tail", "t", "target", "dst", "d", "to", "object", "value", "name"],
}


def _norm_triple_dict(tri: dict) -> dict:
    """將各種鍵名正規化為 {head, relation, tail}。"""
    out = {"head": "未知", "relation": "未知", "tail": "未知"}
    if not isinstance(tri, dict):
        return out
    for std_key, cands in _TRIPLE_KEY_CANDIDATES.items():
        for k in cands:
            if k in tri and isinstance(tri[k], str) and tri[k].strip():
                out[std_key] = tri[k].strip()
                break
    try:
        out["relation"] = _normalize_relation(out.get("relation", "未知"), RELATIONS_ALL)
    except Exception:
        out["relation"] = _normalize_relation(out.get("relation", "未知"), None)
    return out


def _norm_hits(hits: Any) -> List[Tuple[Dict, Dict]]:
    """將各式命中結果規整為 (tri, det) 清單。"""
    normed: List[Tuple[Dict, Dict]] = []
    for h in hits or []:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            tri, det = h[0], h[1]
        elif isinstance(h, dict):
            tri = h.get("triple") or h.get("tri") or h
            det = h.get("detail") or h.get("det") or h
        else:
            continue
        tri_n = _norm_triple_dict(tri if isinstance(tri, dict) else {})
        det_n = det if isinstance(det, dict) else {}
        normed.append((tri_n, det_n))
    return normed


# =========================
# 可選外部關係詞庫（不影響功能）
# =========================

_RELATION_DICT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "knowledge-graph"
    / "relation_dict_all.py"
)


def _load_relation_dict() -> set:
    """載入可選關係詞庫。"""
    try:
        spec = importlib.util.spec_from_file_location(
            "relation_dict_all", _RELATION_DICT_PATH
        )
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore
        rels = getattr(module, "RELATIONS_ALL", set())
        return set(rels) if rels else set()
    except Exception:
        return set()


RELATIONS_ALL = _load_relation_dict()

# =========================
# Relation 正規化
# =========================
_RE_PAREN_FULL = re.compile(r"（.*?）")
_RE_PAREN_HALF = re.compile(r"\(.*?\)")
_RE_SPLIT_CAND = re.compile(r"[／/、,，;；\s]+")

def _normalize_relation(rel: str, whitelist: Optional[Set[str]] = None) -> str:
    """正規化關係字（外層 base 與括號候選；白名單優先）。"""
    if not isinstance(rel, str):
        return "未知"
    s = rel.strip()
    if not s:
        return "未知"

    paren_chunks: List[str] = []
    for m in _RE_PAREN_FULL.findall(s):
        paren_chunks.append(m.strip("（）"))
    for m in _RE_PAREN_HALF.findall(s):
        paren_chunks.append(m.strip("()"))

    base = _RE_PAREN_FULL.sub("", s)
    base = _RE_PAREN_HALF.sub("", base)
    base = re.sub(r"[，、。；;:：\s]+", "", base)

    candidates: List[str] = []
    for chunk in paren_chunks:
        for tok in _RE_SPLIT_CAND.split(chunk):
            tok = tok.strip()
            if tok:
                candidates.append(tok)
    if base:
        candidates.append(base)

    wl = whitelist or set()
    for cand in candidates:
        if cand in wl:
            return cand
    if base:
        return base
    return s


# =========================
# 內容格式化（恢復完整資訊）
# =========================

def _fmt_attrs(d: dict) -> str:
    """將屬性 dict 轉為「k1:v1；k2:v2」字串。"""
    if not isinstance(d, dict) or not d:
        return ""
    items = [f"{k}:{v}" for k, v in d.items() if v not in (None, "", [])]
    return "；".join(items)

# [NEW] 自動解包 {"props": {...}}
def _unwrap_props(d: dict) -> dict:
    """若為雙層 props，回傳最內層；否則原樣。"""
    if isinstance(d, dict) and "props" in d and isinstance(d["props"], dict):
        return d["props"]
    return d if isinstance(d, dict) else {}


def _format_kb_line(tri: dict, det: dict, max_evid: int) -> str:
    """輸出單條比對行（含 type/屬性/關係名/說明/事件時間）。"""
    # [CHANGED] 先解包 props
    h = _unwrap_props(det.get("head", {}) or {})
    t = _unwrap_props(det.get("tail", {}) or {})
    r = _unwrap_props(det.get("rel", {}) or {})

    h_name = tri.get("head", "") or h.get("name", "") or ""
    t_name = tri.get("tail", "") or t.get("name", "") or ""
    # ── 關係名健壯取值（避免出現空的【】）
    rel_name = (
        tri.get("relation")
        or r.get("relation")
        or r.get("name")
        or r.get("label")
        or r.get("關係")
        or r.get("type")
        or r.get("relation_name")
        or r.get("rel")
        or r.get("關係名")
        or ""
    )
    rel_name = str(rel_name).strip()
    # 正規化（若有關係白名單則優先）
    try:
        rel_name = _normalize_relation(rel_name, RELATIONS_ALL)
    except Exception:
        rel_name = _normalize_relation(rel_name, None)
    if not rel_name:
        rel_name = "提及"

    h_type = h.get("type")
    t_type = t.get("type")

    h_attrs = _fmt_attrs({k: v for k, v in h.items() if k not in ("name", "type")})
    t_attrs = _fmt_attrs({k: v for k, v in t.items() if k not in ("name", "type")})

    date = r.get("date") or r.get("事件時間") or ""
    evi = r.get("evidence") or r.get("desc") or ""
    if isinstance(evi, str) and max_evid > 0 and len(evi) > max_evid:
        evi = evi[:max_evid] + "…"

    h_block = h_name
    det_parts: List[str] = []
    if h_type:
        det_parts.append(f"type:{h_type}")
    if h_attrs:
        det_parts.append(h_attrs)
    if det_parts:
        h_block += f"（{'；'.join(det_parts)}）"

    t_block = t_name
    det_parts = []
    if t_type:
        det_parts.append(f"type:{t_type}")
    if t_attrs:
        det_parts.append(t_attrs)
    if det_parts:
        t_block += f"（{'；'.join(det_parts)}）"


    # 一律輸出【關係名】
    line = f"[比對] {h_block} 透過關係【{rel_name}】與 {t_block} 建立連結，說明：{evi}"
    if date:
        line += f"；事件時間：{date}"
    line += "。"
    return line

# =========================
# 機構名擴充：提示工程 + 正則 fallback
# =========================
# 1) 長後綴優先；2) 對單字後綴提高主幹長度門檻；3) 黑名單與英文公司後綴支持

# 擴展後綴清單（分組，長詞在前；單字後綴放最後一組）
_ORG_SUFFIXES_CLEANED_GROUPS = [
    # 核心公司法人與法律實體（長詞優先）
    "股份有限公司|有限公司|控股公司|金控公司|資產管理|傳播公司|出版公司|影視公司|行銷公司|廣告公司|設計公司|投資公司|科技公司|工程公司|建設公司|服務中心|研究中心|檢驗所|認證機構|評估公司|鑑定公司|設計工作室|聯合事務所|律師事務所|會計師事務所|建築師事務所|專利事務所|資產管理公司|管理顧問公司|顧問公司|實驗室|工作室|事務所|總公司|企業|集團|公司|商行|商號|商店|商社|商會|行號|廠|工廠|製造廠|製藥廠|藥廠|銀行|證券|保險公司|經紀公司|代理商|經銷商|開發公司|實業公司|服務社|服務隊|合作社|合作金庫",
    # 核心社會組織與非營利
    "社區發展協會|非營利組織|慈善機構|慈善團體|志願團體|公民組織|人民團體|保育協會|發展協會|推廣會|促進會|聯誼會|宗親會|志工團|社會服務中心|學會|協會|基金會|中心|研究院|研究所|機構|組織|工會|公會|聯盟|社|社團|社群|自救會|互助會|福利會|青年會|婦女會|老人會|環保團體",
    # 核心學術教育與文化場所
    "社區大學|空中大學|實驗學校|教育中心|教育機構|技藝中心|圖書館|博物館|紀念館|文化館|美術館|科學館|天文台|動物園|植物園|大學|學院|學校|中學|小學|幼兒園|補習班|安親班",
    # 核心媒體出版與廣播
    "出版集團|廣播公司|通訊社|新聞社|電視公司|電視臺|電視台|廣播電臺|廣播電台|頻道|出版社|期刊|雜誌|季刊|月報|週報|晚報|時報|日報|報",
    # 核心醫療與公益
    "醫療中心|醫療院所|護理之家|養護中心|復健中心|心理諮商所|心理治療所|捐血中心|動物醫院|獸醫診所|衛生所|醫院|診所|血庫",
    # 核心政府與政黨（長詞在前）
    "立法院|市議會|縣議會|鄉鎮市民代表會|檢察署|特勤中心|情報機關|駐外代表處|辦事處|鄉公所|鎮公所|區公所|里辦公處|村辦公處|警察局|消防局|國會|議會|政黨|黨|法院|院|部|署|局|處|會|司|廳",
    # 交通運輸
    "高鐵公司|港務局|轉運站|物流中心|航運公司|海運公司|船運公司|客運公司|捷運公司|鐵路局|航空站|機場|港口|計程車行|租車公司|快遞公司|貨運公司|郵政公司|郵局",
    # 體育娛樂
    "運動中心|體育館|運動場|體育協會|體育會|俱樂部|表演廳|劇團|樂團|唱片公司|電影院|影城|展覽館|會議中心|遊樂園|休閒農場|度假村|溫泉會館|遊戲公司|動畫公司|漫畫公司|球隊|健身房",
    # 農業漁業
    "農業改良場|農業試驗所|水利會|農會|漁會|農場|牧場|漁場|畜牧場|林場|茶廠|酒莊|水產公司",
    # 宗教信仰（單字後綴較多，放後段）
    "禮拜堂|修道院|清真寺|教會|教堂|佛堂|精舍|道觀|寺|廟|宮|堂|庵|祠|壇|院|殿|觀"
]

# 英文公司/組織後綴（補英文化名）：Inc., Ltd., LLC, GmbH, AG, Corp., Co., PLC…
_EN_ORG_SUFFIXES = r"(?:Inc\.|Incorporated|Ltd\.|Limited|LLC|L\.L\.C\.|PLC|P\.L\.C\.|GmbH|AG|S\.A\.|S\.p\.A\.|Corp\.|Corporation|Company|Co\.)"

# 依長度排序合併（確保長詞優先）
_ORG_SUFFIXES_CLEANED = "|".join(
    sum((grp.split("|") for grp in _ORG_SUFFIXES_CLEANED_GROUPS), [])
)

# 僅在「詞邊界或左引號/書名號」開頭，主幹不跨句讀；限制最長 40 字，避免吞整句
_ORG_REGEX = re.compile(
    rf"(?<![\w\u4e00-\u9fff])"              # 左邊界：不要緊貼字母/數字/CJK
    rf"(?:[《「『(【])?"                    # 可選：開頭引號/書名號
    rf"("
    rf"[A-Za-z0-9\u4e00-\u9fff·＆&．\.\-（）()／\s]{{2,40}}"  # 主幹（不含引號），不跨句讀
    rf")"
    rf"(?:{_ORG_SUFFIXES_CLEANED}|{_EN_ORG_SUFFIXES})"
    rf"(?![\w\u4e00-\u9fff])"              # 右邊界
)

# 過泛詞黑名單（單獨命中時剔除）
_ORG_STOP_SINGLE = {
    "政府","媒體","公司","集團","中心","研究所","研究院","學院","大學","學校",
    "法院","檢察署","委員會","電視台","電視臺","電台","電臺","報","醫院","診所","寺","廟",
}

def _extract_org_seeds(text: str) -> List[str]:
    """
    從原始新聞文本抽取「機構名稱」作為保底檢索種子。
    1) 先用 LLM 嚴格 JSON 提示抽 organizations；
    2) 若失敗或為空，fallback 用中文機構後綴正則抽取；
    3) 去重、截斷。
    """
    out: List[str] = []
    try:
        llm = _build_llm()
        sys_prompt = (
            "你是資訊抽取器。請只輸出 JSON，格式為："
            '{"organizations":["..."]}。規則：\n'
            "1) 專門找『機構/媒體/公司/協會/基金會/政黨/政府單位』等名稱（包含繁中全名）。\n"
            "2) 不要人名、地名、事件名；避免過泛詞（如「政府」「媒體」單獨）。\n"
            "3) 僅列文本中『明確出現』的名稱；每項長度 2–32 字。\n"
            "4) 請去重，最多返回 20 筆。"
        )
        user_prompt = f"[文本]\n{text[:12000]}"
        resp = llm.invoke([{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}])
        raw = resp.content if isinstance(resp.content, str) else json.dumps(resp.content, ensure_ascii=False)
        obj = {}
        try:
            obj = json.loads(_strip_code_fence(raw))
        except Exception:
            blob = _find_first_json_blob(raw or "")
            if blob:
                obj = json.loads(_try_fix_minor_json_issues(blob))
        arr = obj.get("organizations") if isinstance(obj, dict) else None
        if isinstance(arr, list):
            out = [str(x).strip() for x in arr if isinstance(x, (str, int, float)) and str(x).strip()]
    except Exception:
        pass
    # 正則 fallback / 混合
    if not out:
        out = []

    # ⚠️ 拿整段完整匹配，確保把「中央社 / 協會 / 公司」等後綴包含進來
    regex_hits = [m.group(0).strip() for m in _ORG_REGEX.finditer(text or "")]
    out.extend(regex_hits)

    # 清理：去重、過濾過長/過短、去掉孤立泛詞
    seen: set = set()
    cleaned: List[str] = []

    # —— 以實際字元集合處理中英文引號/書名號/括號（避免 r'' + \uXXXX 失效）——
    OPEN_QUOTES = "“„‟‹«『「《(（[【"
    CLOSE_QUOTES = "”‟”›»』」》)）]】"
    OPENERS = set(OPEN_QUOTES)
    CLOSERS = set(CLOSE_QUOTES)
    # 英文直角引號也一併處理
    EXTRA_LEFT = set("\"'")
    EXTRA_RIGHT = set("\"'")

    def normalize_org_name(s: str) -> str:
        s = s.strip()
        # 反覆剝外層引號/括號與空白
        changed = True
        while changed and s:
            changed = False
            # 左側
            while s and (s[0] in OPENERS or s[0] in EXTRA_LEFT or s[0].isspace()):
                s = s[1:]
                changed = True
            # 右側
            while s and (s[-1] in CLOSERS or s[-1] in EXTRA_RIGHT or s[-1].isspace()):
                s = s[:-1]
                changed = True
        # 去常見前導語
        s = re.sub("(?:^)(以|由|據稱?|對於|針對)\\s*", "", s)
        # 句讀/換行前截斷
        s = re.split(r"[，,。．\.、;；：:！？!?\n]", s)[0]
        # 合併多空白
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def strip_all_quotes(s: str) -> str:
        return re.sub(r"[\"'“”„‟‹›«»『』「」《》()（）\[\]【】]", "", s)

    for name in out:
        name = normalize_org_name(name)
        n = re.sub(r"\s+", "", name)  # 無空白版長度檢查
        if not (2 <= len(n) <= 64):
            continue
        # 過濾常見泛詞單獨出現
        if n in _ORG_STOP_SINGLE:
            continue
        # 若是單字後綴命中（如「院/部/局/會/司/報/寺/廟」），要求主幹長度≥2
        # 取末尾 1 字後綴檢查
        if len(n) <= 3 and n[-1] in {"院","部","局","處","會","司","報","寺","廟","堂"}:
            # 主幹 = 去掉最後 1 字後綴
            core = n[:-1]
            core = re.sub(r"[（）()《》「」『』·＆&．\.\-/\s]+", "", core)
            if len(core) < 2:
                continue
        # 去重 key：移除所有引號/括號，避免「吹哨者協會」與「「吹哨者協會」」重複
        key = strip_all_quotes(n)
        if key not in seen:
            seen.add(key)
            cleaned.append(name.strip())
    return cleaned[:MAX_ORG_SEEDS] if MAX_ORG_SEEDS > 0 else cleaned


# =========================
# 向量檢索實作（共用於工具與保底）
# =========================

_VEC_TOOL_LOCK = threading.RLock()

def _vector_search_impl(tp: Dict[str, str], top_k: int = 100) -> List[str]:
    """本地向量檢索實作，回傳比對行。"""
    with _VEC_TOOL_LOCK:
        try:
            tp = _norm_triple_dict(tp)
            q_vec = embed_triple(tp)
            idxs = cosine_search(tp, q_vec)
            lines: List[str] = []
            for i in idxs[:top_k]:
                tri, det = kg_row_to_detail(i)
                lines.extend(_collect_hits_to_lines(_norm_hits([(tri, det)])))
            return lines
        except Exception:
            _dlog("vector_search_impl: failed\n" + traceback.format_exc())
            return []


# =========================
# 輔助：從多種入參格式擷取第一個三元組
# =========================

def _take_first_triple(obj: Any) -> dict:
    """從各種可能格式中取出第一個三元組。"""
    if isinstance(obj, dict):
        if any(k in obj for k in ("head", "relation", "tail")):
            return obj
        if "triples" in obj and isinstance(obj["triples"], list) and obj["triples"]:
            cand = obj["triples"][0]
            return cand if isinstance(cand, dict) else {}
        return {}
    if isinstance(obj, list) and obj:
        return obj[0] if isinstance(obj[0], dict) else {}
    return {}


# =========================
# Tools（任何錯誤都回 error/寫 log）
# =========================

@tool(
    "extract_triples",
    return_direct=False,
    description="抽取三元組：輸入新聞文字，輸出 JSON {'triples': [...]}。",
)
def tool_extract_triples(text: str) -> str:
    """工具：抽取三元組。"""
    try:
        raw = extract_entities_relations(text) or ""
        # 統一使用健壯解析，容忍圍欄/雜訊/尾逗號
        try:
            obj = parse_json_safely(raw) if raw else {}
        except Exception:
            obj = {}
        # 內部以標準鍵名組裝 → 對外轉成 h/r/t 外觀
        triples_std = _er_to_triples(obj)  # [{'head','relation','tail'}, ...]
        triples_hrt = [
            {"h": t.get("head", ""), "r": t.get("relation", ""), "t": t.get("tail", "")}
            for t in triples_std
        ]
        _dlog(f"extract_triples: triples={len(triples_hrt)}")
        return json.dumps({"triples": triples_hrt}, ensure_ascii=False)
    except Exception:
        _dlog("extract_triples: failed\n" + traceback.format_exc())
        return json.dumps(
            {"triples": [], "error": "extract_exception"}, ensure_ascii=False
        )


@tool(
    "pg_search",
    return_direct=False,
    description=(
        "CSV→Property Graph 檢索（支援多跳）。"
        "輸入三元組 JSON、top_k、hops；輸出 {'lines': [...]}。"
    ),
)
def tool_pg_search(triple_json: str, top_k: int = 50, hops: int = 3) -> str:
    """
    工具：PG 檢索（正反雙向），自動處理被動語態造成的方向顛倒。
    （尊重 ENABLE_LI_PG / LI_PG_INDEX_JSON）。
    """
    retriever = _try_load_pg()
    if retriever is None:
        hint = (
            "PG retriever not available. 請確認 .env 的 ENABLE_LI_PG=1、LI_PG_INDEX_JSON 路徑，"
            "或先執行：python -m src.qa.preliminary_work.build_csv_property_graph build <csv>"
        )
        _dlog("pg_search: unavailable\n" + hint)
        return json.dumps(
            {"lines": [], "hits": 0, "error": "pg_unavailable", "note": hint},
            ensure_ascii=False,
        )
    # 以 .env 覆蓋（保留入參作為下限）
    try:
        top_k = max(top_k, int(os.getenv("LI_PG_TOPK", str(top_k))))
    except Exception:
        pass
    try:
        hops = max(hops, int(os.getenv("LI_PG_HOPS", str(hops))))
    except Exception:
        pass
    try:
        # --- 解析 triple_json ---
        try:
            tp_raw = parse_json_safely(triple_json)
        except Exception as e:
            preview = (triple_json or "").strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + " ...<truncated>"
            _dlog(f"pg_search: invalid triple_json: {e}; preview={preview}")
            return json.dumps(
                {
                    "lines": [],
                    "hits": 0,
                    "error": "pg_invalid_json",
                    "note": str(e),
                    "preview": preview,
                },
                ensure_ascii=False,
            )

        # ① 取第一個三元組並正規化（正向）
        tp = _norm_triple_dict(_take_first_triple(tp_raw))
        # ② 構造反向（head/tail 互換），處理被動語態/方向顛倒
        tp_inv = {
            "head": tp.get("tail", ""),
            "relation": tp.get("relation", ""),
            "tail": tp.get("head", ""),
        }

        # --- ORG_SEED（無向 OR）偵測＋log：relation 空、且 h xor t ---
        def _is_org_seed(tp: dict) -> bool:
            try:
                h = (tp.get("head") or "").strip()
                r = (tp.get("relation") or "").strip()
                t = (tp.get("tail") or "").strip()
                return (r == "") and bool(h) ^ bool(t)
            except Exception:
                return False

        if _is_org_seed(tp):
            org_mode = os.getenv("ORG_SEED_MODE","UNDIRECTED")
            use_rel = os.getenv("ORG_SEED_USE_REL","0")
            eff_val = (tp.get("head") or tp.get("tail") or "").strip()
            eff_tri = {
                "head": eff_val,
                "relation": (eff_val if str(use_rel).lower() in {"1","true","yes","y"} else ""),
                "tail": eff_val,
            }
            _dlog(
                "pg_search: ORG_SEED detected "
                f"(MODE={org_mode}, MIN_SHOULD={os.getenv('ORG_SEED_MIN_SHOULD','1')}, "
                f"USE_REL={use_rel}, ALLOW_INVERT={os.getenv('ORG_SEED_ALLOW_INVERT','0')}) | "
                f"original={json.dumps(tp, ensure_ascii=False)} | effective={json.dumps(eff_tri, ensure_ascii=False)}"
            )

       # ③ 先查正向，再視情況查反向
        hits_norm = retriever.search_triple(tp, top_k=top_k, hops=hops)
        hits_inv: list = []
        if tp.get("head") != tp.get("tail"):
            hits_inv = retriever.search_triple(tp_inv, top_k=top_k, hops=hops)

        # ④ 合併並規整、轉文字、去重
        hits_all = _norm_hits(hits_norm + hits_inv)
        lines = _collect_hits_to_lines(hits_all)
        items = _items_from_hits(hits_all, src="pg")

        # 定義 unique_lines
        unique_lines = list(dict.fromkeys(lines))

        _dlog(
            "pg_search: summary | "
            f"hits_norm={len(hits_norm)}, hits_inv={len(hits_inv)}, "
            f"unique={len(unique_lines)}, hops={hops}, top_k={top_k}"
        )
        return json.dumps(
            {"lines": unique_lines, "items": items, "hits": len(unique_lines)},
            ensure_ascii=False
        )
    except Exception:
        # 這是外層 try 的補齊：任何未預期錯誤，記 log 並回傳 JSON
        _dlog("pg_search: exception\n" + traceback.format_exc())
        return json.dumps(
            {
                "lines": [],
                "hits": 0,
                "error": "pg_exception",
                "note": "pg_search failed"
            },
            ensure_ascii=False,
        )


@tool(
    "neo4j_search",
    return_direct=False,
    description=(
        "Neo4j 檢索（支援多跳）。"
        "輸入三元組 JSON、top_k、hops；輸出 {'lines': [...]}。"
    ),
)
def tool_neo4j_search(triple_json: str, top_k: int = 50, hops: int = 2) -> str:
    """工具：Neo4j 檢索。"""
    retriever = _try_load_neo4j()
    if retriever is None:
        note = "Neo4j retriever not available（已關閉或連線失敗）"
        _dlog("neo4j_search: unavailable")
        return json.dumps(
            {"lines": [], "error": "neo4j_unavailable", "note": note},
            ensure_ascii=False,
        )
    try:
        try:
            tp_raw = parse_json_safely(triple_json)
        except Exception as e:
            preview = (triple_json or "").strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + " ...<truncated>"
            _dlog(f"neo4j_search: invalid triple_json: {e}; preview={preview}")
            return json.dumps(
                {
                    "lines": [],
                    "error": "neo4j_invalid_json",
                    "note": str(e),
                    "preview": preview,
                },
                ensure_ascii=False,
            )

        tp = _norm_triple_dict(_take_first_triple(tp_raw))
        hits = retriever.search_triple(tp, top_k=top_k, hops=hops)
        hits = _norm_hits(hits)
        _dlog(f"neo4j_search: hits={len(hits)}, hops={hops}, top_k={top_k}")
        return json.dumps({
            "lines": _collect_hits_to_lines(hits),
            "items": _items_from_hits(hits, src="neo4j")
        }, ensure_ascii=False)
    except Exception:
        _dlog("neo4j_search: search failed\n" + traceback.format_exc())
        return json.dumps(
            {
                "lines": [],
                "error": "neo4j_exception",
                "note": "neo4j_search failed",
            },
            ensure_ascii=False,
        )


@tool(
    "vector_search",
    return_direct=False,
    description="本地向量檢索。輸入三元組 JSON 與 top_k；輸出 {'lines': [...]}。",
)
def tool_vector_search(triple_json: str, top_k: int = 100) -> str:
    """工具：本地向量檢索。"""
    try:
        try:
            tp_raw = parse_json_safely(triple_json)
        except Exception as e:
            preview = (triple_json or "").strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + " ...<truncated>"
            _dlog(f"vector_search: invalid triple_json: {e}; preview={preview}")
            return json.dumps(
                {
                    "lines": [],
                    "error": "vector_invalid_json",
                    "note": str(e),
                    "preview": preview,
                },
                ensure_ascii=False,
            )

        tp = _norm_triple_dict(_take_first_triple(tp_raw))
        lines = _vector_search_impl(tp, top_k=top_k)
        # 將向量檢索結果包成 RankItem（無 det 可用簡單 payload）
        items: List[RankItem] = []
        for i, ln in enumerate(lines, start=1):
            txt = ln if ln.startswith("[比對]") else f"[比對] {ln}"
            items.append({
                "id": _mk_id(txt, "vector"),
                "text": txt,
                "source": "vector",
                "rank": i,
                "score": 0.0,
                "payload": {"line": txt}
            })
        _dlog(f"vector_search: returned_lines={len(lines)}")
        return json.dumps({"lines": lines, "items": items}, ensure_ascii=False)
    except Exception:
        _dlog("vector_search: failed\n" + traceback.format_exc())
        return json.dumps(
            {"lines": [], "error": "vector_exception", "note": "vector_search failed"},
            ensure_ascii=False,
        )


@tool(
    "merge_and_dedup",
    return_direct=False,
    description=(
        "合併多路檢索結果並去重。輸入 JSON 含 'pg'、'neo4j'、'vector'（各自含 'lines'）；"
        "輸出 {'lines': [...]}。"
    ),
)
def tool_merge_and_dedup(payload: str) -> str:
    """工具：合併 → 輸出 lines。"""
    try:
        obj = json.loads(payload)
    except Exception:
        obj = {}

    # 1) 收集三路 items
    def _get_items(key: str) -> List[RankItem]:
        node = obj.get(key) or {}
        return list(node.get("items") or [])
    runs: Dict[str, List[RankItem]] = {
        "pg": _get_items("pg"),
        "vector": _get_items("vector"),
        "neo4j": _get_items("neo4j"),
    }

    # 若完全沒有 items，保留舊行為（單純合併去重 lines）
    if not any(runs.values()):
        lines_all: List[str] = []
        for k in ("pg", "neo4j", "vector"):
            lines_all.extend(list((obj.get(k) or {}).get("lines") or []))
        kept = deduplicate(lines_all, triples=None)
        _dlog(f"merge_and_dedup(fallback): in={len(lines_all)}, out={len(kept)}")
        return json.dumps({"lines": kept}, ensure_ascii=False)
        
    # 替代邏輯：直接合併所有 items 並轉換為 lines
    all_items: List[RankItem] = []
    all_items.extend(runs["pg"])
    all_items.extend(runs["vector"])
    all_items.extend(runs["neo4j"])

    # 5) 渲染回原有 lines（相容既有格式）
    lines: List[str] = []
    
    # [MODIFIED] 從 all_items 渲染，而不是 ranked
    for c in all_items:
        # 結構是 RankItem，text 欄位存了 (JSON 字串) 或 ([比對]行)
        txt = c.get("text") or ""
        
        # 若 text 是 JSON 串 (來自 pg/neo4j)，則取 det/evidence 渲染；
        try:
            objj = json.loads(txt)
            tri2, det2 = objj.get("tri") or {}, objj.get("det") or {}
            s = _format_kb_line(_norm_triple_dict(tri2), det2, max_evid=MAX_EVID_CHARS)
        except Exception:
            # 否則 (來自 vector) 直接輸出
            s = txt if txt.startswith("[比對]") else f"[比對] {txt}"
        lines.append(s)

    # 去重、保序
    lines = list(dict.fromkeys(lines))
    
    # 返回簡化後的 JSON
    return json.dumps({
        "lines": lines
    }, ensure_ascii=False)


# =========================
# 檢索器載入（單例；錯誤寫 log）
# =========================

_RET_PG = None
_RET_NEO4J = None

def _try_load_pg() -> Any:
    """嘗試載入 CSV Property Graph 檢索器（JSON-only）。"""
    global _RET_PG
    if _RET_PG is not None:
        return _RET_PG

    if os.getenv("ENABLE_LI_PG", "1").lower() not in ("1", "true", "yes"):
        _dlog("pg_search: disabled by env ENABLE_LI_PG")
        _RET_PG = None
        return _RET_PG

    try:
        from ...tools.property_graph.li_csv_pg_retriever import (
            CsvPropertyGraphRetriever,
        )

        idx_json = os.getenv("LI_PG_INDEX_JSON", JSON_CACHE_PATH)

        retr = None
        if Path(idx_json).is_file():
            try:
                retr = CsvPropertyGraphRetriever.load_from_json(idx_json=idx_json)
                _dlog(
                    "pg_search: loaded JSON index "
                    f"json={Path(idx_json).exists()}"
                )
            except Exception:
                _dlog("pg_search: load_from_json failed\n" + traceback.format_exc())

        if retr is None:
            retr = CsvPropertyGraphRetriever.ensure_built_and_loaded_json()
            _dlog("pg_search: ensure_built_and_loaded_json ok")

        _RET_PG = retr
        return _RET_PG
    except Exception:
        _dlog("pg_search: failed to load retriever\n" + traceback.format_exc())
        _RET_PG = None
        return _RET_PG


def _try_load_neo4j() -> Any:
    """嘗試載入 Neo4j 檢索器（缺模組/連線失敗時回傳 None）。"""
    global _RET_NEO4J
    if _RET_NEO4J is not None:
        return _RET_NEO4J
    if os.getenv("ENABLE_LI_ONLINE", "0").lower() not in ("1", "true", "yes"):
        _RET_NEO4J = None
        _dlog("neo4j_search: disabled by env ENABLE_LI_ONLINE")
        return _RET_NEO4J
    try:
        from ..kg.llamaIndex.neo4j_li_retriever import LlamaIndexNeo4jRetriever  # type: ignore

        _RET_NEO4J = LlamaIndexNeo4jRetriever(
            date_field="date", evidence_field="evidence"
        )
        return _RET_NEO4J
    except Exception:
        _dlog("neo4j_search: failed to init\n" + traceback.format_exc())
        _RET_NEO4J = None
        return _RET_NEO4J


# =========================
# ER → Triples、命中 → [比對] 行
# =========================

def _er_to_triples(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """由抽取結果物件轉換為三元組清單。"""
    if not isinstance(obj, dict):
        return []
    if "entities" in obj or "relations" in obj:
        ents = {e.get("id"): e for e in obj.get("entities", []) if isinstance(e, dict)}
        triples: List[Dict[str, str]] = []
        for r in obj.get("relations", []) or []:
            s = ents.get(r.get("source") or "", {}) or {}
            t = ents.get(r.get("target") or "", {}) or {}
            head = (s.get("name") or "").strip() or "未知"
            rel = (r.get("relation") or "").strip() or "未知"
            tail = (t.get("name") or "").strip()
            if not tail:
                attrs = t.get("attributes") or {}
                val = attrs.get("value")
                unit = attrs.get("unit") or ""
                tail = f"{val}{unit}".strip() if val not in (None, "") else "未知"
            triples.append({"head": head, "relation": rel, "tail": tail})
        return triples
    if isinstance(obj.get("triples"), list):
        return [_norm_triple_dict(x) for x in obj["triples"] if isinstance(x, dict)]
    return []


def _collect_hits_to_lines(hits: List[Tuple[Dict, Dict]]) -> List[str]:
    """將檢索命中轉為比對文字行（保留 type/屬性/關係名/事件時間/說明）。"""
    lines: List[str] = []
    for tri, det in hits:
        tri = _norm_triple_dict(tri)
        try:
            line = _format_kb_line(tri, det, max_evid=MAX_EVID_CHARS)
            lines.append(line)
        except Exception:
            # 後備格式化（防禦）
            h = _unwrap_props(det.get("head", {}) or {})
            t = _unwrap_props(det.get("tail", {}) or {})
            r = _unwrap_props(det.get("rel", {}) or {})
            head = tri.get("head", "未知") or h.get("name", "未知")
            tail = tri.get("tail", "未知") or t.get("name", "未知")
            ev = r.get("evidence", "") or r.get("desc", "") or ""
            # 後備關係名（避免【】留空，統一輸出方括號）
            rel_name = (
                r.get("relation")
                or r.get("name")
                or r.get("label")
                or r.get("type")
                or ""
            )
            rel_name = str(rel_name).strip() or "提及"
            if MAX_EVID_CHARS > 0 and isinstance(ev, str) and len(ev) > MAX_EVID_CHARS:
                ev = ev[:MAX_EVID_CHARS] + "…"
            dt = r.get("date", "") or ""
            frag = f"[比對] {head} 透過關係【{rel_name}】與 {tail} 建立連結，說明：{ev}"
            if dt:
                frag += f"；事件時間：{dt}"
            frag += "。"
            lines.append(frag)
    return lines


# =========================
# Graph 狀態與節點
# =========================

class AgentState(TypedDict, total=False):
    """LangGraph 狀態定義。"""

    messages: List[BaseMessage]
    news_text: str
    triples: List[Dict[str, str]]
    triples_processed_index: int  # triple 索引（自 0 起）
    accum_lines: List[str]
    step: int
    no_new_steps: int
    notes: str
    tool_availability: str


def _build_llm() -> ChatOpenAI:
    """建立 LLM 物件。"""
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0, max_tokens=2048)


def _get_tool_availability() -> str:
    """產生工具可用性字串。"""
    pg_ok = _try_load_pg() is not None
    if os.getenv("ENABLE_LI_ONLINE", "0").lower() in ("1", "true", "yes"):
        neo_ok = _try_load_neo4j() is not None
    else:
        neo_ok = False
    # 是否啟用向量備援由環境變數控制（預設開）
    vec_ok = os.getenv("ENABLE_VECTOR_FALLBACK", "1").lower() in ("1", "true", "yes")
    return (
        f"pg_search={'available' if pg_ok else 'UNAVAILABLE'}; "
        f"neo4j_search={'available' if neo_ok else 'UNAVAILABLE'}; "
        f"vector_search={'available' if vec_ok else 'UNAVAILABLE'}"
    )


AGENT_SYS_PROMPT = (
    "任務：『新聞查核檢索代理』\n"
    "規則：\n"
    "1) 已為你抽好三元組（triples）。你每一輪只需處理**當前這一個**三元組。\n"
    "2) 不要呼叫 `extract_triples`。\n"
    "3) 僅呼叫標示 available 的工具（如 `pg_search` 與 `vector_search`），並以 `merge_and_dedup` 合併去重，累積 [比對] 行。\n"
    "4) 證據不足則進入下一輪處理下一個三元組；達到耐心值或步數上限或三元組全數處理完畢則結束。\n"
    "5) 聚焦工具使用，避免空泛敘述。"
)

# 依環境決定掛載哪些工具（避免在只想測 PG 時仍被向量備援介入）
# 注意：不要把 extract_triples 暴露給 LLM，因為我們已在 extract_triples_node 抽過了
TOOLS = [tool_pg_search]
if os.getenv("ENABLE_LI_ONLINE", "0").lower() in ("1", "true", "yes"):
    TOOLS.append(tool_neo4j_search)
if os.getenv("ENABLE_VECTOR_FALLBACK", "1").lower() in ("1", "true", "yes"):
    TOOLS.append(tool_vector_search)
TOOLS.append(tool_merge_and_dedup)

def extract_triples_node(state: AgentState) -> Dict:
    """只執行一次：抽取三元組並存入狀態。"""
    _dlog("node: extract_triples_node executing...")
    news_text = state.get("news_text", "") or ""
    triples: List[Dict[str, str]] = []
    try:
        raw = extract_entities_relations(news_text) or ""
        # 與工具一致改用健壯解析
        try:
            obj = parse_json_safely(raw) if raw else {}
        except Exception:
            obj = {}
        triples = _er_to_triples(obj)
    except Exception:
        _dlog("node: extract_triples_node failed\n" + traceback.format_exc())
    # ── 顯名保底種子：機構名 → 並列 triple 種子（不干擾既有三元組） ──
    org_triples: List[Dict[str, str]] = []
    if ENABLE_ORG_SEEDS:
        try:
            orgs = _extract_org_seeds(news_text)
            # 轉成雙向通配 triple： (ORG, *, *) 與 (*, *, ORG)
            for org in orgs:
                org = org.strip()
                if not org:
                    continue
                if ORG_SEED_ONEPASS:
                    # 只產生一條：由 PG 檢索器將其視為「h/r/t 任一命中即可」（無向 OR）
                    org_triples.append({"head": org, "relation": "", "tail": ""})
                else:
                    # 舊行為（相容）：h 與 t 各一條
                    org_triples.append({"head": org, "relation": "", "tail": ""})
                    org_triples.append({"head": "", "relation": "", "tail": org})
            _dlog(f"org_seeds: names={len(orgs)} -> triples={len(org_triples)} | {orgs}")
        except Exception:
            _dlog("org_seeds: failed\n" + traceback.format_exc())

    # 併入：讓 org 種子先被處理（優先補召回），再處理抽出的語意三元組
    merged_triples = (org_triples + triples) if org_triples else triples
    _dlog(f"node: extract_triples_node found {len(triples)} triples; merged={len(merged_triples)} (with org seeds).")
    return {"triples": merged_triples, "triples_processed_index": 0}


def agent_node(state: AgentState) -> Dict:
    """代理節點：產生工具呼叫或回覆訊息。"""
    # 先決定是否還有 triple 要處理
    triples = state.get("triples", []) or []
    idx = int(state.get("triples_processed_index") or 0)
    if not triples or idx >= len(triples):
        _dlog("node: agent_node - no more triples to process.")
        # 沒有 triple，傳訊息讓 accumulate → finalize
        done_msg = AIMessage(content="No more triples.")
        return {"messages": state.get("messages", []) + [done_msg]}

    current_triple = triples[idx]
    current_triple_json = json.dumps(current_triple, ensure_ascii=False)

    # ---- ORG_SEED（無向 OR）偵測：relation 為空 且 僅一側非空（本專案的保底機構種子型態）----
    def _is_org_seed(tp: dict) -> bool:
        try:
            h = (tp.get("head") or tp.get("h") or "").strip()
            r = (tp.get("relation") or tp.get("r") or "").strip()
            t = (tp.get("tail") or tp.get("t") or "").strip()
            return (r == "") and bool(h) ^ bool(t)
        except Exception:
            return False

    org_seed_flag = _is_org_seed(current_triple)
    if org_seed_flag:
        org_mode = os.getenv("ORG_SEED_MODE", "UNDIRECTED")
        org_min_should = os.getenv("ORG_SEED_MIN_SHOULD", "1")
        org_use_rel = os.getenv("ORG_SEED_USE_REL", "0")
        org_allow_inv = os.getenv("ORG_SEED_ALLOW_INVERT", "0")
        # —— 只為顯示而組「有效查詢」三欄（不改動實際傳遞的 triple）——
        org_val = (current_triple.get("head") or current_triple.get("tail") or "").strip()
        eff = {
            "head": org_val,
            "relation": (org_val if str(org_use_rel).lower() in {"1","true","yes","y"} else ""),
            "tail": org_val,
        }
        _dlog(
            "node: agent_node - processing triple "
            f"{idx+1}/{len(triples)} [ORG_SEED:{org_mode} OR] "
            f"(MIN_SHOULD={org_min_should}, USE_REL={org_use_rel}, ALLOW_INVERT={org_allow_inv}) | "
            f"original={current_triple_json} | effective={json.dumps(eff, ensure_ascii=False)}"
        )
    else:
        _dlog(
            "node: agent_node - processing triple "
            f"{idx+1}/{len(triples)}: {current_triple_json}"
        )

    llm = _build_llm().bind_tools(TOOLS, tool_choice="any")
    notes = state.get("notes") or ""
    availability = state.get("tool_availability") or _get_tool_availability()
    sys_prompt = (
        AGENT_SYS_PROMPT
        + f"\n\n【工具可用性】{availability}"
        + f"\n【目前統計】\n{notes}".strip()
    )

    user_msg_content = f"請為這個三元組檢索：\n{current_triple_json}\n請依規則選擇工具並執行。"
    response = llm.invoke([{"role": "system", "content": sys_prompt}, HumanMessage(content=user_msg_content)])

    return {
        "messages": state.get("messages", []) + [response],
        "tool_availability": availability,
        "triples_processed_index": idx + 1,  # 下一輪換下一個 triple
    }


tools_node = ToolNode(TOOLS)


def _extract_kb_lines_from_messages(messages: List[BaseMessage]) -> List[str]:
    """由訊息序列萃取所有出現的 [比對] 行（含工具 JSON 的 lines）。"""
    text = ""
    # 聚合純文字
    for m in messages[::-1]:
        if isinstance(m, (AIMessage, ToolMessage)) and hasattr(m, "content"):
            if isinstance(m.content, str):
                text += "\n" + m.content
            elif isinstance(m.content, list):
                for c in m.content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += "\n" + (c.get("text") or "")

    lines: List[str] = []
    # ① 行首 [比對]
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("[比對]") or re.match(r"^\[\d+\]\s*", s):
            if s.startswith("[比對]"):
                lines.append(s)
            else:
                s2 = re.sub(r"^\[\d+\]\s*", "", s)
                if s2.startswith("[比對]"):
                    lines.append(s2)

    # ② 解析 JSON 取 {"lines":[...]}
    for m in messages[::-1]:
        if isinstance(m, (AIMessage, ToolMessage)) and hasattr(m, "content"):
            raw = m.content if isinstance(m.content, str) else ""
            raw = (raw or "").strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and isinstance(obj.get("lines"), list):
                    for s in obj["lines"]:
                        s = str(s or "").strip()
                        if not s:
                            continue
                        lines.append(s if s.startswith("[比對]") else f"[比對] {s}")
            except Exception:
                # 非 JSON 略過
                pass

    return deduplicate(lines, triples=None)


def accumulate_node(state: AgentState) -> Dict:
    """累積比對行，統計步數與連續無新增計數。"""
    prev = state.get("accum_lines", []) or []
    now = _extract_kb_lines_from_messages(state.get("messages", []) or [])
    merged = deduplicate(prev + now, triples=None)
    no_new = state.get("no_new_steps", 0) or 0
    if len(merged) > len(prev):
        no_new = 0
    else:
        no_new += 1
    step = int(state.get("step") or 0) + 1
    tgt_note = "無下限（僅以耐心值/步數控制）" if AGENT_TOTAL_TARGET <= 0 else f"總行數≥{AGENT_TOTAL_TARGET}"
    per_note = "不檢查" if AGENT_MIN_PER_TRIPLE <= 0 else f"每 triple≥{AGENT_MIN_PER_TRIPLE}"
    notes = (
        f"累積比對行數：{len(merged)}；步數：{step}；連續無新增：{no_new}\n"
        f"條件：{tgt_note}、{per_note}；上限：{AGENT_TOP_K_MAX}"
    )
    return {
        "accum_lines": merged,
        "no_new_steps": no_new,
        "step": step,
        "notes": notes,
    }


def _min_per_triple_ok(lines: List[str], triples: List[Dict[str, str]]) -> bool:
    """檢查每個三元組是否至少被支持指定次數（若啟用）。"""
    if AGENT_MIN_PER_TRIPLE <= 0:
        return True
    if not triples:
        return False
    counter: Dict[str, int] = {}
    for ln in lines:
        # 統一要求必含【關係名】
        m = re.search(r"\[比對\]\s*(.+?)\s*透過關係【.+?】與\s*(.+?)\s*建立連結", ln)
        if m:
            key = m.group(1).strip() + "｜" + m.group(2).strip()
            counter[key] = counter.get(key, 0) + 1
    for tp in triples:
        h = (tp.get("head") or "").strip()
        t = (tp.get("tail") or "").strip()
        ok = any(
            ((h and h in k) or (t and t in k)) and v >= AGENT_MIN_PER_TRIPLE
            for k, v in counter.items()
        )
        if not ok:
            return False
    return True


def should_continue(state: AgentState) -> str:
    """決策是否繼續或完成（無下限時僅以耐心/步數收斂）。"""
    lines = state.get("accum_lines", []) or []
    triples = state.get("triples", []) or []
    idx = int(state.get("triples_processed_index") or 0)
    step = int(state.get("step") or 0)
    no_new = int(state.get("no_new_steps") or 0)

    # 只有在設定 > 0 時才啟用門檻判斷
    total_cond = AGENT_TOTAL_TARGET > 0 and len(lines) >= AGENT_TOTAL_TARGET
    per_cond = _min_per_triple_ok(lines, triples)

    # 若已處理完所有 triples，直接收斂
    if triples and idx >= len(triples) and step > 0:
        _dlog("assess: all triples processed, finalizing.")
        return "finalize"

    cap = step >= AGENT_MAX_STEPS
    patience = no_new >= AGENT_NO_NEW_PATIENCE

    _dlog(
        f"assess: lines={len(lines)} total_cond={total_cond} per_ok={per_cond} "
        f"step={step} cap={cap} no_new={no_new} patience={patience}"
    )

    # 提前終止：若累積比對行超過上限，也立即 finalize（避免爆 token）
    if AGENT_TOP_K_MAX > 0 and len(lines) > AGENT_TOP_K_MAX:
        _dlog("early stop: accumulated lines exceed top_k_max")
        return "finalize"

    if total_cond or (AGENT_MIN_PER_TRIPLE > 0 and per_cond) or cap or patience:
        return "finalize"
    return "agent"


def finalize_node(state: AgentState) -> Dict:
    """最終輸出節點，組合 [原始文本] 與 [比對知識]。"""
    news = (state.get("news_text") or "").strip()
    lines = state.get("accum_lines", []) or []

    # 僅在最後套用上限；不設下限
    if AGENT_TOP_K_MAX > 0 and len(lines) > AGENT_TOP_K_MAX:
        lines = lines[:AGENT_TOP_K_MAX]

    # 移除多餘的前綴（[5]、[比對]），統一由這裡編號
    cleaned: List[str] = []
    for ln in lines:
        s = re.sub(r"^\s*\[比對\]\s*", "", ln)
        s = re.sub(r"^\s*\[\d+\]\s*", "", s)
        cleaned.append(s.strip())

    numbered = [f"[{i}] {t}" for i, t in enumerate(cleaned, 1)]
    body = "[原始文本]\n" + news + "\n\n[比對知識]\n" + "\n".join(numbered)
    return {"messages": state.get("messages", []) + [AIMessage(content=body)]}


def _route_from_agent(state: AgentState) -> str:
    """由代理輸出決定路由到 tools 或 accumulate。"""
    last = state.get("messages", [])[-1] if state.get("messages") else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "accumulate"


def build_graph():
    """建置 LangGraph 並編譯。"""
    graph = StateGraph(AgentState)
    graph.add_node("extract_triples", extract_triples_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("accumulate", accumulate_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "extract_triples")
    graph.add_edge("extract_triples", "agent")
    graph.add_conditional_edges(
        "agent", _route_from_agent, {"tools": "tools", "accumulate": "accumulate"}
    )
    graph.add_edge("tools", "accumulate")
    graph.add_conditional_edges(
        "accumulate", should_continue, {"agent": "agent", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=None)


# =========================
# 對外 API
# =========================

def run_retrieval(
    news_text: str, *, session_id: str = "default", model: str | None = None
) -> List[str]:
    """執行 ReAct 迴圈並以串流列印過程，回傳最終 [比對] 行清單。"""
    original_len = len(news_text or "")
    if original_len > MAX_AGENT_CHARS:
        news_text = (news_text or "")[:MAX_AGENT_CHARS]
        _dlog(f"guard: truncate news_text from {original_len} to {len(news_text)}")
    else:
        _dlog(f"guard: news_text chars={original_len}")

    user_msg = HumanMessage(content=f"輸入新聞：\n{news_text.strip()}\n\n請開始依規則執行。")
    run_tid = f"{session_id}-{uuid4().hex}"
    _dlog(f"graph_run: thread={run_tid} | availability={_get_tool_availability()}")

    graph = build_graph()

    bar = tqdm(total=AGENT_MAX_STEPS, desc="④ Agent 檢索中（ReAct 迴圈）", leave=True)
    messages_last: List[BaseMessage] = []
    accum_lines_last: int = 0

    print("──────── Agent Streaming 開始 ────────")
    try:
        for update in graph.stream(
            {
                "messages": [user_msg],
                "news_text": news_text,
                "triples": [],
                "accum_lines": [],
                "step": 0,
                "no_new_steps": 0,
                "tool_availability": _get_tool_availability(),
            },
            config={"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": run_tid}},
            stream_mode="updates",
        ):
            node_name = list(update.keys())[0] if update else "unknown"
            delta = update.get(node_name, {}) or {}

            if "messages" in delta:
                messages_last = delta["messages"]
                last = messages_last[-1] if messages_last else None
                if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                    print(f"• {node_name}: 產生工具呼叫 → {len(last.tool_calls)} 個")
                    _dlog(f"stream: {node_name} tool_calls={len(last.tool_calls)}")
                elif isinstance(last, ToolMessage):
                    preview = str(last.content)[:120].replace("\n", " ")
                    print(f"• {node_name}: 工具回覆片段 ← {preview}...")
                    _dlog(f"stream: {node_name} tool_msg_len={len(str(last.content))}")
                elif isinstance(last, AIMessage):
                    preview = str(last.content)[:120].replace("\n", " ")
                    print(f"• {node_name}: AI 回覆片段 ← {preview}...")
                    _dlog(f"stream: {node_name} ai_msg_len={len(str(last.content))}")

            if "accum_lines" in delta:
                cur = len(delta["accum_lines"] or [])
                if cur != accum_lines_last:
                    print(f"  ↳ 累積比對行：{cur}（+{cur - accum_lines_last}）")
                    _dlog(
                        f"stream: {node_name} accum_lines={cur} (+{cur - accum_lines_last})"
                    )
                    accum_lines_last = cur

            if "step" in delta:
                step_now = int(delta["step"] or 0)
                if step_now > bar.n:
                    bar.n = min(step_now, AGENT_MAX_STEPS)
                    bar.refresh()
    except GraphRecursionError as e:
        # 安全降落：即時輸出當前累積的 [比對] 行，避免整段失敗後無輸出
        _dlog(f"GraphRecursionError: {e} — fallback to finalize with current state")
        print("⚠️  達到 LangGraph recursion_limit，上限=", RECURSION_LIMIT, "：以目前累積結果提前結束。")
        # 將目前 messages_last 轉為 [比對] 行，與 finalize() 的邏輯一致
        now_lines = _extract_kb_lines_from_messages(messages_last)
        now_lines = now_lines[:AGENT_TOP_K_MAX] if AGENT_TOP_K_MAX > 0 else now_lines
        # 不在此 return；讓下方 finalize/解析沿用 messages_last 輸出
    print("──────── Agent Streaming 結束 ────────")

    # 從最後訊息抽出 [比對] 行
    final_msg = messages_last[-1] if messages_last else AIMessage(content="")
    content = final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)
    lines: List[str] = []
    take = False
    for raw in (content or "").splitlines():
        s = raw.strip()
        if s == "[比對知識]":
            take = True
            continue
        if take and re.match(r"^\[\d+\]\s*", s):
            lines.append(s)

    _dlog(f"graph_done: kb_lines={len(lines)}")

    # 保底：抽取→向量檢索
    if not lines:
        _dlog("fallback: no kb lines; trying one-shot vector_search with extracted triples")
        try:
            raw = extract_entities_relations(news_text) or ""
            obj = json.loads(raw.replace("`", "")) if raw else {}
            triples = _er_to_triples(obj)
            if triples:
                from itertools import islice

                tops: List[str] = []
                for tp in islice(triples, 0, 5):
                    tops.extend(_vector_search_impl(tp, top_k=50))
                tops = deduplicate(tops, triples=None)
                # 僅在保底時先加上 [比對] 讓 finalize 能清理
                tops = [f"[比對] {t}" if not t.startswith("[比對]") else t for t in tops]
                lines = [f"[{i}] {re.sub(r'^\\[\\d+\\]\\s*', '', ln)}" for i, ln in enumerate(tops[:min(20, AGENT_TOP_K_MAX)], 1)]
                _dlog(f"fallback: produced {len(lines)} lines")
        except Exception:
            _dlog("fallback error:\n" + traceback.format_exc())

    return lines


def run_factcheck_middle(
    news_text: str, *, session_id: str = "default", model: str | None = None
) -> str:
    """產生中繼輸出：[原始文本] + [比對知識]。"""
    kb_lines = run_retrieval(news_text, session_id=session_id, model=model)
    # kb_lines 已是編號行，這裡維持一致輸出
    body = "[原始文本]\n" + news_text.strip() + "\n\n[比對知識]\n" + "\n".join(kb_lines)
    _dlog(f"run_factcheck_middle: lines={len(kb_lines)}, body_chars={len(body)}")
    return body


# =========================
# 內建 CLI（新增 alias / pg）
# =========================

def _cmd_alias(_: List[str]) -> int:
    """代理執行：產生/更新 alias_rules.json 骨架（委派 retriever 的 alias）。"""
    try:
        from src.qa.tools.property_graph import li_csv_pg_retriever as retr_mod
        if hasattr(retr_mod, "_cmd_alias"):
            return int(retr_mod._cmd_alias([]))
        # fallback：提示直接呼叫 retriever
        print("提示：亦可直接執行 retriever 的 alias：")
        print("  python -m src.qa.preliminary_work.build_csv_property_graph alias")
        return 0
    except SystemExit as e:
        return int(e.code)
    except Exception:
        print("❌ alias 生成失敗：")
        print(traceback.format_exc())
        return 1


def _cmd_pg(args: List[str]) -> int:
    """偵錯用：直接對 PG 發查詢（輸出 lines/hits）。"""
    if not args:
        print("用法：pg '<triple_json>' [top_k]")
        return 2
    tri_raw = args[0]
    top_k = int(args[1]) if len(args) >= 2 else max(50, int(os.getenv("LI_PG_TOPK", "50")))
    # StructuredTool 不是可呼叫函式；需用 .func(...) 或 .invoke(...)
    res = tool_pg_search.func(triple_json=tri_raw, top_k=top_k, hops=int(os.getenv("LI_PG_HOPS", "3")))
    print(res)
    return 0


# =========================
# CLI 進入點
# =========================

def main() -> None:
    """命令列進入點。"""
    import sys

    if len(sys.argv) > 1:
        # [CHANGED] 新增 alias / pg 子命令
        cmd = sys.argv[1].lower()
        if cmd == "alias":
            raise SystemExit(_cmd_alias(sys.argv[2:]))
        if cmd == "pg":
            raise SystemExit(_cmd_pg(sys.argv[2:]))

        # 舊用法：第一個參數視為新聞 id 或路徑
        arg = sys.argv[1]
        filename = arg if arg.endswith(".txt") else f"{arg}.txt"
        path = Path(filename)
        if not path.is_file():
            path = USER_INPUT_DIR / filename
        if not path.is_file():
            raise SystemExit(f"❌ 找不到檔案：{path.resolve()}")
        news_text = path.read_text(encoding="utf-8-sig").strip()
        news_id = path.stem
        
        # 從 input 推導 run_id 並設定環境變數，確保 CURRENT_RUN_ID 生效
        # 此設定主要用於日誌追蹤，保留無妨)
        run_id = _derive_run_id_from_input(news_id)
        os.environ["NEWS_RUN_ID"] = run_id
        global CURRENT_RUN_ID
        CURRENT_RUN_ID = run_id
        
        print(f"📰 開始處理檔案：{path.resolve()} (Run ID: {run_id})")
        print(f"➡️  輸出目錄：{RES_DIR.resolve()}")
        middle = run_factcheck_middle(news_text, session_id=news_id)
        out_path = RES_DIR / f"news_kg_{news_id}_agent.txt"
        out_path.write_text(middle, encoding="utf-8-sig")
        print(f"✅ 完成 ▶ 中間檔：{out_path.resolve()}")
        return
    print("用法：python -m src.qa.verifier.agent_langchain <news_id|檔名>")
    print("      python -m src.qa.verifier.agent_langchain alias")
    print("      python -m src.qa.verifier.agent_langchain pg '<triple_json>' [top_k]")


if __name__ == "__main__":
    main()