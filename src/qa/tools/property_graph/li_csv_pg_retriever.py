"""
CSV → Property Graph 檢索器（JSON-only）
【職責單一化（SoC/SRP）】只負責「檢索」。
※ 建置 / 別名 alias / 健康檢查 → 全部在
   src/qa/preliminary_work/build_csv_property_graph.py

使用方式：
  - 檢索（本模組）：
      python -m src.qa.tools.property_graph.li_csv_pg_retriever query '<triple_json>' [top_k]
  - 建置/alias/doctor（建置模組）：
      python -m src.qa.preliminary_work.build_csv_property_graph <subcmd>

檢索範例：
python -m src.qa.tools.property_graph.li_csv_pg_retriever query '{"head":"黃國昌","relation":"涉嫌","tail":"狗仔團隊"}' 20
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path
import json
import os
import re
import sys
import time
import traceback
import unicodedata
import tempfile
import signal
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 供「即時查詢擴充」使用（沿用現成快取函式）
# 與原專案一致：search.py 的快取式擴充器
from ...verifier.kg.search import _expand_terms_with_gpt_cached  # type: ignore

# 直接使用專案內的 GPTClient（若缺則退回舊的無標注擴充）
try:
    from ...verifier.llm.gpt import GPTClient  # type: ignore
except Exception:
    GPTClient = None  # pragma: no cover

# 由建置模組引入「資料結構」與「自動建置 API」
from ...preliminary_work.build_csv_property_graph import (  # noqa: E402
    CsvPropertyGraph,
    Node,
    ensure_built_json as _builder_ensure_built_json,
)

# ── Ctrl-C 優雅收尾：收到 SIGINT 時寫入目前增量後結束 ──
_STOP = False


def _graceful_stop(signum, frame):
    del signum, frame
    global _STOP
    _STOP = True


try:
    signal.signal(signal.SIGINT, _graceful_stop)
except Exception:
    pass

load_dotenv()

# ─────────────────────────
# 預設路徑
# ─────────────────────────
CACHE_DIR = Path("data/processed/knowledge-graph")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
JSON_CACHE_PATH = Path(
    os.getenv("LI_PG_INDEX_JSON", str(CACHE_DIR / "pg_index.json")))
RAW_CSV_DEFAULT = Path(
    os.getenv("LI_PG_RAW_CSV", "data/raw/knowledge-graph/neo4j-kg-raw-graph.csv")
)

# ─────────────────────────
# 檢索器（JSON-only）
# ─────────────────────────


class CsvPropertyGraphRetriever:
    """對 CsvPropertyGraph 的輕量檢索封裝（JSON-only）。"""

    def __init__(self, g: CsvPropertyGraph, *, date_field: str = "date", evidence_field: str = "evidence") -> None:
        self.g = g
        self.date_field = date_field
        self.evidence_field = evidence_field
        self._alias_path: Optional[str] = os.getenv("ALIAS_RULES_JSON")
        self._alias_reload_sec: float = float(
            os.getenv("ALIAS_RELOAD_SEC", "0") or 0)
        self._alias_last_load: float = 0.0
        self._alias_rules: Dict[str, Any] = {}
        # relation.hints 僅作提示（非硬過濾）；必要時編譯為 regex
        self._rel_hint_re: Optional[re.Pattern[str]] = None

        # 檢索布林規則可調參
        self._min_should: int = int(
            os.getenv("LI_PG_MIN_SHOULD", "2") or "2")  # H/R/T 至少命中幾個
        self._require_entity: bool = (
            str(os.getenv("LI_PG_REQUIRE_ENTITY", "1")).lower() in {"1", "true", "yes", "y"})
        self._event_pivot: bool = (str(os.getenv("LI_PG_EVENT_PIVOT", "1")).lower() in {
                                   "1", "true", "yes", "y"})
        self._allow_invert: bool = (
            str(os.getenv("LI_PG_ALLOW_INVERT", "1")).lower() in {"1", "true", "yes", "y"})
        # 倒置嚴格（只比對名稱，不吃 props）
        self._invert_strict: bool = (
            str(os.getenv("LI_PG_INVERT_STRICT", "0")).lower() in {"1", "true", "yes", "y"})

        # 正向 props 使用細分成 head/tail
        self._head_use_props: bool = (str(os.getenv("LI_PG_HEAD_USE_PROPS", os.getenv(
            "LI_PG_FORWARD_USE_PROPS", "1"))).lower() in {"1", "true", "yes", "y"})
        self._tail_use_props: bool = (
            str(os.getenv("LI_PG_TAIL_USE_PROPS", "0")).lower() in {"1", "true", "yes", "y"})

        # 檢索期「即時 LLM 擴充」開關（別名不足時用）
        self._llm_runtime_enable: bool = (str(os.getenv(
            "ALIAS_LLM_RUNTIME_ENABLE", "0")).lower() in {"1", "true", "yes", "y"})

    # ====== 檢索：六欄位一致查詢（含同義展開與 props 解包） ======
    @staticmethod
    def _norm_str(s: Any) -> str:
        return str(s or "").strip()

    @staticmethod
    def _normalize_text(s: Any) -> str:
        """NFKC 正規化 + 去除常見雜訊空白：降低全半形、符號差異的影響。"""
        t = str(s or "")
        # 參考 Unicode UAX #15 與 Python unicodedata 文檔
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t.lower()

    @staticmethod
    def _contains(a: str, b: str) -> bool:
        """a 是否包含 b（大小寫不敏感；b 為空視為通配）。"""
        a2, b2 = CsvPropertyGraphRetriever._normalize_text(
            a), CsvPropertyGraphRetriever._normalize_text(b)
        return (b2 == "") or (b2 in a2)

    @staticmethod
    def _unwrap_props(d: Dict[str, Any]) -> Dict[str, Any]:
        """解開雙層 props：{ "props": {...} }。"""
        if isinstance(d, dict) and "props" in d and isinstance(d["props"], dict):
            return d["props"]
        return d if isinstance(d, dict) else {}

    @staticmethod
    def _any_value_contains(d: Dict[str, Any], q: str) -> bool:
        """任一值包含 q（字串化比對）。"""
        d = CsvPropertyGraphRetriever._unwrap_props(d)
        if not isinstance(d, dict):
            return False
        q = CsvPropertyGraphRetriever._normalize_text(q or "")
        if not q:
            return True
        for v in d.values():
            s = CsvPropertyGraphRetriever._normalize_text(
                v if v is not None else "")
            if q in s:
                return True
        return False

    @staticmethod
    def _key_or_value_contains(d: Dict[str, Any], q: str) -> bool:
        """鍵或值任一包含 q（for rel_props）。"""
        d = CsvPropertyGraphRetriever._unwrap_props(d)
        if not isinstance(d, dict):
            return False
        q = CsvPropertyGraphRetriever._normalize_text(q or "")
        if not q:
            return True
        for k, v in d.items():
            ks = CsvPropertyGraphRetriever._normalize_text(
                k if k is not None else "")
            vs = CsvPropertyGraphRetriever._normalize_text(
                v if v is not None else "")
            if q in ks or q in vs:
                return True
        return False

    # 僅允許在少數 key 上做 relation 比對；避免 evidence 長句誤擊
    _REL_KEYS_STRICT = {"relation", "label", "type", "關係", "關係類型"}

    def _rel_key_contains(self, rel_props: dict, q: str) -> bool:
        """relation 比對策略：
        - 一般情況：僅在小集合鍵名上做包含比對，避免 evidence/desc 長文誤擊。
        - 若查詢詞長度 <= 2（例如「由」「被」等功能詞）：放寬為「鍵或值任意包含」。
        """
        if not q:
            return False
        q = self._normalize_text(q)
        rel_props = self._unwrap_props(rel_props)
        if len(q) <= 2:
            return self._key_or_value_contains(rel_props, q)
        for k, v in rel_props.items():
            if k in self._REL_KEYS_STRICT and isinstance(v, str) and q in self._normalize_text(v):
                return True
        hint_re = getattr(self, "_rel_hint_re", None)
        if hint_re and self._any_value_contains(rel_props, hint_re.pattern):
            return True
        return False

    # --------------------------
    # alias_rules 載入 / 原子寫入
    # --------------------------
    def _compile_rel_hint_re(self) -> None:
        """將 relation.hints 編譯為 regex（僅作提示；可為 None）。"""
        try:
            rel_rules = (self._alias_rules or {}).get("relation", {}) or {}
            hints = rel_rules.get("hints") or []
            hints = [str(h).strip() for h in hints if h and str(h).strip()]
            if not hints:
                self._rel_hint_re = None
                return
            hints_sorted = sorted(set(hints), key=len, reverse=True)
            pat = "|".join(re.escape(h) for h in hints_sorted)
            self._rel_hint_re = re.compile(pat)
        except Exception:
            self._rel_hint_re = None

    def _save_alias_rules_atomic(self) -> None:
        """以原子方式覆寫 alias_rules.json。"""
        path = self._alias_path
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self._alias_rules or {}
        if not isinstance(data, dict) or not data:
            data = {"head": {}, "relation": {}, "tail": {}}
            self._alias_rules = data
        # [KEEP] tempfile + os.replace：同檔系統原子換檔
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", dir=str(p.parent), delete=False) as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, str(p))  # 原子更新（tmp -> target）

    def _maybe_load_alias_rules(self) -> None:
        """載入/熱更新同義規則；若未設定路徑或檔案不存在則維持空規則。"""
        if not self._alias_path:
            self._alias_rules = {}
            return
        now = time.time()
        if self._alias_rules and self._alias_reload_sec > 0 and (now - self._alias_last_load) < self._alias_reload_sec:
            if self._rel_hint_re is None:
                self._compile_rel_hint_re()
            return
        try:
            with open(self._alias_path, "r", encoding="utf-8-sig") as f:
                self._alias_rules = json.load(f) or {}
            self._alias_last_load = now
            self._compile_rel_hint_re()
        except FileNotFoundError:
            self._alias_rules = {}
            self._rel_hint_re = None
        except Exception:
            self._alias_rules = {}
            self._rel_hint_re = None

    # --------------------------
    # LLM 取用 / JSON 輔助
    # --------------------------
    def _get_gpt(self):
        """
        建立 GPTClient：
        - api_key：OPENAI_API_KEY 或 GPT_API
        - model_id：優先 ALIAS_LLM_MODEL，其次 OPENAI_CHAT_MODEL，最後 MODEL_ID
        - base_url：MODEL_CONFIG_ENDPOINT（走 OpenAI 相容端點）
        有些專案的 GPTClient 參數簽章不同，因此加上 try/fallback。
        """
        if GPTClient is None:
            return None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API") or ""
        model_id = (
            os.getenv("ALIAS_LLM_MODEL")
            or os.getenv("OPENAI_CHAT_MODEL")
            or os.getenv("MODEL_ID")
            or "gpt-4o-mini"
        )
        base_url = os.getenv("MODEL_CONFIG_ENDPOINT") or os.getenv(
            "OPENAI_BASE_URL") or None
        temperature = float(os.getenv("ALIAS_LLM_TEMPERATURE", "0.0") or "0.0")
        max_tokens = int(os.getenv("ALIAS_LLM_MAX_TOKENS", "2000") or "2000")

        if not api_key and not base_url:
            return None

        try:
            return GPTClient(
                api_key=api_key,
                model_id=model_id,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            try:
                return GPTClient(api_key=api_key, model_id=model_id)
            except Exception:
                return None
        except Exception:
            return None

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        s = (text or "").strip()
        if s.startswith("```"):
            i = s.find("\n")
            if i != -1:
                s = s[i + 1:]
            if s.endswith("```"):
                s = s[:-3]
        return s.strip()

    # --------------------------
    # 智慧化（帶標注）擴充
    # --------------------------
    def _llm_expand_and_label(self, h: str, r: str, t: str) -> Dict[str, List[str]]:
        """
        以 GPT 對 (head, relation, tail) 做「語義擴充 + 角色標注」：
        回傳 {"head": [...], "relation": [...], "tail": [...]}
        ※ 不使用任何固定關係關鍵字或長度/尾綴啟發式。
        """
        gpt = self._get_gpt()
        if gpt is None:
            # 無 GPT 時進入 fallback：至少回傳 relation 的擴充（以現有 Query Expander）
            base = " ".join([h or "", r or "", t or ""]).strip()
            try:
                terms = list(_expand_terms_with_gpt_cached(
                    base)) if base else []
            except Exception:
                terms = []
            terms = [self._normalize_text(x) for x in terms if x]
            return {"head": [], "relation": terms, "tail": []}

        sys_prompt = (
            "你是『三元組查詢擴充器與標注器』。\n"
            "給定 head / relation / tail（可能有誤或不完整），請：\n"
            "1) 產生 head_aliases、relation_terms、tail_aliases（各為字串陣列），皆為可協助檢索的自然語彙或別名；\n"
            "2) 務必包含原詞（若非空）；\n"
            "3) 僅輸出 JSON，格式：\n"
            "{\n"
            "  \"head_aliases\": [\"...\"],\n"
            "  \"relation_terms\": [\"...\"],\n"
            "  \"tail_aliases\": [\"...\"]\n"
            "}\n"
        )
        user_msg = f"head={h!r}\nrelation={r!r}\ntail={t!r}\n請執行擴充與角色標注並只輸出 JSON。"
        try:
            raw = gpt.chat([{"role": "system", "content": sys_prompt}, {
                           "role": "user", "content": user_msg}])
            raw = self._strip_code_fence(raw)
            obj = json.loads(raw)
            head = [self._normalize_text(x) for x in (
                obj.get("head_aliases") or []) if x]
            rel = [self._normalize_text(x) for x in (
                obj.get("relation_terms") or []) if x]
            tail = [self._normalize_text(x) for x in (
                obj.get("tail_aliases") or []) if x]

            def _uniq(arr): return list(dict.fromkeys(arr))
            return {"head": _uniq(head), "relation": _uniq(rel), "tail": _uniq(tail)}
        except Exception:
            return {"head": [], "relation": [], "tail": []}

    # === 以 LLM 在「檢索時」做額外的查詢擴充（alias 不足時） ===
    def _runtime_llm_expand(self, h: str, r: str, t: str) -> Dict[str, List[str]]:
        """改為完全依賴 GPT 標注；若失敗則退回 Query Expander。"""
        labeled = self._llm_expand_and_label(h, r, t)
        if any(labeled.values()):
            return labeled
        base = " ".join([h or "", r or "", t or ""]).strip()
        try:
            terms = list(_expand_terms_with_gpt_cached(base)) if base else []
        except Exception:
            terms = []
        return {"head": [], "relation": [], "tail": [self._normalize_text(x) for x in terms if x]}

    # 為「本次查詢」即時擴充 relation.hints（使用 GPT 標注的 relation_terms），並寫回 alias_rules.json
    def _refresh_relation_hints_for_query(self, seeds: List[str]) -> None:
        """若 GPT 不可用，fallback 使用 Query Expander 的結果補 relation.hints。"""
        try:
            raw = " ".join([s for s in seeds if s]).strip()
            if not raw:
                return
            labeled = self._llm_expand_and_label(seeds[0] if len(seeds) > 0 else "",
                                                 seeds[1] if len(
                                                     seeds) > 1 else "",
                                                 seeds[2] if len(seeds) > 2 else "")
            hints_new = labeled.get("relation", []) if labeled else []
            # Fallback：若仍無結果，用 Query Expander 直接補 relation.hints
            if not hints_new:
                try:
                    hints_new = list(_expand_terms_with_gpt_cached(raw))
                except Exception:
                    hints_new = []
            hints_new = [self._normalize_text(x) for x in hints_new if x]
            if not hints_new:
                return

            if not isinstance(self._alias_rules, dict):
                self._alias_rules = {}
            rel_rules = self._alias_rules.setdefault("relation", {})
            old = rel_rules.get("hints") or []
            merged = list(dict.fromkeys([*old, *hints_new]))
            rel_rules["hints"] = merged
            self._compile_rel_hint_re()
            self._save_alias_rules_atomic()
        except Exception:
            # 略過失敗，不阻斷檢索
            pass

    # --------------------------
    # 別名展開（map/regex/contains/aliases）
    # --------------------------
    def _expand_query_terms(self, field: str, seed: str) -> List[str]:
        """查詢端同義展開（map/regex/contains/aliases）。"""
        if not seed:
            return []
        rules = self._alias_rules.get(field, {}) if isinstance(
            self._alias_rules, dict) else {}
        seed_n = self._normalize_text(seed)
        out = {seed_n}

        # 1) map（精確字面）
        for k, vals in (rules.get("map") or {}).items():
            if seed_n == self._normalize_text(k):
                out.update(vals or [])

        # 2) regex（語形）
        for pair in (rules.get("regex") or []):
            try:
                pat, vals = pair
                if re.search(pat, seed_n):
                    out.update(vals or [])
            except Exception:
                continue

        # 3) contains（子字串觸發）
        for key, vals in (rules.get("contains") or {}).items():
            key_n = self._normalize_text(key)
            if key_n and (key_n in seed_n):
                out.update(vals or [])

        # 4) aliases（專用於 head 實體別名）
        for k, vals in (rules.get("aliases") or {}).items():
            if seed_n == self._normalize_text(k):
                out.update(vals or [])

        return [v for v in out if v]

    # --------------------------
    # 主檢索流程
    # --------------------------
    def search_triple(
        self,
        triple: Dict[str, str],
        *,
        top_k: int = 50,
        hops: int = 1,
    ) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:
        """六欄位檢索（含同義展開與 props 解包）。

        規則：
          - 若 `head` 不空：需命中 e.head 或 e.head_props 任一值（允許 head 別名）。
          - 若 `relation` 不空：需命中 e.relation 或 e.rel_props 的鍵/值（允許同義展開，且僅限少數鍵名）。
          - 若 `tail` 不空：需命中 e.tail 或 e.tail_props 任一值（允許同義展開；可由環境開關）。
          - 無指定者視為通配（不加限制）。
          - 最小命中：H/R/T 三者至少命中 `LI_PG_MIN_SHOULD`（預設 2）
          - 若 `LI_PG_REQUIRE_ENTITY=1`：同時要求 (head 或 tail) 命中至少一者
          - 事件樞紐：tail 類型或名稱含「事件」時允許 (tail ∧ relation) 快速通過（`LI_PG_EVENT_PIVOT=1`）
          - 倒置容忍：必要時以 tail 當作 head 重判一次（`LI_PG_ALLOW_INVERT=1`）
        """
        del hops
        self._maybe_load_alias_rules()

        h_q0 = self._normalize_text(triple.get("head") or triple.get("h"))
        r_q0 = self._normalize_text(triple.get(
            "relation") or triple.get("rel") or triple.get("r"))
        t_q0 = self._normalize_text(triple.get("tail") or triple.get("t"))

        # ① 每次查詢：以本次 (h,r,t) 語境即時擴充 relation.hints 並寫回 alias（無排程）
        if self._llm_runtime_enable:
            self._refresh_relation_hints_for_query([h_q0, r_q0, t_q0])

        # ② alias 展開（map/regex/contains/aliases）
        h_alts = self._expand_query_terms(
            "head", h_q0) or ([h_q0] if h_q0 else [])
        r_alts = self._expand_query_terms(
            "relation", r_q0) or ([r_q0] if r_q0 else [])
        t_alts = self._expand_query_terms(
            "tail", t_q0) or ([t_q0] if t_q0 else [])

        # ③ 即時 LLM 擴充：若 alias 不足（每欄位候選太少），再補一輪（完全依賴 GPT 標注/Fallback）
        if self._llm_runtime_enable and (len(h_alts) <= 1 or len(r_alts) <= 1 or len(t_alts) <= 1):
            extra = self._runtime_llm_expand(h_q0, r_q0, t_q0)
            if extra["head"]:
                h_alts = list(dict.fromkeys(h_alts + extra["head"]))
            if extra["relation"]:
                r_alts = list(dict.fromkeys(r_alts + extra["relation"]))
            if extra["tail"]:
                t_alts = list(dict.fromkeys(t_alts + extra["tail"]))

        out: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
        for e in self.g.edges:
            # 命中紀錄（便於除錯）
            head_hits: List[str] = []
            rel_hits: List[str] = []
            tail_hits: List[str] = []

            # --- head: ---
            if h_alts:
                if self._head_use_props:
                    head_ok = False
                    for hq in h_alts:
                        if self._contains(e.head, hq) or self._any_value_contains(e.head_props, hq):
                            head_ok = True
                            head_hits.append(hq)
                else:
                    head_ok = False
                    for hq in h_alts:
                        if self._contains(e.head, hq):
                            head_ok = True
                            head_hits.append(hq)
            else:
                head_ok = True  # 通配

            # --- relation:（嚴格僅看少數鍵名；短詞才放寬）---
            if r_alts:
                rel_ok = False
                for rq in r_alts:
                    if (self._contains(e.relation, rq) 
                        or self._rel_key_contains(e.rel_props, rq)
                        or self._any_value_contains(e.rel_props, rq)):  # 允許 evidence/desc 直接命中 rq
                        rel_ok = True
                        rel_hits.append(rq)
            else:
                rel_ok = True  # 通配

            # --- tail: ---
            if t_alts:
                if self._tail_use_props:
                    tail_ok = False
                    for tq in t_alts:
                        if self._contains(e.tail, tq) or self._any_value_contains(e.tail_props, tq):
                            tail_ok = True
                            tail_hits.append(tq)
                else:
                    tail_ok = False
                    for tq in t_alts:
                        if self._contains(e.tail, tq):
                            tail_ok = True
                            tail_hits.append(tq)
            else:
                tail_ok = True  # 通配

            # ---------- minimum-should-match + 事件樞紐 + 倒置容忍 ----------
            cond_sum = int(bool(head_ok)) + \
                int(bool(rel_ok)) + int(bool(tail_ok))
            tail_looks_event = (
                (isinstance(self.g.nodes.get(e.tail), Node)
                 and "事件" in (self.g.nodes.get(e.tail).type or ""))
                or ("事件" in (e.tail or ""))
                or ("事件" in str(e.tail_props))
            )
            base_pass = (cond_sum >= max(1, self._min_should)) and (
                (not self._require_entity) or (head_ok or tail_ok))
            event_pivot_pass = (
                self._event_pivot and tail_looks_event and tail_ok and rel_ok)
            base_pass_flag = base_pass
            passed = base_pass_flag or event_pivot_pass

            # 倒置（H↔T）容忍
            invert_pass = False
            if (not passed) and self._allow_invert and (h_alts and t_alts):
                if self._invert_strict:
                    head_ok_inv = any(self._contains(e.head, tq)
                                      for tq in t_alts)  # 倒置只看名稱
                    tail_ok_inv = any(self._contains(e.tail, hq)
                                      for hq in h_alts)
                else:
                    head_ok_inv = any(self._contains(e.head, tq) or self._any_value_contains(
                        e.head_props, tq) for tq in t_alts)
                    tail_ok_inv = any(self._contains(e.tail, hq) or self._any_value_contains(
                        e.tail_props, hq) for hq in h_alts)
                cond_sum_inv = int(bool(head_ok_inv)) + \
                    int(bool(rel_ok)) + int(bool(tail_ok_inv))
                invert_pass = (cond_sum_inv >= max(1, self._min_should)) and (
                    (not self._require_entity) or (head_ok_inv or tail_ok_inv))
                passed = passed or invert_pass

            if not passed:
                continue

            tri = {"head": e.head, "relation": e.relation, "tail": e.tail}
            det = {
                "head": {"name": e.head, "type": self.g.nodes.get(e.head, Node(e.head)).type, **e.head_props},
                "tail": {"name": e.tail, "type": self.g.nodes.get(e.tail, Node(e.tail)).type, **e.tail_props},
                "rel": {"relation": e.relation, **e.rel_props},
                "meta": {
                    "_debug": {
                        "cond_sum": cond_sum,
                        "event_pivot": bool(event_pivot_pass),
                        "used_min_should": self._min_should,
                        "require_entity": self._require_entity,
                        "allow_invert": self._allow_invert,
                        "base_pass": bool(base_pass_flag),
                        "invert_pass": bool(invert_pass),
                        "head_ok": bool(head_ok),
                        "rel_ok": bool(rel_ok),
                        "tail_ok": bool(tail_ok),
                        # [NEW] 附帶各欄位實際命中的候選詞，便於回溯
                        "hits": {
                            "head": head_hits[:8],
                            "rel":  rel_hits[:8],
                            "tail": tail_hits[:8],
                        },
                    }
                },
            }
            out.append((tri, det))
            if 0 < top_k <= len(out):
                break
        return out

    # ---- 載入器 ----
    @classmethod
    def load_from_json(cls, *, idx_json: str | Path = JSON_CACHE_PATH) -> "CsvPropertyGraphRetriever":
        jsn = Path(idx_json)
        if not jsn.is_file():
            g = _builder_ensure_built_json(idx_json=jsn)
            return cls(g)
        g = CsvPropertyGraph.load_json(jsn)
        if not g.is_valid():
            raise ValueError("❌ JSON 索引存在但內容異常（無有效邊或結構錯誤）。")
        return cls(g)

    @classmethod
    def ensure_built_and_loaded_json(
        cls,
        *,
        csv_path: Path | None = None,
        idx_json: str | Path = JSON_CACHE_PATH,
    ) -> "CsvPropertyGraphRetriever":
        g = _builder_ensure_built_json(
            csv_path=csv_path, idx_json=Path(idx_json))
        return cls(g)

    # 相容舊稱
    ensure_built_and_loaded = ensure_built_and_loaded_json


# ─────────────────────────
# 對外工具函式（名稱維持，行為改為 JSON-only）
# ─────────────────────────
def load_cache(idx_json: Path = JSON_CACHE_PATH) -> CsvPropertyGraph:
    """載入圖（JSON-only）。"""
    if idx_json.is_file():
        g = CsvPropertyGraph.load_json(idx_json)
        if not g.is_valid():
            raise ValueError("❌ JSON 索引存在但內容異常。")
        return g
    raise FileNotFoundError(f"❌ 找不到 JSON 索引：{idx_json}")

# ─────────────────────────
# CLI
# ─────────────────────────


def _parse_triple_arg(tri_raw: str) -> Dict[str, str]:
    """允許使用 h/r/t 或 head/relation/tail。"""
    tri = json.loads(tri_raw)
    if not isinstance(tri, dict):
        raise ValueError("triple_json 非 dict")
    return {
        "head": str(tri.get("head") or tri.get("h") or "").strip(),
        "relation": str(tri.get("relation") or tri.get("rel") or tri.get("r") or "").strip(),
        "tail": str(tri.get("tail") or tri.get("t") or "").strip(),
    }


def _cmd_query(args: List[str]) -> int:
    """query 子命令（JSON-only）。"""
    if not args:
        print("用法：query '<triple_json>' [top_k]")
        return 2
    tri_raw = args[0]
    top_k = int(args[1]) if len(args) >= 2 else 10
    try:
        tri = _parse_triple_arg(tri_raw)
    except Exception as e:
        print(f"❌ triple_json 解析失敗：{e}")
        return 3

    retr = CsvPropertyGraphRetriever.load_from_json(idx_json=JSON_CACHE_PATH)
    hits = retr.search_triple(tri, top_k=top_k, hops=1)
    out = []
    for t, d in hits:
        row = {"tri": t, "det": {"head": d.get("head", {}), "rel": d.get(
            "rel", {}), "tail": d.get("tail", {})}}
        if str(os.getenv("LI_PG_CLI_META", "0")).lower() in {"1", "true", "yes", "y"}:
            row["det"]["meta"] = d.get("meta", {})
        out.append(row)
    print(json.dumps(out, ensure_ascii=False))
    return 0


def _safe_int(v: Any, dv: int) -> int:
    """將任意值安全轉為 int，失敗回傳預設值。"""
    try:
        return int(str(v))
    except Exception:
        return dv


def main() -> None:
    """命令列進入點。"""
    if len(sys.argv) <= 1:
        print(
            "用法：python -m src.qa.tools.property_graph.li_csv_pg_retriever "
            "query '<triple_json>' [top_k]\n"
            "※ 建置/alias/doctor：python -m src.qa.preliminary_work.build_csv_property_graph <subcmd>"
        )
        return
    cmd = sys.argv[1].lower()
    args = sys.argv[2:]
    try:
        if cmd == "query":
            raise SystemExit(_cmd_query(args))
        print(f"未知子命令：{cmd}")
        raise SystemExit(2)
    except SystemExit:
        raise
    except Exception:
        print("❌ 執行失敗：")
        print(traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
