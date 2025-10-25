"""機構名保底種子抽取（LLM + 正則 fallback）。"""

from __future__ import annotations

import json
import os
import re
from typing import List

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

from ..common.config import (
    ENABLE_ORG_SEEDS,
    MAX_ORG_SEEDS,
    OPENAI_CHAT_MODEL,
)
from ..common.config import _dlog  # noqa: F401  # 供外部使用
from ..common.json_utils import parse_json_safely

# 擴展後綴（略）
_ORG_SUFFIXES_CLEANED_GROUPS = [
    "股份有限公司|有限公司|控股公司|金控公司|資產管理|傳播公司|出版公司|影視公司|行銷公司|廣告公司|設計公司|投資公司|科技公司|工程公司|建設公司|服務中心|研究中心|檢驗所|認證機構|評估公司|鑑定公司|設計工作室|聯合事務所|律師事務所|會計師事務所|建築師事務所|專利事務所|資產管理公司|管理顧問公司|顧問公司|實驗室|工作室|事務所|總公司|企業|集團|公司|商行|商號|商店|商社|商會|行號|廠|工廠|製造廠|製藥廠|藥廠|銀行|證券|保險公司|經紀公司|代理商|經銷商|開發公司|實業公司|服務社|服務隊|合作社|合作金庫",
    "社區發展協會|非營利組織|慈善機構|慈善團體|志願團體|公民組織|人民團體|保育協會|發展協會|推廣會|促進會|聯誼會|宗親會|志工團|社會服務中心|學會|協會|基金會|中心|研究院|研究所|機構|組織|工會|公會|聯盟|社|社團|社群|自救會|互助會|福利會|青年會|婦女會|老人會|環保團體",
    "社區大學|空中大學|實驗學校|教育中心|教育機構|技藝中心|圖書館|博物館|紀念館|文化館|美術館|科學館|天文台|動物園|植物園|大學|學院|學校|中學|小學|幼兒園|補習班|安親班",
    "出版集團|廣播公司|通訊社|新聞社|電視公司|電視臺|電視台|廣播電臺|廣播電台|頻道|出版社|期刊|雜誌|季刊|月報|週報|晚報|時報|日報|報",
    "醫療中心|醫療院所|護理之家|養護中心|復健中心|心理諮商所|心理治療所|捐血中心|動物醫院|獸醫診所|衛生所|醫院|診所|血庫",
    "立法院|市議會|縣議會|鄉鎮市民代表會|檢察署|特勤中心|情報機關|駐外代表處|辦事處|鄉公所|鎮公所|區公所|里辦公處|村辦公處|警察局|消防局|國會|議會|政黨|黨|法院|院|部|署|局|處|會|司|廳",
    "高鐵公司|港務局|轉運站|物流中心|航運公司|海運公司|船運公司|客運公司|捷運公司|鐵路局|航空站|機場|港口|計程車行|租車公司|快遞公司|貨運公司|郵政公司|郵局",
    "運動中心|體育館|運動場|體育協會|體育會|俱樂部|表演廳|劇團|樂團|唱片公司|電影院|影城|展覽館|會議中心|遊樂園|休閒農場|度假村|溫泉會館|遊戲公司|動畫公司|漫畫公司|球隊|健身房",
    "農業改良場|農業試驗所|水利會|農會|漁會|農場|牧場|漁場|畜牧場|林場|茶廠|酒莊|水產公司",
    "禮拜堂|修道院|清真寺|教會|教堂|佛堂|精舍|道觀|寺|廟|宮|堂|庵|祠|壇|院|殿|觀",
]
_EN_ORG_SUFFIXES = r"(?:Inc\.|Incorporated|Ltd\.|Limited|LLC|L\.L\.C\.|PLC|P\.L\.C\.|GmbH|AG|S\.A\.|S\.p\.A\.|Corp\.|Corporation|Company|Co\.)"

_ORG_SUFFIXES_CLEANED = "|".join(sum((grp.split("|") for grp in _ORG_SUFFIXES_CLEANED_GROUPS), []))

_ORG_REGEX = re.compile(
    rf"(?<![\w\u4e00-\u9fff])"
    rf"(?:[《「『(【])?"
    rf"([A-Za-z0-9\u4e00-\u9fff·＆&．\.\-（）()／\s]{{2,40}})"
    rf"(?:{_ORG_SUFFIXES_CLEANED}|{_EN_ORG_SUFFIXES})"
    rf"(?![\w\u4e00-\u9fff])"
)

_ORG_STOP_SINGLE = {
    "政府", "媒體", "公司", "集團", "中心", "研究所", "研究院", "學院", "大學", "學校",
    "法院", "檢察署", "委員會", "電視台", "電視臺", "電台", "電臺", "報", "醫院", "診所", "寺", "廟",
}


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, max_tokens=1024)


def _normalize_org_name(s: str) -> str:
    OPEN_QUOTES = "“„‟‹«『「《(（[【"
    CLOSE_QUOTES = "”‟”›»』」》)）]】"
    s = (s or "").strip()

    changed = True
    while changed and s:
        changed = False
        while s and (s[0] in OPEN_QUOTES or s[0] in "\"'" or s[0].isspace()):
            s = s[1:]
            changed = True
        while s and (s[-1] in CLOSE_QUOTES or s[-1] in "\"'" or s[-1].isspace()):
            s = s[:-1]
            changed = True

    s = re.sub(r"(?:^)(以|由|據稱?|對於|針對)\s*", "", s)
    s = re.split(r"[，,。．\.、;；：:！？!?\n]", s)[0]
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_org_seeds(text: str) -> List[str]:
    """從原文抽取機構名作為保底檢索種子。"""
    if not ENABLE_ORG_SEEDS:
        return []

    out: List[str] = []
    # 1) LLM 嚴格 JSON 抽取
    try:
        llm = _build_llm()
        sys_prompt = (
            "任務：找出文本中出現的機構/媒體/公司/協會/基金會/政黨/政府單位名稱；"
            "只輸出 JSON：{\"organizations\":[...]}。不得包含人名、地名或過度抽象詞彙。"
        )
        resp = llm.invoke(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"[文本]\n{text[:12000]}"}]
        )
        raw = resp.content if isinstance(resp.content, str) else json.dumps(resp.content, ensure_ascii=False)
        obj = parse_json_safely(raw)
        arr = obj.get("organizations") if isinstance(obj, dict) else None
        if isinstance(arr, list):
            out = [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass

    # 2) 正則補齊
    regex_hits = [m.group(0).strip() for m in _ORG_REGEX.finditer(text or "")]
    out.extend(regex_hits)

    # 3) 清理去重
    seen = set()
    cleaned: List[str] = []
    for name in out:
        name = _normalize_org_name(name)
        n = re.sub(r"\s+", "", name)
        if not (2 <= len(n) <= 64):
            continue
        if n in _ORG_STOP_SINGLE:
            continue
        if len(n) <= 3 and n[-1] in {"院", "部", "局", "處", "會", "司", "報", "寺", "廟", "堂"}:
            core = n[:-1]
            core = re.sub(r"[（）()《》「」『』·＆&．\.\-/\s]+", "", core)
            if len(core) < 2:
                continue
        key = re.sub(r"[\"'“”„‟‹›«»『』「」《》()（）\[\]【】]", "", n)
        if key not in seen:
            seen.add(key)
            cleaned.append(name.strip())
    return cleaned[:MAX_ORG_SEEDS] if MAX_ORG_SEEDS > 0 else cleaned
