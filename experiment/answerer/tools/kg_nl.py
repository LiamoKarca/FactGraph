"""
KG → 自然語言模組

本模組提供將知識圖譜三元組轉換為自然語言描述的工具：
  - `_fmt_props`      : 屬性字典格式化
  - `format_entity`   : 組裝實體描述
  - `verbalize`       : 將單一三元組轉為句子
  - `build_block`     : 批量生成多行描述

修訂：若關係屬性含 `date` 或 `time`，會在敘述末端附加事件時間。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

# 排除不需顯示的屬性鍵
_EXCLUDE_KEYS: set[str] = {'id', 'doc_id', 'name'}


def _fmt_props(props: Dict[str, Any]) -> str:
    """將屬性字典轉為 「k:v; k:v」 形式的文字串。"""
    items = (
        f"{k}:{v}"
        for k, v in props.items()
        if k not in _EXCLUDE_KEYS and v is not None
    )
    return '；'.join(items)


def format_entity(
        name: str,
        props: Dict[str, Any],
        *,
        role: str = '未知實體'
) -> str:
    """
    組裝實體描述：
      - 若 props 非空，顯示 "名稱（屬性…）"
      - 否則顯示 "名稱（role）"

    Args:
        name: 實體名稱
        props: 實體屬性字典
        role: 無屬性時的替代描述
    Returns:
        完整實體描述文字
    """
    formatted = _fmt_props(props)
    if not formatted:
        return f'{name}（{role}）'

    # 若未提供 type，且 props 包含 type
    if 'type' in props and props.get('type') and 'type:' not in formatted:
        formatted = f"類型:{props['type']}" + ('；' + formatted if formatted else '')

    return f'{name}（{formatted}）'


def verbalize(
        idx: int,
        head: str,
        relation: str,
        tail: str,
        head_props: Dict[str, Any],
        rel_props: Dict[str, Any],
        tail_props: Dict[str, Any]
) -> str:
    """
    將單條三元組及其屬性轉為自然語言描述句。

    Args:
        idx: 編號
        head: 主體名稱
        relation: 關係名稱
        tail: 受體名稱
        head_props: 主體屬性
        rel_props: 關係屬性
        tail_props: 受體屬性
    Returns:
        格式化後的描述句，例如：
        "1. A（屬性…）透過關係【rel】與 B（屬性…）建立連結…。"
    """
    head_desc = format_entity(head, head_props, role='主體')
    tail_desc = format_entity(tail, tail_props, role='受體')
    desc = rel_props.get('evidence', relation)
    date = rel_props.get('date') or rel_props.get('time')
    time_part = f'；事件時間：{date}' if date else ''

    return (
        f'{idx}. {head_desc} 透過關係【{relation}】'
        f'與 {tail_desc} 建立連結，說明：{desc}{time_part}。'
    )


def build_block(
        triples: List[Dict[str, str]],
        detail_map: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]]
) -> str:
    """
    將多條三元組及對應屬性映射轉為多行自然語言區塊。

    Args:
        triples: 三元組列表，每個 dict 包含 'head','relation','tail'
        detail_map: 以 (head,relation,tail) 為鍵對應 'head','rel','tail' 屬性字典
    Returns:
        多行字串，每行為編號後的描述句
    """
    lines: List[str] = []
    for idx, tp in enumerate(triples, start=1):
        key = (tp['head'], tp['relation'], tp['tail'])
        props = detail_map.get(key, {})
        lines.append(
            verbalize(
                idx,
                tp['head'], tp['relation'], tp['tail'],
                props.get('head', {}),
                props.get('rel', {}),
                props.get('tail', {})
            )
        )
    return '\n'.join(lines)
