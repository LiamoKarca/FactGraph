import hashlib

def generate_hash(title: str) -> str:
    """對新聞標題進行 SHA-256 哈希轉換，取前 8 碼作為識別碼"""
    hash_object = hashlib.sha256(title.encode('utf-8'))
    return hash_object.hexdigest()[:8]  # 取前 8 碼

def generate_news_id(publisher: str, title: str, date: str) -> str:
    """根據 publisher、title（哈希）與 date 生成新聞 ID"""
    title_hash = generate_hash(title)
    formatted_date = date.replace("/", "")  # 移除日期中的 "/"
    return f"{publisher}_{title_hash}_{formatted_date}"

# 測試用文本
publisher = "CNA"
title = "駐瑞典代表投書：對台威脅就是對全世界威脅"
date = "2025/2/25"

# 生成 ID
news_id = generate_news_id(publisher, title, date)

# 顯示結果
print("測試新聞 ID:", news_id)