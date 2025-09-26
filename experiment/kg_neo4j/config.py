"""
Configuration and Connection Module

此模組負責讀取 .env 設定，設定 Neo4j、LM Studio API 及 MongoDB 連線資訊，
並建立相應的資料庫連線。模組內也包含測試連線的函式，但僅在直接執行時運行，
避免在跨檔案 import 時執行測試程式碼。
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymongo import MongoClient

# 載入環境變數
load_dotenv()

# --------------------------
# Neo4j 資料庫配置
# --------------------------
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI"),
    "user": os.getenv("NEO4J_USER"),
    "password": os.getenv("NEO4J_PASSWORD"),
    "database": os.getenv("NEO4J_DATABASE", "neo4j")  # 預設為 neo4j
}

# --------------------------
# LM Studio API 配置
# --------------------------
LM_STUDIO_CONFIG = {
    "endpoint": os.getenv("MODEL_CONFIG_endpoint"),  # LM Studio API 路徑
    "model_id": os.getenv("MODEL_ID"),  # 使用的模型 ID
    "temperature": 0.1,  # 降低溫度以提升確定性
    "max_tokens": 12288,  # 最大生成字數
    "top_k": 40,  # 限制詞彙選擇範圍
    "top_p": 0.9,  # 控制選擇詞的機率範圍
    "min_p": 0.07,  # 保留一定機率的低頻詞
    "repeat_penalty": 1.1,  # 避免重複產生相同詞彙
}

print(f"Loadded Database: {NEO4J_CONFIG['database']}")

if not LM_STUDIO_CONFIG["endpoint"]:
    raise ValueError("❌ 未正確讀取 .env 中的 MODEL_CONFIG_endpoint，請檢查 .env 設定！")

# --------------------------
# MongoDB 配置
# --------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("❌ 未正確讀取 .env 中的 MONGODB_URI，請檢查 .env 設定！")

# 建立 Neo4j 與 MongoDB 連線
driver = GraphDatabase.driver(
    NEO4J_CONFIG["uri"],
    auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
)
mongo_client = MongoClient(MONGODB_URI)


def test_connections() -> None:
    """測試連線至 Neo4j 與 MongoDB"""
    # 測試 Neo4j 連線（⚠️ 必須指定 database 才會切換）
    with driver.session(database=NEO4J_CONFIG["database"]) as session:
        # 顯示資料庫名稱
        db_result = session.run("CALL db.info() YIELD name RETURN name")
        db_name = db_result.single()["name"]

        # 顯示歡迎訊息
        result = session.run("RETURN 'Neo4j: Hello Neo4j!' AS greeting")
        greeting = result.single()["greeting"]

        print(f"{greeting}（目前使用的資料庫：{db_name}）")

    # 測試 MongoDB 連線
    try:
        server_info = mongo_client.server_info()
        print(f"成功連線至 MongoDB，版本為: {server_info['version']}")
    except Exception as e:
        raise ConnectionError(f"❌ 無法連線至 MongoDB: {e}")


if __name__ == '__main__':
    test_connections()
