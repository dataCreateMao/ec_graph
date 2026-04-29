from calendar import EPOCH
from pathlib import Path

# 1. 目录路径
ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / 'data'
NER_DIR = 'ner'
RAW_DATA_DIR = DATA_DIR / NER_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / NER_DIR / 'processed'

LOG_DIR = ROOT_DIR / 'logs'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'

# 2. 数据文件 和 模型名称
RAW_DATA_FILE = str(RAW_DATA_DIR / 'data.json')
MODEL_NAME = 'google-bert/bert-base-chinese'

# 3. 模型参数
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 5e-5

SAVE_STEPS = 20

# 4. NER 任务分类标签
LABELS = ['B','I','O']

# MYSQL 数据库连接信息
MYSQL_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'Aa123456',
    'database': 'gmall',
}

NEO4J_CONFIG = {
    'uri': 'neo4j://localhost:7687' ,
    'auth': ('neo4j', 'Aa123456'),
}

# Embedding 后端配置：支持 local / api
# - local: 使用 HuggingFaceEmbeddings（本地 sentence-transformers）
# - api: 使用 OpenAI 兼容 Embeddings API（通过 base_url + api_key 调用）
EMBEDDING_CONFIG = {
    "backend": "api",  # "local" | "api"
    "local": {
        "model_name": "BAAI/bge-small-zh-v1.5",
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "api": {
        # 优先从 .env 读取 ZAI_* 变量；未配置时回退到默认值
        "model_env": "ZAI_MODEL",
        "model": "embedding-3",
        "url_env": "ZAI_AI_URL",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZAI_API_KEY",
    },
}

