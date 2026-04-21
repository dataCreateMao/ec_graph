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


