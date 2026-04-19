import os
import sys
from pathlib import Path

# 允许直接 `python ner/preprocess.py`（此时 sys.path 不含 `src`）
_src_root = Path(__file__).resolve().parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from datasets import load_dataset
from transformers import AutoTokenizer

from configuration.config import *

def process():
    # 设置 HF_ENDPOINT 使用国内镜像（可选）
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 1. 读取数据
    dataset = load_dataset('json', data_files=RAW_DATA_FILE)['train']
    # print( dataset)

    # 2. 去除多余的列
    dataset = dataset.remove_columns(['id', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time'])

    # 3. 划分数据集
    dataset_dict = dataset.train_test_split(test_size=0.2)
    dataset_dict['test'], dataset_dict['valid'] = dataset_dict['test'].train_test_split(test_size=0.5).values()

    # 4. 定义分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 5. 数据编码(输入文本和标签)
    def encode(example):
        # 5.1 将文本数据转成字符列表
        tokens = list(example['text'])
        
        # 5.2 文本编码
        inputs = tokenizer(tokens, is_split_into_words=True, truncation=True, return_offsets_mapping=True)
        
        # 5.3 进行实体标注 标签编码
        entities = example['label']
        
        # 定义标注列表,默认都为'O'
        labels = ['O'] * len(tokens)
        
        # 遍历每个实体，标记为'B'或'I'
        for entity in entities:
            start, end = entity['start'], entity['end']
            
            # 边界检查
            if start >= end or start < 0 or end > len(tokens):
                continue
                
            # 标记第一个字符为 B，后续字符为 I（需与 configuration.config.LABELS 一致）
            labels[start] = 'B'
            for i in range(start + 1, end):
                labels[i] = 'I'
        
        # 5.4 将标签转换为 token 级别，并处理 subword 映射
        word_ids = inputs.word_ids()
        token_labels = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # 特殊 token (CLS, SEP, PAD) 标记为 -100
                token_labels.append(-100)
            else:
                # 获取该 word 的标签并转换为 id
                label = labels[word_idx]
                token_labels.append(LABELS.index(label))
        
        inputs['labels'] = token_labels
        
        # 删除 offsets_mapping，训练时不需要（部分 tokenizer/版本可能不返回该键）
        inputs.pop('offsets_mapping', None)
        
        return inputs

    dataset_dict = dataset_dict.map(encode, remove_columns=['text', 'label'])

    # 6. 保存到文件
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)


if __name__ == '__main__':
    process()

