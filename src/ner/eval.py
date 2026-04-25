import sys

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification, AutoTokenizer, \
    AutoModelForTokenClassification, EvalPrediction
import evaluate

from pathlib import Path

# 允许直接以脚本方式运行：自动将 src 加入导入路径
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import *

#1. 分词器
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR / NER_DIR / 'best_model')

#2. 模型
model = AutoModelForTokenClassification.from_pretrained(
    CHECKPOINT_DIR / NER_DIR / 'best_model',
)

#3. 加载数据集
test_dataset = load_dataset(PROCESSED_DATA_DIR/'test')

#4. 数据整理集
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors='pt',
)

#5. 评估指标函数
seqeval = evaluate.load('seqeval')
def compute_metrics(p: EvalPrediction):
    # 提取模型的预测输出和真实标签
    logits = p.predictions
    preds = logits.argmax(axis=-1)      #预测分类标签
    labels = p.label_ids                #真实分类标签
    # 将标签id转换为真正的标注标签BIO
    unpad_labels = []
    unpad_preds = []
    for label, pred in zip(labels, preds):
        #去掉填充对应的id
        unpad_label = label[label != -100]
        unpad_pred = pred[ label != -100]
        #转BIO标签,添加到列表
        unpad_labels.append([model.config.id2label[id] for id in unpad_pred])
        unpad_preds.append([model.config.id2label[id] for id in unpad_label])
    return seqeval.compute(predictions=unpad_preds, references=unpad_labels)


# 6. 创建训练器
trainer = Trainer(
    model=model,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. 验证评估
evaluate = trainer.evaluate()

print(evaluate)
