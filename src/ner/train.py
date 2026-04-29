import time
import sys
from pathlib import Path

from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification, EvalPrediction, EarlyStoppingCallback
import evaluate

# 允许直接以脚本方式运行：自动将 src 加入导入路径
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import *

#1. 分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 标签分类
id2label = {id:label for id,label in enumerate(LABELS)}
label2id = {label:id for id,label in enumerate(LABELS)}

#2. 模型
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

#3. 加载数据集（预处理阶段使用 save_to_disk 持久化）
train_dataset = load_from_disk(str(PROCESSED_DATA_DIR / 'train'))
valid_dataset = load_from_disk(str(PROCESSED_DATA_DIR / 'valid'))
# test_dataset = load_dataset(PROCESSED_DATA_DIR/'test')

#4. 数据整理集
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors='pt',
)

# 训练参数
use_fp16 = torch.cuda.is_available()
numpy_available_in_torch = True
try:
    _ = torch.zeros(1).numpy()
except Exception:
    numpy_available_in_torch = False

if torch.cuda.is_available():
    device_name = f"cuda:{torch.cuda.current_device()}"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"

print(f"[INFO] Training device: {device_name}")
print(f"[INFO] fp16 enabled: {use_fp16}")
if not numpy_available_in_torch:
    print("[WARN] Torch-NumPy bridge unavailable, fallback mode enabled: disable eval/metrics/early-stopping.")

training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR / NER_DIR),
    logging_dir=str(LOG_DIR / NER_DIR / time.strftime('%Y-%m-%d_%H-%M-%S')),
    num_train_epochs= EPOCHS,       #训练总轮数
    per_device_train_batch_size=BATCH_SIZE, #批大小

    save_strategy='steps',          #保存策略
    save_steps=SAVE_STEPS,          #每20次迭代进行一次保存
    save_total_limit=3,             #最多保存3个检查点

    fp16=use_fp16,              # 仅在 CUDA 环境开启混合精度，避免 MPS+旧版 torch 报错

    logging_strategy='steps',
    logging_steps=SAVE_STEPS,

    eval_strategy='steps' if numpy_available_in_torch else 'no',
    eval_steps=SAVE_STEPS if numpy_available_in_torch else None,

    metric_for_best_model='eval_overall_f1' if numpy_available_in_torch else None,     #模型评估指标
    greater_is_better=True if numpy_available_in_torch else None,
    load_best_model_at_end=numpy_available_in_torch,    #训练结束加载最佳模型
)

#6. 评估指标函数
seqeval = evaluate.load('seqeval') if numpy_available_in_torch else None
def compute_metrics(p: EvalPrediction):
    if seqeval is None:
        return {}
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
        #转BIO标签
        unpad_labels.append([id2label[id] for id in unpad_pred])
        unpad_preds.append([id2label[id] for id in unpad_label])
    return seqeval.compute(predictions=unpad_preds, references=unpad_labels)

#7. 早停回调
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=20)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset if numpy_available_in_torch else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if numpy_available_in_torch else None,
    callbacks=[early_stopping_callback] if numpy_available_in_torch else []
)

# 训练
trainer.train()

# 保存模型
trainer.save_model(CHECKPOINT_DIR / NER_DIR / 'best_model')