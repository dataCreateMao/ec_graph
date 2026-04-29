# NER 训练/评估/预测使用说明

本文档说明 `src/ner/train.py`、`src/ner/eval.py`、`src/ner/predict.py` 的使用方式与典型输出。

## 1. 环境准备

在项目根目录执行：

```bash
uv sync
```

若你使用已有虚拟环境，也可直接用：

```bash
/Users/linlong/code/nlp/ec_graph/.venv/bin/python ...
```

## 2. 训练（train）

### 命令

```bash
python src/ner/train.py
```

### 典型输出（节选）

```text
[INFO] Training device: mps
[INFO] fp16 enabled: False
[WARN] Torch-NumPy bridge unavailable, fallback mode enabled: disable eval/metrics/early-stopping.
...
100%|██████████| 500/500 [28:54<00:00,  3.47s/it]
{'train_runtime': 1734.6682, 'train_samples_per_second': 2.306, 'train_steps_per_second': 0.288, 'train_loss': 0.2990938522815704, 'epoch': 5.0}
```

### 训练产物

默认保存到：

```text
checkpoints/ner/best_model
```

常见文件：

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `training_args.bin`

## 3. 评估（eval）

### 命令

默认评估测试集：

```bash
python src/ner/eval.py
```

指定模型目录与测试集目录：

```bash
python src/ner/eval.py \
  --model-dir checkpoints/ner/best_model \
  --test-dir data/ner/processed/test
```

### 输出

成功时会输出 JSON 指标，例如：

```json
{
  "eval_loss": 0.1234,
  "eval_overall_precision": 0.91,
  "eval_overall_recall": 0.89,
  "eval_overall_f1": 0.90,
  "eval_overall_accuracy": 0.97
}
```

### 注意事项

当前环境若存在 `torch` 与 `numpy` 不兼容（如 `torch 2.2.x + numpy 2.x`），`eval.py` 会直接报错并提示环境不兼容，这是保护行为，避免中途崩溃。

## 4. 预测（predict）

### 命令

默认示例文本：

```bash
python src/ner/predict.py
```

指定输入文本并输出 JSON：

```bash
python src/ner/predict.py \
  --text "苹果手机壳超薄磨砂保护套" \
  --json
```

指定模型目录：

```bash
python src/ner/predict.py \
  --model-dir checkpoints/ner/best_model \
  --text "麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩" \
  --json
```

### 典型输出（JSON）

```json
{
  "text": "麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩",
  "labels": ["O", "O", "O", "B", "I", "..."],
  "entities": ["德国进口", "护眼", "营养软胶囊"]
}
```

## 5. 常见问题

- 出现 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`：
  - 这是当前 `torch` 与 `numpy` 兼容性问题；`train.py` 已支持降级训练模式。
- `eval.py` 直接报 `Torch-NumPy bridge is unavailable`：
  - 说明当前环境无法执行评估路径，需切换到兼容的 `torch/numpy` 组合后再评估。
