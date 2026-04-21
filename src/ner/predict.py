import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
from configuration.config import *

# 自定义预测期类
class Predictor:
    # 初始化
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    # 预测方法
    def predict(self, inputs: str | list[str]):
        # 如果是1条数据也转成列表处理
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        # 1. 预分词，得到字符列表
        tokens_list = [list(input) for input in inputs]
        # 2. 用分词器id化处理
        inputs_tensor = self.tokenizer(
            tokens_list,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        # 3. 加载到设备
        inputs_tensor = {key: value.to(self.device) for key, value in inputs_tensor.items()}

        # 4. 前向传播，推理预测
        with torch.no_grad():
            outputs = self.model(**inputs_tensor)
            logits = outputs.logits
            predictions = logits.argmax(axis=-1).tolist()

        # 5. 将id列表转换为BIO标签
        final_predictions = []
        for tokens, prediction in zip(tokens_list, predictions):
            # 截取预测输出中，真实长度
            prediction = prediction[1:len(tokens)+1]
            #转换成标签
            final_prediction = [self.model.config.id2label[id] for id in prediction]
            final_predictions.append(final_prediction)
        if is_str:
            return final_predictions[0]
        return final_predictions

def predict():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model'))
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model'))
    # 定义预测器
    predictor = Predictor(model, tokenizer, device)

    # 定义数据
    text = "麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩"

    # 预测
    result = predictor.predict(text)
    for token, label in zip(text, result):
        print(f'{token:<10} {label}')

if __name__ == '__main__':
    predict()
