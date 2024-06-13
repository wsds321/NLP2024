import re
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch

novel_dir = './cn_words/'
punctuation_path = "./DLNLP2023-main/cn_punctuation.txt"
stopwords_path = "./DLNLP2023-main/cn_stopwords.txt"

data = []
# 读取并处理文本数据
for filename in os.listdir(novel_dir):
    with open(novel_dir+filename, "r", encoding="ANSI") as f:
        text = f.read()
        # 简单的文本清理
        text = re.sub(r'\s+', ' ', text)
        paragraphs = text.split("。")
        for i in range(len(paragraphs) - 1):
            prompt = paragraphs[i] + "。"
            continuation = paragraphs[i + 1]
            data.append({"prompt": prompt, "continuation": continuation})


# 转换为DataFrame
df = pd.DataFrame(data)
train_df, val_df = train_test_split(df, test_size=0.1)

train_df.to_csv("./data/train.csv", index=False)
val_df.to_csv("./data/val.csv", index=False)

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# 加载数据集
train_dataset = load_dataset('./data/', data_files='train.csv')['train']
val_dataset = load_dataset('./data/', data_files='val.csv')['train']

# 加载T5模型和分词器
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 数据预处理函数
def preprocess_function(examples):
    inputs = [ex for ex in examples['prompt']]
    targets = [ex for ex in examples['continuation']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 应用预处理
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 训练模型
trainer.train()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到合适的设备
model.to(device)
model.eval()
def generate_text(prompt, model, tokenizer, max_length=200):
    # 将输入张量移动到与模型相同的设备
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "少年下山游历，"
generated_text = generate_text(prompt, model, tokenizer)
print(generated_text)