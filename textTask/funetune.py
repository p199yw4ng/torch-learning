import torch
import numpy as np
import datasets
import evaluate 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

## 加载预训练模型及tokenizer

# Tokenizer：分词器；用自己的数据funetune，AutoTokenizer是通用封装，根据载入预训练模型来自适应。
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# num_labels指定分类数量
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) 

# datasets 加载
dataset = datasets.load_dataset('glue','sst2')

metric = evaluate.load('glue','sst2')
print(metric)
# 处理数据集 encode_dataset = tokenizer(dataset)
def preprocess_function(item):
    # truncation 将大于长度的部分截断
    # tokenizer 返回的参数 input_ids:代表token在词典里的位置； token_type_ids: token在句子里的位置； attention_mask：是否参与attention计算，
    return tokenizer(item['sentence'],truncation=True)

encode_dataset = dataset.map(preprocess_function, batched=True)


# Traing优化
batch_size=4
args = TrainingArguments(
    "bert-base-uncased-finetuned-sst2", # output_dir
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 计算
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions,references=labels)

trainer  = Trainer(
    model,
    args,
    train_dataset=encode_dataset['train'],
    eval_dataset=encode_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    
)
trainer.train()
trainer.evaluate()