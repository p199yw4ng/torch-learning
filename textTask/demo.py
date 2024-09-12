import torch
from transformers import pipeline

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 创建一个文本分类pipeline
classifier = pipeline('sentiment-analysis', device=deivce)

# 使用pipeline分析一段文本
result = classifier("I really like the new Hugging Face Transformers library.")
print(result)