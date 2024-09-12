
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

input=tokenizer('i love you',return_tensors='pt')

model.to(device)
result = model(input)
print(result)


