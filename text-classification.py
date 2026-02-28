import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd


text = str(input("put a text: "))

                         
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer(text , return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

if predicted_class_id == 0:
    print("Warning")
else :
    print("Safe")