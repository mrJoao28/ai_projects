import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytesseract 
import cv2
import pyautogui
import os


model_name1 = 't5-small'
model1 = T5ForConditionalGeneration.from_pretrained(model_name1, device_map="auto").eval()
tokenizer1 = T5Tokenizer.from_pretrained(model_name1)

tokenizer2 = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model2 = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").eval()
caminho = "C:\Program Files\Tesseract-OCR"

while True:
    img = pyautogui.screenshot()
    img.save("imagem.png")
    image = cv2.imread("imagem.png")
    pytesseract.pytesseract.tesseract_cmd = caminho + r"\tesseract.exe"
    text = pytesseract.image_to_string(image)
    text = f"translate Portuguese to English: {text}"
    input_ids = tokenizer1(text, return_tensors="pt").input_ids.to(model1.device)
    outputs = model1.generate(input_ids=input_ids)
    translate_text =  tokenizer1.decode(outputs[0], skip_special_tokens=True)
    
    inputs = tokenizer2(translate_text , return_tensors="pt")
    with torch.no_grad():
        logits = model2(**inputs).logits
    predicted_class_id = logits.argmax().item()
    model2.config.id2label[predicted_class_id]

    if predicted_class_id == 0:
        print("Warning")
    else :
        print("Safe")

    os.remove("imagem.png")


