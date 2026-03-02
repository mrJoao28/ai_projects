from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import cv2
import os
def main():
    print("oi")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("The camera dosent open")
        exit()
        return 1
    
    print("aqui")
    
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

    cv2.imshow()


    while True:
        ret , frame = cap.read()

        frame = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2BGR)

        if not ret:
            return 1
        
        inputs = image_processor(frame, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax(-1).item()

        print(model.config.id2label[predicted_label])
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__" :
    main()

