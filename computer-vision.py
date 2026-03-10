from google.colab import files
import matplotlib.pyplot as plt 
import numpy as np 
import cv2

arquivo = files.upload()
filename = list(arquivo.keys())[0]

img = cv2.imread(filename)

img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img , cv2.COLOR_BGR2LAB)
