from google.colab import files
import numpy as np 
import cv2

arquivo = files.upload()
filename = list(arquivo.keys())[0]

img = cv2.imread(filename)

img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img , cv2.COLOR_BGR2LAB)

h = img_hsv[:,:,0]

#INTER_NEAREST
#INTER_LINEAR

fhd = (1920,1080)

img_fhd  = cv2.resize(
    img_rgb,
    fhd,
    interpolation = cv2.INTER_CUBIC

)

cubic  = cv2.resize(
    img_rgb,
    fhd,
    interpolation = cv2.INTER_CUBIC
)

nearest  = cv2.resize(
    img_rgb,
    fhd,
    interpolation = cv2.INTER_NEAREST
)

h , w = img_rgb.shape[:2]

tx = 100
ty = 150

M = np.float32([
    [1,0,tx],
    [0,1,ty]
])
