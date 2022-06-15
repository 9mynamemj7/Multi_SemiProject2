import cv2, os, warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
img = plt.imread('다운로드.png')
img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR)
plt.imshow(img)