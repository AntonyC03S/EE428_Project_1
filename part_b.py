# activate environment in powershell: .venv/Scripts/Activate.ps1   
import numpy as np
from matplotlib import pyplot as plt
import cv2
print("hi!")
bact_img = cv2.imread("images/bacteria.bmp") # read the image into a numpy array
print(np.shape(bact_img))
cv2.imshow("title", bact_img)
# cv2.waitKey(0)
cv2.destroyAllWindows()
gray_img = cv2.cvtColor(bact_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray_img)
# cv2.waitKey(0)  
cv2.destroyAllWindows
# cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])()
hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
print(np.shape(hist))