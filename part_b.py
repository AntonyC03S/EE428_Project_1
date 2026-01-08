# activate environment in powershell: .venv/Scripts/Activate.ps1   
import numpy as np
from matplotlib import pyplot as plt
import cv2
bact_img = cv2.imread("images/bacteria.bmp") # read the image into a numpy array
cv2.imshow("title", bact_img)
# cv2.waitKey(0)
cv2.destroyAllWindows()
gray_img = cv2.cvtColor(bact_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray_img)
cv2.waitKey(0)  
cv2.destroyAllWindows()
# cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])()
hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
plt.hist(gray_img.ravel(),256,[0,256])
plt.show()
# set threshold at 100
_, bin_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Binary", bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
out = cv2.connectedComponentsWithStats(bin_img, connectivity=4, ltype=cv2.CV_32S)
number_of_bacteria = out[0] - 1 # 0th connected component is the black area, so ignore it
centroids = out[-1]
stats = out[2]
nums = 30
label_char = 65
total_bact_area = 0
for i in range(1,len(centroids)): # 0th connected component is the black area, so ignore it
    row = int(centroids[i][0])
    col = int(centroids[i][1])
    cv2.putText(bact_img, chr(label_char), (row, col), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    print(f"Label: {chr(label_char)}, Area in pixels: {stats[i, cv2.CC_STAT_AREA]}")
    label_char += 1
    total_bact_area += int(stats[i, cv2.CC_STAT_AREA])
cv2.imshow("labelled", bact_img)
cv2.waitKey(0)

print(f"number of bacteria: {number_of_bacteria}")
print(f"pixel area occupied by bacteria: {int(np.sum(bin_img, axis=None)/255)}")
print(f"total number of pixels in image: {np.size(bin_img)}")