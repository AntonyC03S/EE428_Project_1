import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the images
img1 = cv2.imread('images/landscape.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('images/group_photo.jpg', cv2.IMREAD_COLOR)

cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)

# Convert the image to grayscale image 
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale 1', gray_img1)
cv2.imshow('Grayscale 2', gray_img2)
cv2.waitKey(0)  
cv2.destroyAllWindows()

# Maximum and Minimum Intensity Values 
min_val1 = np.min(gray_img1)
max_val1 = np.max(gray_img1)
min_loc1 = np.unravel_index(np.argmin(gray_img1), gray_img1.shape)
max_loc1 = np.unravel_index(np.argmax(gray_img1), gray_img1.shape)

min_val2 = np.min(gray_img2)
max_val2 = np.max(gray_img2)
min_loc2 = np.unravel_index(np.argmin(gray_img2), gray_img2.shape)
max_loc2 = np.unravel_index(np.argmax(gray_img2), gray_img2.shape)

print(f"Image1 -> Min: {min_val1} at {min_loc1}, Max: {max_val1} at {max_loc1}")
print(f"Image2 -> Min: {min_val2} at {min_loc2}, Max: {max_val2} at {max_loc2}")


# Find File Size
size1 = os.path.getsize('images/landscape.png')
size2 = os.path.getsize('images/group_photo.jpg')

print(f"File Size - Landscape: {size1} bytes")
print(f"File Size - Group Photo: {size2} bytes")
