import cv2
import numpy as np

def reduce_resolution(image):
    reduce_image = []
    temp_y = []
    for x in range(image.shape[0]):
        if x % 2 != 0:
            continue
        for y in range(image.shape[1]):
            if y % 2 == 0:
                temp_y.append(image[x][y])
        reduce_image.append(temp_y)
        temp_y = []
    reduce_image = np.array(reduce_image)
    return reduce_image

image = cv2.imread("images/landscape.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Displayed Image",gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

reduce1 = reduce_resolution(gray_image)
cv2.imshow("Reduce1",reduce1)

cv2.waitKey(0)
cv2.destroyAllWindows()

reduce2 = reduce_resolution(reduce1)
cv2.imshow("Reduce2",reduce2)

cv2.waitKey(0)
cv2.destroyAllWindows()

reduce3 = reduce_resolution(reduce2)
cv2.imshow("Reduce2",reduce3)

cv2.waitKey(0)
cv2.destroyAllWindows()