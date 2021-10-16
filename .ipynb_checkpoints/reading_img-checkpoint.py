# Importing
import cv2 as cv
import numpy as np

# reading image
img = cv.imread('Encryptedimage.png')
cv.imshow('Encryptedimage', img)

img_0 = img[:, :, 0]
img_1 = img[:, :, 1]
img_2 = img[:, :, 2]


print(img_0.all() == img_1.all() == img_2.all())
print(img.shape)
# Converting to greyscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Encryptedimage gray', gray)

# getting edges
for i in range(0, 3):
    canny = cv.Canny(img, 0, 0.1)
    cv.imshow(f'Edges_{i}', canny)

# finding contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))
      
# Waittime/delay for a key to be pressed
# 0 is infinite amount of time (time is in ms)
cv.waitKey(0)
cv.destroyAllWindows()