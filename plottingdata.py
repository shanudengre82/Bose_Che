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