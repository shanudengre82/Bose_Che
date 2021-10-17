
# Importing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# reading image
img = cv.imread('EncryptedImage.png', cv.IMREAD_GRAYSCALE)
cv.imshow('testingimage', img)

print(img.shape)

img_cov = np.cov(img, img)
# cv.normalize(img_cov, img_cov, 0, 1, cv.NORM_MINMAX)

cv.imshow('Cov', img_cov)

plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('magnitude_spectrum_withoutshift')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(img_cov, cmap = 'gray')
# plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()