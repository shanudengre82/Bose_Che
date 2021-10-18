# Importing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from scipy import signal
from scipy import misc

# reading image
img = cv.imread('Encryptedimage.png', cv.IMREAD_GRAYSCALE)
# cv.imshow('Encryptedimage', img)
print(np.mean(img))

img = img - img.mean()
print(np.mean(img))

print('1')
img_auto = signal.correlate2d(img, img, boundary='fill', mode='same')

print('2')
cv.normalize(img_auto, img_auto, 0, 255, cv.NORM_MINMAX)
cv.imwrite('EncryptedImage_auto.png', img_auto)

plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('magnitude_spectrum_withoutshift'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_auto, cmap = 'gray')
# plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()