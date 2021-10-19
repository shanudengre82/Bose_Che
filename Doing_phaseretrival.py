
# Importing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from definingfft import fft_online
from fineupphaseretirval import fienup_phase_retrieval

np.random.seed(1)
image = cv.imread('EncryptedImage.png', cv.IMREAD_GRAYSCALE)
magnitudes = np.abs(np.fft.fft2(image))
result = fienup_phase_retrieval(magnitudes, beta=0.2, 
                           steps=200, mode='hybrid', verbose=True)

plt.show()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(122)
plt.imshow(result, cmap='gray');
plt.title('Reconstruction')
plt.show()

np.random.seed(1)
# image = cv.imread('EncryptedImage.png', cv.IMREAD_GRAYSCALE)
magnitudes = np.abs(np.fft.fft2(result))
result = fienup_phase_retrieval(magnitudes, steps=100,
                                verbose=False)

plt.show()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(122)
plt.imshow(result, cmap='gray');
plt.title('Reconstruction')
plt.show()

np.random.seed(1)
# image = cv.imread('EncryptedImage.png', cv.IMREAD_GRAYSCALE)
magnitudes = np.abs(np.fft.fft2(result))
result = fienup_phase_retrieval(magnitudes, steps=100,
                                verbose=False)

plt.show()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(122)
plt.imshow(result, cmap='gray');
plt.title('Reconstruction')
plt.show()