# Importing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from numpy.core.fromnumeric import shape

# from definingfft import fft_online

# reading image
img = cv.imread('EncryptedImage.png', cv.IMREAD_GRAYSCALE)
cv.imshow('EncryptedImg', img)
print(img.shape)

# Basic exploration from edges
img_1 = cv.Canny(img, 30, 100)
cv.imshow(f'Edges_30_100', img_1)

img_2 = cv.Canny(img, 1, 30)
cv.imshow(f'Edges_1_30', img_2)

img_3 = cv.Canny(img, 100, 200)
cv.imshow(f'Edges_100_200', img_3)

# Plotting intensity patterns horizontally
figure(figsize=(16, 12))
for i in range(0, img.shape[0]):
    plt.plot(img[i][:])
plt.show()


## Understanding distribution of the intensity
sum_vertical = np.sum(img, axis = 0)
sum_vertical = sum_vertical/np.max(sum_vertical)
sum_horizontal = np.sum(img, axis = 1)
sum_horizontal = sum_horizontal/np.max(sum_horizontal)
figure(figsize=(16, 12))
plt.plot(sum_vertical)
plt.plot(sum_horizontal)
plt.title('Horizontal and vertical intensities distributions')
plt.show()


# Fourior transform
f = np.fft.fft2((img))
cv.imshow('f', np.sqrt(np.abs(f)))
fshift = np.fft.ifftshift(f)
magnitude_spectrum_withoutshift = (20*np.log(np.abs(f)))
magnitude_spectrum = (20*np.log(np.abs(fshift)))

figure(figsize=(16, 12))
plt.subplot(121),plt.imshow(magnitude_spectrum_withoutshift**2, cmap = 'gray')
plt.title('magnitude_spectrum_withoutshift'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum**2, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
