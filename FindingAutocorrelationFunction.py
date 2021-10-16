# Importing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

# reading image
img = cv.imread('Encryptedimage.png', cv.IMREAD_GRAYSCALE)
cv.imshow('Encryptedimage', img)

print(np.mean(img))

# Autocorrelated img
# img_auto = np.zeros(img.shape, dtype='uint8')
# for i in range(0, img_auto.shape[0]):
#     for j in range(0, img_auto.shape[1]):
#         img_auto[i][j] = (img[i][j]-10.65135)
        

# f = np.fft.fft2(img_auto)
# img_auto = (np.log(np.abs(f)))**2 

# cv.imwrite('Autocorrelated_averagesub_fft_square.png', img_auto)

img_auto = cv.imread('Autocorrelated_averagesub_fft_square.png', cv.IMREAD_GRAYSCALE)   
plt.imshow(img_auto, cmap='gray')
plt.title('Average subtracted and fft and square')    
plt.show()

# cv.imwrite('Autocorrelated.png', img_auto)

# fourior transform autocorrelated image image
f = np.fft.fft2(img_auto)
fshift_inv = np.fft.fftshift(f)
magnitude_spectrum_withoutshift = 20*np.log(np.abs(f))
magnitude_spectrum = 20*np.log(np.abs(fshift_inv))

plt.subplot(121)
plt.imshow(np.abs(f), cmap = 'gray')
plt.title('magnitude_spectrum_withoutshift'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(np.abs(fshift_inv), cmap = 'gray')
# plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


cv.waitKey(0)
cv.destroyAllWindows()