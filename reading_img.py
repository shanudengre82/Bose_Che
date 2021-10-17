# Importing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from definingfft import fft_online

# reading image
img = cv.imread('testimg_2.png', cv.IMREAD_GRAYSCALE)
cv.imshow('testingimage', img)

print(img.shape)
# img_0 = img[:, :, 0]
# img_1 = img[:, :, 1]
# img_2 = img[:, :, 2]

# img_square_root = np.zeros(img.shape, dtype='uint8')
# for i in range(0, img_square_root.shape[0]):
#     for j in range(0, img_square_root.shape[1]):
        
#         # if img[i][j] < 150:
#         img_square_root[i][j] = np.sqrt(img[i][j])
#         # else: 
#         #     pass

# cv.imwrite('Square_root.png', img_copy)

img_square = cv.imread('Square.png', cv.IMREAD_GRAYSCALE)
cv.imshow('Square', img_square)

img_square_root = cv.imread('Square_root.png', cv.IMREAD_GRAYSCALE)
cv.imshow('Square root', img_square_root)

# print(img_0.all() == img_1.all() == img_2.all())
# print(img.shape)
# Converting to greyscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Encryptedimage gray', gray)

# getting edges
# for i in range(0, 10):
#     canny = cv.Canny(img, 0.1+0.2*i, 0.2+10*i)
#     cv.imshow(f'Edges_{i}', canny)


"""
Threshold method. Threshold limits basically make the imgae binary.
"""
# for i in range(4, 7):
#     ret, thresh = cv.threshold(img, thresh = 1+i, maxval= 255, type=cv.THRESH_BINARY)
#     cv.imshow(f'Thresh_{i}', thresh)

# finding contours
# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours))

img_1 = cv.Canny(img, 30, 100)
cv.imshow(f'Edges_30_100', img_1)

img_2 = cv.Canny(img, 1, 30)
cv.imshow(f'Edges_1_30', img_2)

img_3 = cv.Canny(img, 100, 200)
cv.imshow(f'Edges_100_200', img_3)

# j = 0
# i = 32
# for j in range(0, 1):
#     img_crop = img[(int(img.shape[0]/2) - 200*j - i): (int(img.shape[0]/2) - 200*j+ i), (int(img.shape[1]/2) - 200*j - i): (int(img.shape[1]/2) - 200*j + i)]
#     f = np.fft.ifft2(img_crop)
#     fshift = np.fft.ifftshift(f)
#     magnitude_spectrum_withoutshift = 20*np.log(np.abs(f))
#     magnitude_spectrum = 20*np.log(np.abs(fshift))

#     plt.subplot(121),plt.imshow(magnitude_spectrum_withoutshift, cmap = 'gray')
#     plt.title('magnitude_spectrum_withoutshift'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
#     # plt.imshow(magnitude_spectrum, cmap='gray')
#     plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#     plt.show()


fft_online(argv=['EncryptedImage.png'])

fft_online(argv=['testimg_2.png'])

# fourior transform original image
f = np.fft.ifft2(img)
fshift_inv = np.fft.ifftshift(f)
magnitude_spectrum_withoutshift = (20*np.log(np.abs(f)))
magnitude_spectrum = (20*np.log(np.abs(fshift_inv)))

plt.subplot(121),plt.imshow(magnitude_spectrum_withoutshift, cmap = 'gray')
plt.title('magnitude_spectrum_withoutshift'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# fourior transform
f = np.fft.fft2((magnitude_spectrum_withoutshift))
cv.imshow('f', np.sqrt(np.abs(f)))
fshift = np.fft.ifftshift(f)
magnitude_spectrum_withoutshift = (20*np.log(np.abs(f)))
magnitude_spectrum = (20*np.log(np.abs(fshift)))

plt.subplot(121),plt.imshow(magnitude_spectrum_withoutshift**2, cmap = 'gray')
plt.title('magnitude_spectrum_withoutshift'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum**2, cmap = 'gray')
# plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# fourior transform

# ret, thresh = cv.threshold(magnitude_spectrum, thresh = 0, maxval= 200, type=cv.THRESH_BINARY)
# cv.imshow(f'Thresh', thresh)
      
# Waittime/delay for a key to be pressed
# 0 is infinite amount of time (time is in ms)
cv.waitKey(0)
cv.destroyAllWindows()