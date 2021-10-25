import numpy as np
import numpy.fft as fft
import cv2 as cv
import matplotlib.pyplot as plt
from EstimatingAutocorr import *
# import scipy.ndimage as nd
# import scipy.misc as misc


from math import pi


#Read in source image
# source = cv.imread("EncryptedImage_fft.png", cv.IMREAD_GRAYSCALE)
auto = object_auto_correlation('EncryptedImage.png')

#Pad image to simulate oversampling
pad_len = 0
padded = np.pad(auto, ((pad_len, pad_len), (pad_len, pad_len)), 
                'constant', constant_values=((0,0),(0,0)))

plt.imshow(padded, cmap ='gray')
plt.show()

# padded = padded[50:200, 100:300]
#print(source.padded)

# ft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(padded))))
ft = np.fft.fftshift(np.fft.fft2(padded))

#simulate diffraction pattern
diffract = np.abs(ft)

# diffract = padded

# cv.normalize(diffract, diffract, 0, 1, cv.NORM_MINMAX)

plt.imshow(np.log(diffract), cmap='gray')
plt.show()

l = len(padded)
print(padded.shape)

#keep track of where the image is vs the padding
mask = np.ones((auto.shape[0], auto.shape[1]))
mask = np.pad(mask, ((pad_len, pad_len),(pad_len, pad_len)), 'constant', 
                constant_values=((0,0),(0,0)))

print(mask.shape)
print(mask.shape == padded.shape) 

#Initial guess using random phase info
guess = diffract * np.exp(1j * np.random.rand(mask.shape[0], mask.shape[1]) * 2 * pi)

#number of iterations
r = 801

#step size parameter
beta = 0.3

#previous result
prev = None
for s in range(0,r):
    #apply fourier domain constraints
    update = diffract * np.exp(1j * np.angle(guess)) 
    
    inv = np.fft.ifft2(update)
    inv = np.real(inv)
    if prev is None:
        prev = inv
        
    #apply real-space constraints
    temp = inv
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            #image region must be positive
            if inv[i,j] < 0 and mask[i,j] == 1:
                inv[i,j] = prev[i,j] - beta*inv[i,j]
            #push support region intensity toward zero
            if mask[i,j] == 0:
                inv[i,j] = prev[i,j] - beta*inv[i,j]
    
    
    prev = temp
    
    #prev_1 = cv.normalize(prev, prev, 0, 1, cv.NORM_MINMAX)
    
    guess = np.fft.fft2(inv)
        
    #save an image of the progress
    if s % 50 == 0:
        cv.imwrite(str(s) +
                    ".png", prev)
        print(s)

