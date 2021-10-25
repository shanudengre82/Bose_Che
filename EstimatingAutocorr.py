"""
In this module we are going to wirte algoritm to estimate object autocorrelation function from a given speckle pattern.
"""

"""For a large image first we would like to divide it into smaller parts"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure



def get_img_size(img: str):
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    height = img.shape[0]
    width = img.shape[1]
    return height, width


def plot_img(img: str, width = 16, height = 12):
    """
    Plots the image with the given size
    """
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    figure(figsize=(width, height))
    plt.imshow(img, cmap='gray')
    plt.show()
    
    
def getting_smaller_images(img: str, size=1024):
    """
    This function will devide the image in a square shape of given size
    """
    
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    # print(img.shape)
    
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    dict_to_return = {}
    
    
    img_height_crop_number = int(img_height//size) 
    img_width_crop_number = int(img_width//size) 
    
    # print(img_width_crop_number*img_height_crop_number)
    
    count = 0
    
    for i in range(img_height_crop_number):
        for j in range(img_width_crop_number):
            dict_to_return[f'{count}'] = img[i*size :i*size+size, j*size:j*size+size]
            count+=1
            
    return dict_to_return

    
def object_auto_correlation(img):
    """
    As defined in ref 
    Title: Deep-inverse correlography: towards real-time
           high-resolution non-line-of-sight imaging
    """
    smaller_images=getting_smaller_images(img)
    
    print(len(smaller_images))
    
    sum_1 = np.zeros(smaller_images['0'].shape, dtype=complex)
    sum_2 = np.zeros(smaller_images['0'].shape, dtype=complex)
    # print(sum_1.shape)
    
    for i in smaller_images:
        # print(i)
        # print(smaller_images[i], smaller_images[i].shape)
        
        sum_1 += (np.abs(np.fft.ifft2(smaller_images[i])))**2
        sum_2 += (np.fft.ifft2(smaller_images[i]))
        
    # for i in smaller_images:
        
    return np.log(np.real(sum_1/len(smaller_images) - (np.abs(sum_2)/len(smaller_images))**2))
        
    

if __name__ == '__main__':
    
    image_size = get_img_size('EncryptedImage.png')
    
    print(image_size, type(image_size))
    
    plot_img('EncryptedImage.png')
    
    smaller_image = getting_smaller_images('EncryptedImage.png')
    
    print(len(smaller_image))
    
    auto = object_auto_correlation('EncryptedImage.png')
    
    plt.imshow(auto, cmap='gray')
    plt.show()