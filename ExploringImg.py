
import cv2 as cv
import matplotlib.pyplot as plt

reading = cv.imread('10.png', cv.IMREAD_GRAYSCALE)
cv.imwrite('Output.png', reading)
plt.imshow(reading, cmap='gray')
plt.show()