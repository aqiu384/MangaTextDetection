import cv2
import numpy as np
from matplotlib import pyplot as plt

clean_images = [
    'Cleaning_Test_s01.png',
    'Cleaning_Test_s02.png',
    'Cleaning_Test_s03.png',
    'Cleaning_Test_m01.jpg',
    'Cleaning_Test_m02.jpg',
    'Cleaning_Test_m03.jpg',
    'doraemon_raw.jpg'
]

SIGMA = 1

img = cv2.imread('./inputs/' + clean_images[4], cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (SIGMA*6 + 1, SIGMA*6 + 1), SIGMA)
thresh, im2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
im3, contours, hierarchy = cv2.findContours(np.copy(im2), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_img = np.copy(img) # cv2.drawContours(np.copy(img), contours, -1, (0, 255, 0), 3)

MIN_SIZE = 10
MAX_SIZE = 100
MIN_ECCENTRICITY = 0.1
MAX_ECCENTRICITY = 1 / MIN_ECCENTRICITY

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if MIN_SIZE < w < MAX_SIZE and MIN_SIZE < h < MAX_SIZE and MIN_ECCENTRICITY < w / h < MAX_ECCENTRICITY:
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), 255, 2)

output_img = np.stack([contour_img, img, img], axis=2)

print(output_img.shape)

# BLACK = 0.2
# WHITE = 0.7
#
# img[img > WHITE] = WHITE

# img[img < BLACK] = BLACK
# img[mask] = BLACK

# img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
# img = cv2.resize(img, orig_size)

plt.subplot(121)
plt.imshow(img, cmap='Greys_r')
plt.subplot(122)
plt.imshow(output_img)
plt.show()
