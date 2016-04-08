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
    'doraemon_raw.jpg',
    'bleach.jpg',
    'gyaku.jpg'
]

WSIG = 4 # 4 # 1
BSIG = 4 # 4 # 3

img = cv2.imread('./inputs/' + clean_images[7], cv2.IMREAD_GRAYSCALE)
wgauss = cv2.GaussianBlur(img, (WSIG*6+1, WSIG*6+1), WSIG)
bgauss = cv2.GaussianBlur(img, (BSIG*6+1, BSIG*6+1), BSIG)

output = np.copy(img)

WHITE = 240 # 240 # 170
BLACK = 70 # 70 # 70

output[output > WHITE] = WHITE
output[output < BLACK] = BLACK

output2 = np.copy(output)

output2[wgauss > WHITE] = WHITE
output2[bgauss < BLACK] = BLACK

threshed = np.copy(output)
threshed[wgauss >= WHITE] = 255
threshed[wgauss < WHITE] = 0

thresh, im2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
im3, contours, hierarchy = cv2.findContours(np.copy(im2), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_img = np.copy(img)

MIN_SIZE = 15 # 10
MAX_SIZE = 80 # 50 # 100
MIN_ECCENTRICITY = 0.1
MAX_ECCENTRICITY = 1 / MIN_ECCENTRICITY
MIN_AREA = 100

output_img = np.stack([img, img, img], axis=2)

# cv2.drawContours(output_img, contours, -1, (255, 0, 0), 3)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if MIN_SIZE < h < MAX_SIZE and MIN_SIZE < w < MAX_SIZE and MIN_ECCENTRICITY < w / h < MAX_ECCENTRICITY:
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

plt.subplot(121)
plt.imshow(im2, 'gray')
plt.subplot(122)
plt.imshow(output_img, 'gray')
# plt.hist(img.ravel(), 256, [0, 256])
plt.show()
