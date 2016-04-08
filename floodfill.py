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

img = cv2.imread('./inputs/' + clean_images[0], cv2.IMREAD_GRAYSCALE)
wgauss = cv2.GaussianBlur(img, (WSIG*6+1, WSIG*6+1), WSIG)
bgauss = cv2.GaussianBlur(img, (BSIG*6+1, BSIG*6+1), BSIG)

output = np.copy(img)

WHITE = 240 # 240 # 170
BLACK = 70 # 70 # 70

threshed = np.copy(output)
threshed[wgauss >= WHITE] = 255
threshed[wgauss < WHITE] = 0

FLOOD_POINTS = [
    213, 1825,
    340, 1542,
    935, 1461,
    1195, 1386,
    242, 866,
    577, 941,
    860, 860,
    438, 225,
    675, 271,
    1160, 294
]

mask = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)

for x, y in zip(FLOOD_POINTS[::2], FLOOD_POINTS[1::2]):
    cv2.floodFill(np.copy(threshed), mask, (x, y), 255)

mask = np.bitwise_not(mask)
mask2 = np.zeros((img.shape[0]+4, img.shape[1]+4), np.uint8)

cv2.floodFill(np.copy(mask), mask2, (1, 1), 255)
mask2 = mask2[2:-2, 2:-2]
mask2 = np.bitwise_not(mask2)

output_image = np.copy(mask2)
mask3 = np.copy(mask2)

_, contours, hierarchy = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(output_image, contours, -1, 127, 3)

for contour in contours:
    print('contour')
    x, y, w, h = cv2.boundingRect(contour)
    if False:
        cv2.rectangle(output_image, (x, y), (x+w, y+h), 127, 2)

plt.subplot(121)
plt.imshow(img, 'gray')
plt.subplot(122)
plt.imshow(output_image, 'gray')
plt.show()
