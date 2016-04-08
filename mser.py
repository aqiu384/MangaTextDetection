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

img = cv2.imread('./inputs/' + clean_images[4], cv2.IMREAD_GRAYSCALE)
vis = img.copy()
mser = cv2.MSER_create()

regions = mser.detectRegions(img, None)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(vis, hulls, 1, 255)

output_img = np.stack([vis, img, img], axis=2)

plt.imshow(output_img)
plt.show()

