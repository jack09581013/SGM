import matplotlib.pyplot as plt
from algorithm.lib import *

# img1 = cv2.imread('data/taipei/180808-4761.tif', cv2.IMREAD_GRAYSCALE).astype('float')
# img2 = cv2.imread('data/taipei/180808-4764.tif', cv2.IMREAD_GRAYSCALE).astype('float')

img1 = cv2.imread('data/taipei/rectify/4747_P4747_4748_rectify.tif', cv2.IMREAD_GRAYSCALE).astype('float')
img2 = cv2.imread('data/taipei/rectify/4748_P4747_4748_rectify.tif', cv2.IMREAD_GRAYSCALE).astype('float')

img_merge = np.zeros(img1.shape)
img_merge = (5*img1 + 5*img2)/10

# cv2.imwrite('data/taipei/rectify/4748_P4747_4748_merge.tif', img_merge)

plt.imshow(img_merge.astype('uint8'), cmap='gray')
plt.show()

