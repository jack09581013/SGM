from algorithm.lib import *

img = cv2.imread('data/taipei/true orthophoto/mean.tif')

img = img.astype('float64')
black = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)

for i in range(1, img.shape[0]-1):
    print(i)
    for j in range(1, img.shape[1]-1):
        if black[i, j]:
            img[i, j, :] = (img[i-1, j, :] + img[i+1, j, :] + img[i, j-1, :] + img[i, j+1, :])/4

img = img.astype('uint8')
cv2.imwrite('data/taipei/true orthophoto/mean_intepolation.tif', img)