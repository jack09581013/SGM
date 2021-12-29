from algorithm.lib import *

distCoeffs = np.array([-0.04591155158913425, 0.035591475053250415, -1.3125789551377542e-006, -3.3011941430374701e-006, -0.013634871666012505]).reshape(5, 1)
cameraMatrix = np.array([10912.905472192366, 0, 5800.8672228100659, 0, 10912.905472192366, 4348.9916944617316, 0, 0, 1]).reshape(3, 3)

img1 = cv2.imread('data/taipei/images/180808-4747.tif')
img2 = cv2.imread('data/taipei/images/180808-4748.tif')

img1 = img1[3100:6300, 2800:6000]
img2 = img2[3100:6300, 2800:6000]

cameraMatrix[0, 2] -= 2800
cameraMatrix[1, 2] -= 3100

print(cameraMatrix)

cv2.imwrite('data/taipei/small/180808-4747.tif', img1)
cv2.imwrite('data/taipei/small/180808-4748.tif', img2)

