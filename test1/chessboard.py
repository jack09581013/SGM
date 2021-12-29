import cv2
import numpy as np
import tools

iop = tools.load('data/test1/iop.pk')
cameraMatrix = iop['cameraMatrix']
distCoeffs = iop['distCoeffs']

img1 = cv2.imread('data/test1/2_L.JPG')
img2 = cv2.imread('data/test1/5_R.JPG')

imgpoints1 = [iop['imgpoints'][1]]
imgpoints2 = [iop['imgpoints'][4]]
objpoints = [iop['objpoints']]

# cv2.drawChessboardCorners(img1, (w, h), imgpoints1[0], True)
# plt.imshow(img1)
# plt.show()
#
# cv2.drawChessboardCorners(img2, (w, h), imgpoints2[0], True)
# plt.imshow(img2)
# plt.show()

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, cameraMatrix, distCoeffs, cameraMatrix, distCoeffs, img1.shape[:2])

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, img1.shape[0:2], R, T)

map11, map12 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, cameraMatrix1, img1.shape[0:2], cv2.CV_32FC1)
map21, map22 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, cameraMatrix2, img2.shape[0:2], cv2.CV_32FC1)

# map11[:, :, 0] -= 1000
# map11[:, :, 1] += 1000
# map21[:, :, 0] -= 1000
# map21[:, :, 1] += 1000

# rectify
img1_rect = cv2.remap(img1, map11, map12, cv2.INTER_LINEAR)
img2_rect = cv2.remap(img2, map21, map22, cv2.INTER_LINEAR)
color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

i = 0
# print('Draw line, gap = 200')
for row in range(0, img1.shape[0], 100):
    if i > 2:
        i = 0
    img1_rect = cv2.line(img1_rect, (0, row), (img1.shape[1], row), color[i], 1)
    img2_rect = cv2.line(img2_rect, (0, row), (img2.shape[1], row), color[i], 1)
    i += 1

img_merge = np.concatenate((img1_rect, img2_rect), axis=1)

cv2.imwrite('merge.tif', img_merge)