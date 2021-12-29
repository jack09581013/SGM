from algorithm.lib import *
import tools

w = 9
h = 6

iop = tools.load('data/calib_imgs/rotate/iop.pk')
cameraMatrix = iop['cameraMatrix']
distCoeffs = iop['distCoeffs']


img1 = cv2.imread('data/calib_imgs/rotate/left1.jpg')
img2 = cv2.imread('data/calib_imgs/rotate/right1.jpg')

R1 = cv2.Rodrigues(iop['rvecs'][0])[0]
R2 = cv2.Rodrigues(iop['rvecs'][1])[0]
T1 = iop['tvecs'][0]
T2 = iop['tvecs'][1]

Rc = R2.dot(R1.T)
Tc = T2 - Rc.dot(T1)
Rc_rod = cv2.Rodrigues(Rc)[0]

imgpoints1 = [iop['imgpoints']['left1']]
imgpoints2 = [iop['imgpoints']['right1']]
objpoints = [iop['objpoints']]

# cv2.drawChessboardCorners(img1, (w, h), imgpoints1[0], True)
# plt.imshow(img1)
# plt.show()

# cv2.drawChessboardCorners(img2, (w, h), imgpoints2[0], True)
# plt.imshow(img2)
# plt.show()

# array([[-0.9455941 ],
#        [-0.32364528],
#        [ 0.03324963]])

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, cameraMatrix, distCoeffs, cameraMatrix, distCoeffs, img1.shape[0:2])

new_image_size = (img1.shape[1], img1.shape[0])

# flags=cv2.CALIB_ZERO_DISPARITY
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, new_image_size, R, T, alpha=0)

map11, map12 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, new_image_size, cv2.CV_32FC1)
map21, map22 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, new_image_size, cv2.CV_32FC1)

# rectify
img1_rect = cv2.remap(img1, map11, map12, cv2.INTER_LINEAR)
img2_rect = cv2.remap(img2, map21, map22, cv2.INTER_LINEAR)

# ----- Principle point -----
# img1 = cv2.circle(img1, tuple(np.around(cameraMatrix[0:2, 2]).astype('int32')), 2, (0, 0, 255), -1)
# img2 = cv2.circle(img2, tuple(np.around(cameraMatrix[0:2, 2]).astype('int32')), 2, (0, 0, 255), -1)
#
img1_rect = cv2.circle(img1_rect, tuple(np.around(P1[0:2, 2]).astype('int32')), 2, (0, 0, 255), -1)
img2_rect = cv2.circle(img2_rect, tuple(np.around(P2[0:2, 2]).astype('int32')), 2, (0, 0, 255), -1)
img1_rect = cv2.rectangle(img1_rect, roi1[0:2], roi1[2:4], (0, 0, 255), thickness=1)
img2_rect = cv2.rectangle(img2_rect, roi2[0:2], roi2[2:4], (0, 0, 255), thickness=1)

# cv2.imwrite('left_rect.tif', img1_rect)
# cv2.imwrite('right_rect.tif', img2_rect)

color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
i = 0
# print('Draw line, gap = 200')
for row in range(0, img1_rect.shape[0], 20):
    if i > 2:
        i = 0
    img1_rect = cv2.line(img1_rect, (0, row), (img1_rect.shape[1], row), color[i], 1)
    img2_rect = cv2.line(img2_rect, (0, row), (img2_rect.shape[1], row), color[i], 1)
    i += 1

img_merge = np.concatenate((img1_rect, img2_rect), axis=1)

cv2.imwrite('merge.tif', img_merge)
