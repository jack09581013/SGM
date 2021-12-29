import cv2
import numpy as np
import tools

w = 9
h = 8

iop = tools.load('data/test1/iop.pk')
cameraMatrix = iop['cameraMatrix']
distCoeffs = iop['distCoeffs']

img1 = cv2.imread('data/test1/2.JPG')
img2 = cv2.imread('data/test1/5.JPG')

imgpoints1 = iop['imgpoints'][1].reshape(-1,1,2)
imgpoints2 = iop['imgpoints'][4].reshape(-1,1,2)

img1u = cv2.undistort(img1, cameraMatrix, distCoeffs)
img2u = cv2.undistort(img2, cameraMatrix, distCoeffs)

imgpoints1_u = cv2.undistortPoints(imgpoints1, cameraMatrix, distCoeffs).reshape(-1, 2)
imgpoints2_u = cv2.undistortPoints(imgpoints2, cameraMatrix, distCoeffs).reshape(-1, 2)

one_array = np.ones((imgpoints1_u.shape[0], 1))

imgpoints1_xyz = np.concatenate([imgpoints1_u, one_array], axis=1).T
imgpoints2_xyz = np.concatenate([imgpoints2_u, one_array], axis=1).T

imgpoints1_u_rc = cameraMatrix.dot(imgpoints1_xyz)[:2].T
imgpoints2_u_rc = cameraMatrix.dot(imgpoints2_xyz)[:2].T

imgpoints1_u_rc = imgpoints1_u_rc.reshape(-1, 1, 2).astype('float32')
imgpoints2_u_rc = imgpoints2_u_rc.reshape(-1, 1, 2).astype('float32')


# cv2.drawChessboardCorners(img1, (w, h), imgpoints1, True)
# cv2.drawChessboardCorners(img2, (w, h), imgpoints2, True)
#
# plt.subplot(121)
# plt.imshow(img1)
# plt.subplot(122)
# plt.imshow(img2)

# cv2.drawChessboardCorners(img1u, (w, h), imgpoints1_u_rc, True)
# cv2.drawChessboardCorners(img2u, (w, h), imgpoints2_u_rc, True)
#
# plt.subplot(121)
# plt.imshow(img1u)
# plt.subplot(122)
# plt.imshow(img2u)


# imgpoints1 = imgpoints1.reshape(-1, 2)
# imgpoints2 = imgpoints2.reshape(-1, 2)

# for p in imgpoints1:
#     img1 = cv2.circle(img1, tuple(p.astype('int32')), 3, (0, 0, 255), -1)  # BGR
#
# for p in imgpoints1:
#     img1 = cv2.circle(img2, tuple(p.astype('int32')), 3, (0, 0, 255), -1)

F, mask = cv2.findFundamentalMat(imgpoints1_u_rc, imgpoints2_u_rc)

# lines1 = cv2.computeCorrespondEpilines(imgpoints2_u.reshape(-1,1,2).astype('float32'), 2, F)
# lines2 = cv2.computeCorrespondEpilines(imgpoints1_u.reshape(-1,1,2).astype('float32'), 1, F)
#
# lines1 = lines1.reshape(-1,3)
# img5, img6 = lib.drawlines(img1u, img2u, lines1, imgpoints1_u, imgpoints2_u)
#
# lines2 = lines2.reshape(-1,3)
# img3, img4 = lib.drawlines(img2u, img1u, lines2, imgpoints2_u, imgpoints1_u)


#
retval, H1, H2 = cv2.stereoRectifyUncalibrated(imgpoints1_u_rc, imgpoints2_u_rc, F, (img1.shape[1], img1.shape[0]))
#
# # K = np.eye(3)
# # invK = np.linalg.inv(K)
# # R1 = invK*H1*K
# # R2 = invK*H2*K
#
# # map11, map12 = cv2.initUndistortRectifyMap(K, distCoeffs, R1, K, (img1.shape[1], img1.shape[0]), cv2.CV_32FC1)
# # map21, map22 = cv2.initUndistortRectifyMap(K, distCoeffs, R2, K, (img1.shape[1], img1.shape[0]), cv2.CV_32FC1)
# #
# # img1_rect = cv2.remap(img1, map11, map12, cv2.INTER_LINEAR)
# # img2_rect = cv2.remap(img2, map21, map22, cv2.INTER_LINEAR)
#
img1_rect = cv2.warpPerspective(img1u, H1, (img1.shape[1], img1.shape[0]))
img2_rect = cv2.warpPerspective(img2u, H2, (img2.shape[1], img2.shape[0]))

# plt.subplot(121)
# plt.imshow(img1_rect)
# plt.subplot(122)
# plt.imshow(img2_rect)

color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
i = 0
for row in range(0, img1_rect.shape[0], 50):
    if i > 2:
        i = 0
    img1_rect = cv2.line(img1_rect, (0, row), (img1_rect.shape[1], row), color[i], 1)
    img2_rect = cv2.line(img2_rect, (0, row), (img2_rect.shape[1], row), color[i], 1)
    i += 1

img_merge = np.concatenate((img1_rect, img2_rect), axis=1)

cv2.imwrite('merge.tif', img_merge)