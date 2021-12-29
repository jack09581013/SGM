from algorithm.lib import *
import tools

w = 9
h = 8

iop = tools.load('data/calib_imgs/iop.pk')
cameraMatrix = iop['cameraMatrix']
distCoeffs = iop['distCoeffs']

img1 = cv2.imread('data/calib_imgs/left1.jpg')
img2 = cv2.imread('data/calib_imgs/right1.jpg')

imgpoints1 = iop['imgpoints']['left1'].reshape(-1, 1, 2)
imgpoints2 = iop['imgpoints']['right1'].reshape(-1, 1, 2)

img1u = cv2.undistort(img1, cameraMatrix, distCoeffs)
img2u = cv2.undistort(img2, cameraMatrix, distCoeffs)

imgpoints1_u = cv2.undistortPoints(imgpoints1, cameraMatrix, distCoeffs).reshape(-1, 2)
imgpoints2_u = cv2.undistortPoints(imgpoints2, cameraMatrix, distCoeffs).reshape(-1, 2)

one_array = np.ones((imgpoints1_u.shape[0], 1))

imgpoints1_xyz = np.concatenate([imgpoints1_u, one_array], axis=1).T
imgpoints2_xyz = np.concatenate([imgpoints2_u, one_array], axis=1).T

imgpoints1_u_rc = cameraMatrix.dot(imgpoints1_xyz)[:2].T
imgpoints2_u_rc = cameraMatrix.dot(imgpoints2_xyz)[:2].T

imgpoints1_u_rc = imgpoints1_u_rc.astype('float32')
imgpoints2_u_rc = imgpoints2_u_rc.astype('float32')

F, mask = cv2.findFundamentalMat(imgpoints1_u_rc, imgpoints2_u_rc)
retval, H1, H2 = cv2.stereoRectifyUncalibrated(imgpoints1_u_rc, imgpoints2_u_rc, F, img1.shape[0:2])
img1_rect = cv2.warpPerspective(img1u, H1, (img1u.shape[0]*2, img1u.shape[1]*2))
img2_rect = cv2.warpPerspective(img2u, H2, (img2u.shape[0]*2, img2u.shape[1]*2))

print('Find boundary')
b_north_1, b_south_1, b_east_1, b_west_1 = boundary(img1_rect)
b_north_2, b_south_2, b_east_2, b_west_2 = boundary(img2_rect)

b_north = min(b_north_1, b_north_2)
b_south = max(b_south_1, b_south_2)
b_east = min(b_east_1, b_east_2)
b_west = max(b_west_1, b_west_2)

img1_rect = img1_rect[b_north:b_south, b_east:b_west]
img2_rect = img2_rect[b_north:b_south, b_east:b_west]

cv2.imwrite('left.tif', img1_rect)
cv2.imwrite('right.tif', img2_rect)

color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
i = 0
for row in range(0, img1_rect.shape[0], 20):
    if i > 2:
        i = 0
    img1_rect = cv2.line(img1_rect, (0, row), (img1_rect.shape[1], row), color[i], 1)
    img2_rect = cv2.line(img2_rect, (0, row), (img2_rect.shape[1], row), color[i], 1)
    i += 1

img_merge = np.concatenate((img1_rect, img2_rect), axis=1)

cv2.imwrite('merge.tif', img_merge)