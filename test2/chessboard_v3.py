from algorithm.lib import *
import tools

w = 9
h = 6

iop = tools.load('data/calib_imgs/iop.pk')
cameraMatrix = iop['cameraMatrix']
distCoeffs = iop['distCoeffs']

img1 = cv2.imread('data/calib_imgs/left1.jpg')
img2 = cv2.imread('data/calib_imgs/right1.jpg')

imgpoints1 = iop['imgpoints']['left1'].reshape(-1, 1, 2)
imgpoints2 = iop['imgpoints']['right1'].reshape(-1, 1, 2)

# cv2.drawChessboardCorners(img1, (w, h), imgpoints1[0], True)
# plt.imshow(img1)
# plt.show()

# cv2.drawChessboardCorners(img2, (w, h), imgpoints2[0], True)
# plt.imshow(img2)
# plt.show()

imgpoints1_u = cv2.undistortPoints(imgpoints1, cameraMatrix, distCoeffs).reshape(-1, 2)
imgpoints2_u = cv2.undistortPoints(imgpoints2, cameraMatrix, distCoeffs).reshape(-1, 2)

E, mask = cv2.findEssentialMat(imgpoints1_u, imgpoints2_u)
print('total={}, accept={} ({:.2f}%), ignore={}'.format(len(mask), sum(mask), float(sum(mask)/len(mask)*100), len(mask)-sum(mask)))
pass_count, R, T, mask = cv2.recoverPose(E, imgpoints1_u, imgpoints2_u, mask=mask)

new_image_size = (img1.shape[1], img1.shape[0])
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix, distCoeffs, cameraMatrix, distCoeffs, new_image_size, R, T, alpha=1)

map11, map12 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R1, P1, new_image_size, cv2.CV_32FC1)
map21, map22 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R2, P2, new_image_size, cv2.CV_32FC1)

# rectify
img1_rect = cv2.remap(img1, map11, map12, cv2.INTER_LINEAR)
img2_rect = cv2.remap(img2, map21, map22, cv2.INTER_LINEAR)

# print('Find boundary')
# b_north_1, b_south_1, b_east_1, b_west_1 = boundary(img1_rect)
# b_north_2, b_south_2, b_east_2, b_west_2 = boundary(img2_rect)
#
# b_north = min(b_north_1, b_north_2)
# b_south = max(b_south_1, b_south_2)
# b_east = min(b_east_1, b_east_2)
# b_west = max(b_west_1, b_west_2)
#
# img1_rect = img1_rect[b_north:b_south, b_east:b_west]
# img2_rect = img2_rect[b_north:b_south, b_east:b_west]

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
