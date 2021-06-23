from algorithm.map_relation import *

distCoeffs = np.array([-0.04591155158913425, 0.035591475053250415, -1.3125789551377542e-006, -3.3011941430374701e-006, -0.013634871666012505]).reshape(5, 1)
cameraMatrix = np.array([10912.905472192366, 0, 5800.8672228100659, 0, 10912.905472192366, 4348.9916944617316, 0, 0, 1]).reshape(3, 3)
camera_data = tools.load('data/taipei/parameters/camera.pkl')

# img1 = cv2.imread('data/taipei/images/180808-{}.tif'.format(4747))
# img1_u = cv2.undistort(img1, cameraMatrix, distCoeffs)
# cv2.imwrite('data/taipei/images/180808-{}_undistort.tif'.format(4747), img1_u)

# 4766
for i in range(4747, 4755):
    tools.tic()
    img_name1 = str(i)
    img_name2 = str(i+1)

    print('Process image pair:', img_name1, img_name2)

    img1 = cv2.imread('data/taipei/images/180808-{}.tif'.format(img_name1))
    img2 = cv2.imread('data/taipei/images/180808-{}.tif'.format(img_name2))

    R1 = camera_data[img_name1]['R']
    T1 = camera_data[img_name1]['T']
    R2 = camera_data[img_name2]['R']
    T2 = camera_data[img_name2]['T']

    R = R2.T.dot(R1)
    T = R2.T.dot(T1 - T2)

    new_image_size = (img1.shape[1], img1.shape[0])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix, distCoeffs, cameraMatrix, distCoeffs, new_image_size, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)
    # Tx = P2[1, 3]/P2[0, 0]

    # OpenCV rectification function
    # cv2.INTER_LINEAR, cv2.INTER_LANCZOS4
    # 13 second
    map11, map12 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R1, P1, new_image_size, cv2.CV_32FC1)
    map21, map22 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R2, P2, new_image_size, cv2.CV_32FC1)
    img1_rect = cv2.remap(img1, map11, map12, cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2, map21, map22, cv2.INTER_LINEAR)

    # rectification function which is implemented by myself
    # 1 minute
    # img1u = cv2.undistort(img1, cameraMatrix, distCoeffs)
    # img2u = cv2.undistort(img2, cameraMatrix, distCoeffs)
    # map11, map12 = initRectifyMap(cameraMatrix, R1, P1, img1u.shape[0:2])
    # map21, map22 = initRectifyMap(cameraMatrix, R2, P2, img2u.shape[0:2])
    # img1_rect = cv2.remap(img1u, map11, map12, cv2.INTER_LINEAR)
    # img2_rect = cv2.remap(img2u, map21, map22, cv2.INTER_LINEAR)

    # img1_rect = cv2.rotate(img1_rect, cv2.ROTATE_90_CLOCKWISE)
    # img2_rect = cv2.rotate(img2_rect, cv2.ROTATE_90_CLOCKWISE)

    cv2.imwrite('data/taipei/rectify/{0}_P{0}_{1}_rectify.tif'.format(img_name1, img_name2), img1_rect)
    cv2.imwrite('data/taipei/rectify/{1}_P{0}_{1}_rectify.tif'.format(img_name1, img_name2), img2_rect)
    tools.save((R1, R2, P1, P2, Q), 'data/taipei/rectify/{0}_{1}_params.pkl'.format(img_name1, img_name2))
    tools.toc()

# # Draw epipolar lines
# color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
# i = 0
# # print('Draw line, gap = 200')
# for row in range(0, img1_rect.shape[0], 50):
#     if i > 2:
#         i = 0
#     img1_rect = cv2.line(img1_rect, (0, row), (img1_rect.shape[1], row), color[i], 1)
#     img2_rect = cv2.line(img2_rect, (0, row), (img2_rect.shape[1], row), color[i], 1)
#     i += 1
#
# img_merge = np.concatenate((img1_rect, img2_rect), axis=1)
#
# cv2.imwrite('data/taipei/epipolar line/{}_{}_epipolar_line.tif'.format(data['name1'], data['name2']), img_merge)


