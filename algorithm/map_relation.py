from algorithm.lib import *
import tools

def distort2rectify(points, K, D, R, P):
    # point on row, column
    # K: camera matrix
    # D distortion coefficient
    # R: rotation matrix from original image to rectify image
    # P: projection matrix for rectify image

    points = np.array(points).astype('float64').reshape(-1, 1, 2)
    points = cv2.undistortPoints(points, K, D).reshape(-1, 2).T
    points = np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)
    points = P[:, 0:3].dot(R.dot(points))
    points[0:2, :] /= points[2, :]
    return points[0:2, :].T

def rectify2distort(points, K, D, R, P):
    # point on row, column
    # K: camera matrix
    # D distortion coefficient
    # R: rotation matrix from original image to rectify image
    # P: projection matrix for rectify image
    points = np.array(points).T.astype('float64')
    points[0, :] = (points[0, :] - P[0, 2]) / P[0, 0]  # row
    points[1, :] = (points[1, :] - P[1, 2]) / P[1, 1]  # column
    points = np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)
    points = R.T.dot(points)

    # add distortion
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]

    points[0:2, :] /= points[2, :]
    u = points[0, :]
    v = points[1, :]
    r2 = u * u + v * v
    du = (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) * u + 2 * p1 * u * v + p2 * (r2 + 2 * u * u)
    dv = (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) * v + 2 * p2 * u * v + p1 * (r2 + 2 * v * v)
    du = du.reshape(1, -1)
    dv = dv.reshape(1, -1)

    points = K.dot(np.concatenate([du, dv, np.ones((1, points.shape[1]))], axis=0))
    points[0:2, :] /= points[2, :]
    return points[0:2, :].T

def rectify2undistort(points, K, R, P):
    # point on row, column
    # K: camera matrix
    # D distortion coefficient
    # R: rotation matrix from original image to rectify image
    # P: projection matrix for rectify image
    points = np.array(points).T.astype('float64')
    points[0, :] = (points[0, :] - P[0, 2]) / P[0, 0]  # row
    points[1, :] = (points[1, :] - P[1, 2]) / P[1, 1]  # column
    points = np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)

    points = K.dot(R.T.dot(points))
    points[0:2, :] /= points[2, :]
    return points[0:2, :].T

def undistort2rectify(points, K, R, P):
    # point on row, column
    # K: camera matrix
    # D distortion coefficient
    # R: rotation matrix from original image to rectify image
    # P: projection matrix for rectify image
    points = np.array(points).T.astype('float64')
    points[0, :] = (points[0, :] - K[0, 2]) / K[0, 0]  # row
    points[1, :] = (points[1, :] - K[1, 2]) / K[1, 1]  # column
    points = np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)

    points = P[:, 0:3].dot(R.dot(points))
    points[0:2, :] /= points[2, :]
    return points[0:2, :].T

def initRectifyMap(K, R, P, size):
    points = np.mgrid[0:size[1], 0:size[0]].reshape(2, -1).T
    map = rectify2undistort(points, K, R, P).astype('float32')
    map1 = map[:, 0].reshape((size[1], size[0])).T
    map2 = map[:, 1].reshape((size[1], size[0])).T
    return map1, map2

def initUndistortRectifyMap(K, D, R, P, size):
    points = np.mgrid[0:size[1], 0:size[0]].reshape(2, -1).T
    map = rectify2distort(points, K, D, R, P).astype('float32')
    map1 = map[:, 0].reshape((size[1], size[0])).T
    map2 = map[:, 1].reshape((size[1], size[0])).T
    return map1, map2

# Testing
if __name__ == '__main__':
    k = 4747
    img_name1 = str(k)
    img_name2 = str(k + 1)

    distCoeffs = np.array(
        [-0.04591155158913425, 0.035591475053250415, -1.3125789551377542e-006, -3.3011941430374701e-006,
         -0.013634871666012505]).reshape(5, 1)
    cameraMatrix = np.array(
        [10912.905472192366, 0, 5800.8672228100659, 0, 10912.905472192366, 4348.9916944617316, 0, 0, 1]).reshape(3, 3)

    img1 = cv2.imread('data/taipei/images/180808-{}.tif'.format(img_name1))
    img2 = cv2.imread('data/taipei/images/180808-{}.tif'.format(img_name2))
    img1u = cv2.undistort(img1, cameraMatrix, distCoeffs)
    img2u = cv2.undistort(img2, cameraMatrix, distCoeffs)
    R11, R12, P1, P2, Q = tools.load('data/taipei/test_map/{}_{}_params.pkl'.format(img_name1, img_name2))

    # Test initUndistortRectifyMap
    map11, map12, map21, map22 = tools.load('data/taipei/test_map/{}_{}_maps.pkl'.format(img_name1, img_name2))
    print(map11[100, 200], map12[100, 200])
    new_image_size = (img1.shape[0], img1.shape[1])
    map11, map12 = initUndistortRectifyMap(cameraMatrix, distCoeffs, R11, P1, img1.shape[:2])
    print(map11[100, 200], map12[100, 200])

    # map11, map12 = initRectifyMap(cameraMatrix, R11, P1, new_image_size)
    # map21, map22 = initRectifyMap(cameraMatrix, R12, P2, new_image_size)

    # img1_rect = cv2.remap(img1u, map11, map12, cv2.INTER_LINEAR)
    # img2_rect = cv2.remap(img2u, map21, map22, cv2.INTER_LINEAR)
    #
    # cv2.imwrite('data/taipei/rectify/{0}_P{0}_{1}_undistort_rectify.tif'.format(img_name1, img_name2), img1_rect)
    # cv2.imwrite('data/taipei/rectify/{1}_P{0}_{1}_undistort_rectify.tif'.format(img_name1, img_name2), img2_rect)
    #
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
    # cv2.imwrite('data/taipei/epipolar line/{}_{}_undistort_epipolar_line.tif'.format(img_name1, img_name1), img_merge)
