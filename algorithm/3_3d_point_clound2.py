from algorithm.lib import *
import tools

def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

camera_data = tools.load('data/taipei/parameters/camera.pkl')
point_cloud_all = []
intensities_all = []

R_photogrammetry = np.array([[-0.340939236291931,-0.939823540381523,0.0221844563019801],
                            [-0.939299283937019,0.339594472564166,-0.0489126711524645],
                            [0.0384355610350326,-0.0375140926667016,-0.998556658632507]])

R1 = None
T1 = None

M = np.array([[-1, 0, 0],
              [0, -1, 0],
              [0, 0, -1]])

numDisparities = 16*50

min_row = 3230
max_row = 7000
min_col = 7388
max_col = 9336

# 4766
for i in range(4747, 4748):
    img_name1 = str(i)
    img_name2 = str(i+1)

    print('Process image pair:', img_name1, img_name2)
    # BGR order
    img1_rect = cv2.imread('data/taipei/rectify/{0}_P{0}_{1}_rectify.tif'.format(img_name1, img_name2))
    R11, R12, P1, P2, Q = tools.load('data/taipei/rectify/{}_{}_params.pkl'.format(img_name1, img_name2))
    disp = tools.load('data/taipei/sgm/D{}_{}.dpkl'.format(img_name1, img_name2))

    point_cloud = cv2.reprojectImageTo3D(disp, Q)

    point_cloud_v = point_cloud[min_row:max_row, min_col:max_col].reshape(-1, 3)
    img1_rect_v = img1_rect[min_row:max_row, min_col:max_col].reshape(-1, 3)

    coordinates = np.zeros((max_row - min_row, max_col - min_col, 2))
    for i in range(0, coordinates.shape[0]):
        for j in range(0, coordinates.shape[1]):
            coordinates[i, j, 0] = i + min_row
            coordinates[i, j, 1] = j + min_col
    coordinates = coordinates.reshape(-1, 2)

    filter_indexes = (point_cloud_v[:, 2] > 0) & (point_cloud_v[:, 2] < 1)
    point_cloud_v = point_cloud_v[filter_indexes, :]
    img1_rect_v = img1_rect_v[filter_indexes, :]
    coordinates = coordinates[filter_indexes, :]

    accept_index = reject_outliers(point_cloud_v[:, 2], 10)
    point_cloud = point_cloud_v[accept_index, :]
    intensities = img1_rect_v[accept_index, :]
    coordinates = coordinates[accept_index, :]

    intensities_new = np.zeros(intensities.shape)
    intensities_new[:, 0] = intensities[:, 2]
    intensities_new[:, 1] = intensities[:, 1]
    intensities_new[:, 2] = intensities[:, 0]

    # coordinate transformation
    if R1 is None and T1 is None:
        R1 = camera_data[img_name1]['R']
        T1 = camera_data[img_name1]['T']
        point_cloud = M.dot(R11.T.dot(point_cloud.T)).T

    else:
        R2 = camera_data[img_name1]['R']
        T2 = camera_data[img_name1]['T']
        R = R2.T.dot(R1).T  # right camera to left camera
        T = R2.T.dot(T1 - T2) / 16  # right camera to left camera
        point_cloud = R.dot(M.dot(R11.T.dot(point_cloud.T)) - T).T

    point_cloud = R_photogrammetry.dot(point_cloud.T).T
    tools.save((point_cloud, intensities_new, coordinates), 'data/taipei/point cloud/{}_{}.pcpkl'.format(img_name1, img_name2))


# ----------- depth --------------
# depth = np.copy(point_cloud[:, :, 2])[:-2048, :]
# depth[depth < 0.56974286] = 0.56974286
# depth[depth > 0.65977824] = 0.65977824
#
# depth = reject_outliers(depth)
#
# order = np.sort(depth.reshape(-1))
# order_shift = np.zeros((order.shape[0]+1,))
# order_shift[1:] = order[:]
# diff = np.abs((order - order_shift[:-1])[1:])
# index = np.argmax(diff)
# print(index, order[index], order[index+1])
# plt.plot(diff)


# rate = []
# for i in np.arange(0, 1, 0.01):
#     print(order.shape[0]*i)
#     rate.append(order[int(order.shape[0]*i)])
# plt.plot(rate)


# DSM = depth.max() - depth
# depth = cv2.normalize(depth, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
# depth = cv2.equalizeHist(depth.astype('uint8'), None)
# DSM = cv2.normalize(DSM, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
# DSM = cv2.equalizeHist(DSM.astype('uint8'), None)
# # plt.imshow(depth, cmap='gray')
#
# cv2.imwrite('data/taipei/3d point cloud/depth_map.tif', depth)
# cv2.imwrite('data/taipei/3d point cloud/DSM.tif', DSM)