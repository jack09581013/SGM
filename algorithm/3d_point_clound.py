from algorithm.lib import *
import pptk
import tools

def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

camera_data = tools.load('data/taipei/parameters/camera.pkl')
point_cloud_all = []
intensities_all = []

R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

T = np.array([0, 0, 0]).reshape(-1, 1)

M = np.array([[-1, 0, 0],
              [0, -1, 0],
              [0, 0, -1]])

# 4766
for i in range(4747, 4748):
    img_name1 = str(i)
    img_name2 = str(i+1)

    print('Process image pair:', img_name1, img_name2)
    # BGR order
    img1_rect = cv2.imread('data/taipei/rectify/{0}_P{0}_{1}_rectify.tif'.format(img_name1, img_name2))
    Q = tools.load('data/taipei/rectify/{}_{}_Q.pkl'.format(img_name1, img_name2))
    disp = tools.load('data/taipei/sgm/{}_{}_1024_5_disp.pkl'.format(img_name1, img_name2))

    # disp = disp/16
    # vmax = disp.max()
    # vmin = disp.min()
    # plt.imshow(disp, cmap='gray', vmin=vmin, vmax=vmax)

    point_cloud = cv2.reprojectImageTo3D(disp, Q)

    # point_cloud_v = point_cloud[6000:-512, 9000:].reshape(-1, 3)
    # img1_rect_v = img1_rect[6000:-512, 9000:].reshape(-1, 3)
    # point_cloud_v = point_cloud[:-512, :].reshape(-1, 3)
    # img1_rect_v = img1_rect[:-512, :].reshape(-1, 3)
    point_cloud_v = point_cloud[3230:7000, 7388:9336].reshape(-1, 3)
    img1_rect_v = img1_rect[3230:7000, 7388:9336].reshape(-1, 3)

    filter_indexes = (point_cloud_v[:, 2] > 0) &(point_cloud_v[:, 2] < 1)
    point_cloud_v = point_cloud_v[filter_indexes, :]
    img1_rect_v = img1_rect_v[filter_indexes, :]

    accept_index = reject_outliers(point_cloud_v[:, 2], 10)
    point_cloud = point_cloud_v[accept_index, :]
    intensities = img1_rect_v[accept_index, :]

    intensities_new = np.zeros(intensities.shape)
    intensities_new[:, 0] = intensities[:, 2]
    intensities_new[:, 1] = intensities[:, 1]
    intensities_new[:, 2] = intensities[:, 0]

    # coordinate transformation
    R11, R12 = tools.load('data/taipei/rectify/{}_{}_R.pkl'.format(img_name1, img_name2))

    R1 = camera_data[img_name1]['R']
    T1 = camera_data[img_name1]['T']
    R2 = camera_data[img_name2]['R']
    T2 = camera_data[img_name2]['T']

    point_cloud = R.dot(M.dot(R11.T.dot(point_cloud.T)) - T).T

    R = R.dot(R2.T.dot(R1).T)  # right camera to left camera
    T = T + R2.T.dot(T1 - T2)/16  # right camera to left camera

    point_cloud_all.append(point_cloud)
    intensities_all.append(intensities_new)

point_cloud_all = np.concatenate(point_cloud_all)
intensities_all = np.concatenate(intensities_all)

print('number of pixels: {:,}'.format(point_cloud_all.shape[0]))

v = pptk.viewer(point_cloud_all)
v.attributes(intensities_all/255.0, 0.5*(1+point_cloud_all))

tools.save((point_cloud_all, intensities_all), 'data/taipei/point cloud/points_cloud_2.pkl')

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