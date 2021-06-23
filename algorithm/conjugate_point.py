from algorithm.lib import *
import tools

f = 10912.905472192366
cx = 5800.8672228100659
cy = 4348.9916944617316

cor_pairs = []

pre_map2 = None
pre_disp = None

for k in range(4747, 4749):
    tools.tic()
    img_name1 = str(k)
    img_name2 = str(k+1)
    print('Process image pair:', img_name1, img_name2)
    cor_pair = []

    img1 = cv2.imread('data/taipei/images/180808-{}.tif'.format(img_name1))
    img2 = cv2.imread('data/taipei/images/180808-{}.tif'.format(img_name2))

    img1_rect = cv2.imread('data/taipei/rectify/{0}_P{0}_{1}_rectify.tif'.format(img_name1, img_name2))
    img2_rect = cv2.imread('data/taipei/rectify/{1}_P{0}_{1}_rectify.tif'.format(img_name1, img_name2))

    R11, R12, P1, P2, Q = tools.load('data/taipei/rectify/{}_{}_params.pkl'.format(img_name1, img_name2))
    map11, map12, map21, map22 = tools.load('data/taipei/rectify/{0}_{1}_maps.pkl'.format(img_name1, img_name2))
    disp = tools.load('data/taipei/sgm/{}_{}_1024_5_disp.pkl'.format(img_name1, img_name2))

    point_cloud, intensities_new, coordinates = tools.load('data/taipei/point cloud/{}_{}_point_cloud.pkl'.format(img_name1, img_name2))

    coordinates = coordinates.astype('int32')
    for i in range(coordinates.shape[0]):
        x = coordinates[i, 0]
        y = coordinates[i, 1]

        # Rectification coordinate to original coordinate
        coordinates[i, :] = [map11[x, y], map12[x, y]]

    if pre_map2 is None and pre_disp is None:
        pre_disp = disp
        pre_map2 = []
        map21 = map21.astype('int32')
        map22 = map22.astype('int32')

        for i in range(map21.shape[0]):
            for j in range(map21.shape[1]):
                # Original coordinate to rectification coordinate
                pre_map2.append([map21[i, j], map22[i, j], i, j])
        pre_map2 = np.array(pre_map2)

    else:
        for i in range(coordinates.shape[0]):
            x = coordinates[i, 0]
            y = coordinates[i, 1]
            index = (pre_map2[:, 0] == x) & (pre_map2[:, 1] == y)
            index = np.argmax(index)
            coordinates[i, 0] = pre_map2[index, 2]
            coordinates[i, 1] = pre_map2[index, 3]
        coordinates[i, 1] += pre_disp[coordinates[i, 0], coordinates[i, 1]]/16
    tools.toc()


# cv2.circle(img1_rect, (2000, 2000), 5, (255, 0, 0), -1)
# cv2.circle(img2_rect, (2000, int(2000 + disp[2000, 2000]/16)), 5, (255, 0, 0), -1)
#
# plt.subplot(121)
# plt.imshow(img1_rect)
# plt.subplot(122)
# plt.imshow(img2_rect)
# plt.show()

# u1 = 2000
# v1 = 2000
#
# p_rect = np.array([u1 - cx, v1 - cy, f]).reshape(-1, 1)
# p_orin = R11.T.dot(p_rect)
#
# print(p_orin)


