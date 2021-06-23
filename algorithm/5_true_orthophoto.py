from algorithm.lib import *
import tools

tools.tic()
point_cloud_all, intensities_all, coordinates = tools.load('data/taipei/point cloud/all.pcpkl')
pixel_size = 0.0001

min_x = np.min(point_cloud_all[:, 0])
max_x = np.max(point_cloud_all[:, 0])
min_y = np.min(point_cloud_all[:, 1])
max_y = np.max(point_cloud_all[:, 1])
max_d = np.max(point_cloud_all[:, 2])

w = int((max_x - min_x)/pixel_size)
h = int((max_y - min_y)/pixel_size)
print('w={}, h={}'.format(w, h))
print('total pixels={:,}'.format(w*h))
print('number of points: {:,}'.format(point_cloud_all.shape[0]))

frame = np.zeros((h+1, w+1, 3))

# point_cloud_all[:, 0] = (point_cloud_all[:, 0] - min_x)/pixel_size
# point_cloud_all[:, 1] = (point_cloud_all[:, 1] - min_y)/pixel_size
#
# group = {}
#
# for i in range(frame.shape[0]):
#     for j in range(frame.shape[1]):
#         group[(i, j)] = []
#
# for i in range(point_cloud_all.shape[0]):
#     c = point_cloud_all[i, 0]
#     r = point_cloud_all[i, 1]
#     d = point_cloud_all[i, 2]
#
#     group[(int(r), int(c))].append((r, c, d, i))
#
# tools.save(group, 'group.pkl')
group = tools.load('group.pkl')

print('Integration')
for i in range(frame.shape[0]):
    if i % 100 == 0:
        print('i', i)

    for j in range(frame.shape[1]):
        g = group[(i, j)]
        if len(g) > 0:
            # --- AVG ---
            high_pixel = sorted(g, key=lambda x: x[2])[:7]
            high_intensity = intensities_all[[x[3] for x in high_pixel], :]

            avg = np.mean(high_intensity, axis=0)
            frame[i, j, :] = avg[:]

            # --- IDW ---
            # high_pixel = sorted(g, key=lambda x: x[2])[:math.ceil(len(g) * 0.5)]
            # high_intensity = intensities_all[[x[3] for x in high_pixel], :]
            # cr = i + 0.5
            # cc = j + 0.5
            # cz = high_pixel[0][2]
            #
            # distance = []
            # for r, c, z, k in high_pixel:
            #     rd = r - cr
            #     cd = c - cc
            #     zd = (z - cz)/pixel_size
            #     distance.append(math.sqrt(rd*rd + cd*cd + zd*zd))
            #
            # sz = 0
            # sd = 0
            # for k in range(len(high_pixel)):
            #     sz += high_intensity[k, :]/math.pow(distance[k], 3)
            #     sd += 1/math.pow(distance[k], 3)
            # intensity = sz/sd
            # frame[i, j, :] = intensity[:]

frame_new = np.zeros(frame.shape)
frame_new[:, :, 0] = frame[:, :, 2]
frame_new[:, :, 1] = frame[:, :, 1]
frame_new[:, :, 2] = frame[:, :, 0]
frame_new = frame_new.astype('uint8')

tools.toc()
cv2.imwrite('data/taipei/true orthophoto/mean.tif', frame_new)