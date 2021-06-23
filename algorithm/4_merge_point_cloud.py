from algorithm.lib import *
import pptk
import tools

point_cloud_all = []
intensities_all = []
coordinates_all = []

for i in range(4747, 4748):
    img_name1 = str(i)
    img_name2 = str(i+1)

    point_cloud, intensities_new, coordinates = tools.load('data/taipei/point cloud/{}_{}.pcpkl'.format(img_name1, img_name2))

    point_cloud_all.append(point_cloud)
    intensities_all.append(intensities_new)
    coordinates_all.append(coordinates)

point_cloud_all = np.concatenate(point_cloud_all)
intensities_all = np.concatenate(intensities_all)
coordinates_all = np.concatenate(coordinates_all)

tools.save((point_cloud_all, intensities_all, coordinates_all), 'data/taipei/point cloud/all.pcpkl')

print('number of points: {:,}'.format(point_cloud_all.shape[0]))

v = pptk.viewer(point_cloud_all)
v.attributes(intensities_all/255.0, 0.5*(1+point_cloud_all))