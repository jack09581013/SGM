from algorithm.lib import *
import pptk
import struct
import tools

print('Read file')

# filename = '../data/taipei/sgm/ps-0-points.ary'
# filename = '../data/taipei/sgm/0-rect-points.ary'
# filename = '../data/taipei/sgm/0-rectM-points.ary'
filename = '../data/taipei/sgm/0-photogrammetry-points.ary'
# filename = '../data/taipei/sgm/0-trueortho-points.ary'

with open(filename, "rb") as file:
    size = struct.unpack_from('<I', file.read(4))[0]

    # Coordinates
    rows = struct.unpack_from('<Q', file.read(8))[0]
    cols = struct.unpack_from('<Q', file.read(8))[0]
    element_size = struct.unpack_from('<B', file.read(1))[0]
    cv_type = struct.unpack_from('<B', file.read(1))[0]

    if cv_type == 5:
        coordinates = np.frombuffer(file.read(rows * cols * 4), dtype='<f')
    elif cv_type == 6:
        coordinates = np.frombuffer(file.read(rows * cols * 8), dtype='<f8')
    coordinates = coordinates.reshape(-1, 3)

    # BGR
    rows = struct.unpack_from('<Q', file.read(8))[0]
    cols = struct.unpack_from('<Q', file.read(8))[0]
    element_size = struct.unpack_from('<B', file.read(1))[0]
    cv_type = struct.unpack_from('<B', file.read(1))[0]

    bgr = np.frombuffer(file.read(rows * cols), dtype='<B')
    bgr = bgr.reshape(-1, 3)

# camera_data = tools.load('../data/taipei/parameters/camera.pkl')
# R11, R12, P1, P2, Q = tools.load('../data/taipei/rectify/4747_4748_params.pkl')
# R1 = camera_data['4747']['R']
# T1 = camera_data['4748']['T']
# M = np.array([[-1, 0, 0],
#               [0, -1, 0],
#               [0, 0, -1]])
# R_photogrammetry = np.array([[-0.340939236291931,-0.939823540381523,0.0221844563019801],
#                             [-0.939299283937019,0.339594472564166,-0.0489126711524645],
#                             [0.0384355610350326,-0.0375140926667016,-0.998556658632507]])
# coordinates = M.dot(R11.T.dot(coordinates.T)).T
# coordinates = R_photogrammetry.dot(coordinates.T).T


# Reading opencv yml
# data = cv2.FileStorage("data/taipei/sgm/point_cloud.yml", cv2.FILE_STORAGE_READ)
# coordinates = data.getNode("coordinates").mat()
# bgr = data.getNode("bgr").mat()

print('Number of points:', coordinates.shape[0])
rgb = np.zeros(bgr.shape, dtype='float64')
rgb[:, 0] = bgr[:, 2]
rgb[:, 1] = bgr[:, 1]
rgb[:, 2] = bgr[:, 0]

# coordinates = coordinates[:10000000, :]
# rgb = rgb[:10000000, :]

print('Display point cloud')
v = pptk.viewer(coordinates)
v.attributes(rgb / 255, 0.5 * (1 + rgb))
