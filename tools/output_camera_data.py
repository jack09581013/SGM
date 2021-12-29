from algorithm.lib import *
import tools
from math import *

def get_rotation_matrix(omega, phi, kappa):
    Rx = np.array([[1, 0, 0],
                   [0, cos(omega), sin(omega)],
                   [0, -sin(omega), cos(omega)]])
    Ry = np.array([[cos(phi), 0, -sin(phi)],
                   [0, 1, 1],
                   [sin(phi), 0, cos(phi)]])
    Rz = np.array([[cos(kappa), sin(kappa), 0],
                  [-sin(kappa), cos(kappa), 0],
                  [0, 0, 1]])

    return Rz.dot(Ry).dot(Rx)

def calcROP_C2W(R1, T1, R2, T2):
    # [R|T]*C = W
    R = R2.T.dot(R1) # left camera to right camera
    T = R2.T.dot(T1 - T2) # right camera to left camera in right camera coordinate system
    return R, T

def calcROP_W2C(R1, T1, R2, T2):
    # [R|T]*W = C
    R = R2.dot(R1.T)  # left camera to right camera
    T = T2 - R.dot(T1)  # right camera to left camera in right camera coordinate system
    return R, T

def calcROP_YG(R1, T1, R2, T2, norm):
    Ry = R1.dot(R2.T)
    Ty = - T2 + Ry.T.dot(T1)
    if norm:
        Ty = Ty/np.linalg.norm(Ty)
    return Ry, Ty


def readYml(filename):
    with open(filename, 'r') as file:
        text = file.read().replace('\n', '')
        result = text.split('!!opencv-matrix')
        params = {}
        params['R'] = []
        params['T'] = []

        for mat in result[1:]:
            start = mat.find('data: [')
            end = mat.find(']')
            matrix = mat[start + 7:end - 1].split(',')
            ary = []
            for x in matrix:
                ary.append(float(x))

            if len(ary) == 9:
                # Rotation Matrix
                ary = np.array(ary, dtype='float64').reshape(3, 3)
                params['R'].append(ary)
            elif len(ary) == 3:
                # Rotation Matrix
                ary = np.array(ary, dtype='float64').reshape(3, 1)
                params['T'].append(ary)
        return params


def toCsArray(m):
    csstr = 'new double[,] {'
    for row in m:
        csstr += '{'
        for x in row:
            csstr += str(x) + ', '
        csstr = csstr[:-2] + '}, '
    csstr = csstr[:-2] + '};'
    return csstr

def toCppArray(m):
    cppstr = '(Mat_<double>(3, 3) << '
    for x in m.reshape(-1):
        cppstr += str(x) + ', '
    cppstr = cppstr[:-2] + ');'
    return cppstr

def checkROP(a1, a2, camera_data, params):
    R1 = params['R'][a1]
    R2 = params['R'][a2]
    T1 = params['T'][a1]
    T2 = params['T'][a2]

    print('雅筑')
    Ry, Ty = calcROP_YG(R1, T1, R2, T2, norm=True)

    print('Ry')
    print(Ry)
    print('Ty')
    print(Ty)
    print()

    R1 = camera_data[str(4747 + a1)]['R']
    R2 = camera_data[str(4747 + a2)]['R']
    T1 = camera_data[str(4747 + a1)]['T']
    T2 = camera_data[str(4747 + a2)]['T']

    print('Photoscan')
    Rp, Tp = calcROP_C2W(R1, T1, R2, T2)
    Tp = Tp/np.linalg.norm(Tp)

    print('Rp')
    print(Rp)
    print('Tp')
    print(Tp)
    print()

    print('R diff')
    print(np.abs(Rp - Ry))
    print('T diff')
    print(np.abs(Tp - Ty))

def checkROP_YG_EOP(a1, a2, camera_data, YG_EOP):
    R1 = YG_EOP[a1][0]
    R2 = YG_EOP[a2][0]
    T1 = YG_EOP[a1][1]
    T2 = YG_EOP[a2][1]


    print('雅筑')
    Ry, Ty = calcROP_W2C(R1, T1, R2, T2)
    Ty = Ty / np.linalg.norm(Ty)

    print('Ry')
    print(Ry)
    print('Ty')
    print(Ty)
    print()

    R1 = camera_data[str(4747 + a1)]['R']
    R2 = camera_data[str(4747 + a2)]['R']
    T1 = camera_data[str(4747 + a1)]['T']
    T2 = camera_data[str(4747 + a2)]['T']

    print('Photoscan')
    Rp, Tp = calcROP_C2W(R1, T1, R2, T2)

    Tp = Tp/np.linalg.norm(Tp)
    print('Rp')
    print(Rp)
    print('Tp')
    print(Tp)
    print()

    print('R diff')
    print(np.abs(Rp - Ry))
    print('T diff')
    print(np.abs(Tp - Ty))


def produceEROP_text_file(camera_data):
    with open('data/taipei/parameters/photoscan_EOP.txt', 'w') as file:
        for i in range(4747, 4767):
            n1 = str(i)
            print(n1, file=file)
            print('R', file=file)
            print(camera_data[n1]['R'], file=file)
            print('T', file=file)
            print(camera_data[n1]['T'], file=file)
            print(file=file)

    with open('data/taipei/parameters/photoscan_ROP.txt', 'w') as file:
        for i in range(4747, 4766):
            n1 = str(i)
            n2 = str(i+1)
            Rp, Tp = calcROP_C2W(camera_data[n1]['R'], camera_data[n1]['T'], camera_data[n2]['R'], camera_data[n2]['T'])
            print(n1 + '-' + n2, file=file)
            print('R', file=file)
            print(Rp, file=file)
            print('T', file=file)
            print(Tp, file=file)
            print(file=file)

def print_RT_code(version='photoscan'):
    camera_data = tools.load('../data/taipei/parameters/camera.pkl')
    distCoeffs = np.array(
        [-0.04591155158913425, 0.035591475053250415, -1.3125789551377542e-006, -3.3011941430374701e-006,
         -0.013634871666012505]).reshape(5, 1)
    cameraMatrix = np.array(
        [10912.905472192366, 0, 5800.8672228100659, 0, 10912.905472192366, 4348.9916944617316, 0, 0, 1]).reshape(3, 3)
    M = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])

    computerVisionToPhotogrammetry = np.array([[0, 1, 0],
                                               [1, 0, 0],
                                               [0, 0, -1]])

    angles = np.array([-2.804299, -1.271180, 160.060777]) / 180 * pi
    photogrammetryToTrueOrtho = get_rotation_matrix(angles[0], angles[1], angles[2]).T

    if version == 'photoscan':
        # Photoscan C#
        for i in range(4747, 4767):
            R = camera_data[str(i)]['R']
            T = camera_data[str(i)]['T']

            R = R.T
            T = R.dot(-T)

            print('R[{}] = '.format(i - 4747) + toCsArray(R))
            print('T[{}] = '.format(i - 4747) + toCsArray(T))

        print('double[,] cameraMatrix = ' + toCsArray(cameraMatrix))
        print('double[,] distCoeffs = ' + toCsArray(distCoeffs))
        print('double[,] computerVisionToPhotogrammetry = ' + toCsArray(computerVisionToPhotogrammetry))
        print('double[,] photogrammetryToTrueOrtho = ' + toCsArray(photogrammetryToTrueOrtho))
        print('double[,] _M = ' + toCsArray(M))

    elif version == 'yg':
        # YG EOP
        params = readYml('../data/taipei/parameters/EOP47to50.yml')
        YG_EOP = []

        for i in range(4747, 4751):
            R, T = calcROP_YG(params['R'][0], params['T'][0], params['R'][i - 4747], params['T'][i - 4747], norm=False)
            YG_EOP.append([R, T])

        for i in range(4747, 4751):
            print('R[{}] = '.format(i - 4747) + toCsArray(YG_EOP[i - 4747][0]))
            print('T[{}] = '.format(i - 4747) + toCsArray(YG_EOP[i - 4747][1]))

        print('double[,] cameraMatrix = ' + toCsArray(cameraMatrix))
        print('double[,] distCoeffs = ' + toCsArray(distCoeffs))
        print('double[,] computerVisionToPhotogrammetry = ' + toCsArray(computerVisionToPhotogrammetry))
        print('double[,] photogrammetryToTrueOrtho = ' + toCsArray(photogrammetryToTrueOrtho))
        print('double[,] _M = ' + toCsArray(M))

if __name__ == '__main__':
    versions = ['photoscan', 'yg']
    print_RT_code(versions[0])







