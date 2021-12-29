import cv2
import numpy as np
import tools

# 找棋盘格角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#棋盘格模板规格
w = 9
h = 8

objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1,2)

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点

for img_no in range(1, 10):
    print('image no', img_no)
    image = cv2.imread('data/test1/{}.JPG'.format(img_no))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

    # 如果找到足够点对，将其存储起来
    if ret == True:
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners.reshape(-1, 2))
        # print(corners[0][0])
        # # # 将角点在图像上显示
        # cv2.drawChessboardCorners(image, (w, h), corners, ret)
        # plt.imshow(image)
        # plt.show()
    else:
        print('Pattern not found')

# 标定
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

iop = {
    'cameraMatrix': cameraMatrix,
    'distCoeffs': distCoeffs,
    'rvecs': rvecs,
    'tvecs': tvecs,
    'objpoints': objp,
    'imgpoints': imgpoints
}

tools.save(iop, 'data/test1/iop.pk')