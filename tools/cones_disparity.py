from algorithm.lib import *

img1 = cv2.imread('data/cones/conesQ/im2.png')
img2 = cv2.imread('data/cones/conesQ/im6.png')

blockSize = 5

P1 = int(8*3*blockSize*blockSize)   # 600
P2 = int(32*3*blockSize*blockSize)  # 2400

stereo = cv2.StereoSGBM.create(0, 64, blockSize, P1=P1, P2=P2)

disp = stereo.compute(img1, img2)
disp_nor = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disp_eql = cv2.equalizeHist(disp_nor.astype('uint8'), None)

cv2.imwrite('data/cones/conesQ/out.png', disp_nor)