from algorithm.lib import *
import tools

numDisparities = 16*50

# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
blockSize = 7

preFilterCap = 31

# Normally, a value within the 5-15 range is good enough.
uniquenessRatio = 15

# Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
disp12MaxDiff = 1

# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckle_window_size = 200

# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
speckle_range = 2

# MODE_SGBM = 0
# MODE_HH = 1
# MODE_SGBM_3WAY = 2
# MODE_HH4 = 3
mode = cv2.STEREO_SGBM_MODE_SGBM

# P1 = int(pow((1/16)*8*5, 2))
# P2 = int(pow((1/16)*32*5, 2))

# 8*number_of_image_channels*SADWindowSize*SADWindowSize
# 32*number_of_image_channels*SADWindowSize*SADWindowSize
P1 = int(8*3*blockSize*blockSize)   # 600
P2 = int(32*3*blockSize*blockSize)  # 2400

print('P1=', P1)
print('P2=', P2)
print('numDisparities=', numDisparities)
print('blockSize=', blockSize)

# 4766
for i in range(4749, 4755):
    tools.tic()
    img_name1 = str(i)
    img_name2 = str(i + 1)

    print('Process image pair:', img_name1, img_name2)

    img1 = cv2.imread('data/taipei/rectify/{0}_P{0}_{1}_rectify.tif'.format(img_name1, img_name2))
    img2 = cv2.imread('data/taipei/rectify/{1}_P{0}_{1}_rectify.tif'.format(img_name1, img_name2))

    img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)

    stereo = cv2.StereoSGBM.create(0, numDisparities, blockSize, P1=P1, P2=P2,
                                   preFilterCap=preFilterCap,
                                   uniquenessRatio=uniquenessRatio,
                                   disp12MaxDiff=disp12MaxDiff,
                                   speckleWindowSize=speckle_window_size,
                                   speckleRange=speckle_range,
                                   mode=mode)

    # stereo = cv2.StereoSGBM.create(0, numDisparities, blockSize, P1=P1, P2=P2, mode=mode)

    disp = stereo.compute(img1, img2)
    disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    disp_nor = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_eql = cv2.equalizeHist(disp_nor.astype('uint8'), None)

    filename = 'D{}_{}'.format(img_name1, img_name2)
    print('Output file name: {}'.format(filename))
    cv2.imwrite('data/taipei/sgm/{}.tif'.format(filename), disp_eql)
    tools.save(disp, 'data/taipei/sgm/{}.dpkl'.format(filename))
    tools.toc()

# hist = cv2.calcHist([disp_nor], [0], None, [256], [0, 256])
# plt.figure()
# plt.plot(hist)
# plt.figure()
# plt.imshow(disp_nor, cmap='gray', vmin=0, vmax=256)
# plt.show()

# vmax = disp.max()
# vmin = disp.min()
# plt.imshow(disp, cmap='gray', vmin=vmin, vmax=vmax)
# plt.show()


