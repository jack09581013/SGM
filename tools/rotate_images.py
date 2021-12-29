from algorithm.map_relation import *

for i in range(4747, 4767):
    img_name = str(i)

    print('Process image:', img_name)

    img = cv2.imread('data/taipei/images/180808-{}.tif'.format(img_name))
    img_r = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite('data/taipei/images/rotate/180808-{}-r.tif'.format(img_name), img_r)

