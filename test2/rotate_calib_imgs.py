import cv2

for img_no in range(1, 30):
    for side in ['left', 'right']:
        filename = '{}{}'.format(side, img_no)
        print('filename', filename)
        image = cv2.imread('data/calib_imgs/{}.jpg'.format(filename))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite('data/calib_imgs/rotate/{}.jpg'.format(filename), image)