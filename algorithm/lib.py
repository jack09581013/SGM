import cv2
import numpy as np

def normalize(img):
    _min = img.min()
    _max = img.max()
    return ((img - _min) / (_max - _min) * 255).astype('uint8')

def rotate(image):
    rotated = np.zeros((image.shape[1], image.shape[0], 3), dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rotated[image.shape[1] - j - 1, i, :] = image[i, j, :]
    return rotated

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2 lines - corresponding epilines '''
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1, tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2, tuple(pt2),5,color,-1)
    return img1, img2

def boundary(I):
    b_north = 0
    b_south = I.shape[0]-1
    b_east = 0
    b_west = I.shape[1]-1

    for i in range(I.shape[0]):
        if (I[i, :] == 0).all():
            b_north = i
        else:
            break

    for i in range(I.shape[0]-1, -1, -1):
        if (I[i, :] == 0).all():
            b_south = i
        else:
            break

    for i in range(I.shape[1]):
        if (I[:, i] == 0).all():
            b_east = i
        else:
            break

    for i in range(I.shape[1]-1, -1, -1):
        if (I[:, i] == 0).all():
            b_west = i
        else:
            break

    return b_north, b_south, b_east, b_west

def draw_boundary(I, b_north, b_south, b_east, b_west):
    I = cv2.line(I, (0, b_north), (I.shape[1], b_north), (0, 0, 255), 8)
    I = cv2.line(I, (0, b_south), (I.shape[1], b_south), (0, 0, 255), 8)
    I = cv2.line(I, (b_east, 0), (b_east, I.shape[0]), (0, 0, 255), 8)
    I = cv2.line(I, (b_west, 0), (b_west, I.shape[0]), (0, 0, 255), 8)
    return I

def rstr(R, print_flag=True):
    string = '['
    for row in R:
        for e in row:
            string += str(e) + ' '
        string = string[:-1] + '\n'
    string = string[:-1] + ']'

    if print_flag: print(string)
    return string

# print('Find boundary')
# b_north_1, b_south_1, b_east_1, b_west_1 = boundary(img1_rect)
# b_north_2, b_south_2, b_east_2, b_west_2 = boundary(img2_rect)
#
# b_north = min(b_north_1, b_north_2)
# b_south = max(b_south_1, b_south_2)
# b_east = min(b_east_1, b_east_2)
# b_west = max(b_west_1, b_west_2)
#
# img1_rect = img1_rect[b_north:b_south, b_east:b_west]
# img2_rect = img2_rect[b_north:b_south, b_east:b_west]
