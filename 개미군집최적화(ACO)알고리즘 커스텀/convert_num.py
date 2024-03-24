# ================================================
# convert_num.py
# - 자주 사용하는 함수
# ================================================

import numpy as np
import matplotlib as plot
import cv2
# from Grid_Classify import WIDTH, HEIGHT, img


WIDTH = 103
HEIGHT = 90
img = cv2.imread("section.png", 1)
img = cv2.resize(dsize=(WIDTH, HEIGHT), src=img)

# <editor-fold desc="num2coord">
# 그리드 번호를 좌표로 매핑할 배열
# : num2coord['그리드 번호'] = 좌표의 y, x값 반환
num2coord = []
for x in range(WIDTH):
    for y in range(HEIGHT):
        if img[y][x][0] != 255 or img[y][x][1] != 255 or img[y][x][2] != 255:
            num2coord.append([y, x])

# </editor-fold>

# <editor-fold desc="coord2num">
# 좌표 번호를 그리드 좌표로 반환
# : coord2num['y 좌표']['x 좌표'] = 그리드 번호 반환
# : 해당 좌표가 배경이라면 -1 반환
coord2num = np.full((HEIGHT, WIDTH), -1)
idx = 0
for x in range(WIDTH):
    for y in range(HEIGHT):
        if img[y][x][0] != 255 or img[y][x][1] != 255 or img[y][x][2] != 255:
            coord2num[y][x] = idx
            idx += 1
# </editor-fold>

# <editor-fold desc="컨벌루션 함수">
def conv(image, kernel, padding=1, strides=1):
    xImgShape, yImgShape = image.shape  # 103, 90
    xKernShape, yKernShape = kernel.shape  # 3, 3

    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)  # 103
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)  # 90
    output = np.zeros((xOutput, yOutput))

    if padding != 0:
        imagePadded = np.zeros((xImgShape + padding * 2, yImgShape + padding * 2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image
    for y in range(yImgShape):
        if y % strides == 0:
            for x in range(xImgShape):
                try:
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return output
# </editor-fold>