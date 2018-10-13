"""
  @Time    : 2018-10-13 23:48
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : edge.py
  @Function: 
  
"""
import os
import numpy as np
import cv2

DATA_DIR = "/home/taylor/mirror/augmentation/train"
IMAGE_DIR = os.path.join(DATA_DIR, "image")

imglist = os.listdir(IMAGE_DIR)
print("Total {} masks will be computed!".format(len(imglist)))

for imgname in imglist:
    img_path = IMAGE_DIR + "/" + imgname
    edge_path = DATA_DIR + "/mask/" + imgname[:-4] + "_json/edge.png"

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(edge_path[:-8]+"image.png", image)
    height, width = image.shape

    edge = np.zeros([height, width, 1])
    image_padded = np.pad(image, [(1, 1), (1, 1)], mode="edge")
    for y in range(height):
        for x in range(width):
            left = np.abs(float(image_padded[y + 1, x + 1]) - float(image_padded[y + 1, x]))
            top = np.abs(float(image_padded[y + 1, x + 1]) - float(image_padded[y, x + 1]))
            right = np.abs(float(image_padded[y + 1, x + 1]) - float(image_padded[y + 1, x + 2]))
            bottom = np.abs(float(image_padded[y + 1, x + 1]) - float(image_padded[y + 2, x + 1]))
            edge[y, x] = (left + top + right + bottom)/4

    cv2.imwrite(edge_path, edge)
    print(edge_path)
